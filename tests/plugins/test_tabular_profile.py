"""Tests for TASK-023: tabular schema inference and base profile report.

Acceptance criteria covered:

* profiler computes ``row_count``, ``column_count``, schema, nullability;
* detects ``target_column``, ``group_key_columns``, id-like and PII-like
  columns;
* report saved as ``tabular_profile_report`` immutable artifact;
* report ``ArtifactRef`` is contract-compatible with
  ``DataForgeReport.detail_artifacts``.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ColumnRole,
    ColumnType,
    ErrorCode,
    TabularProfileReport,
)
from app.ingestion import (
    BuildManifestRequest,
    build_asset_manifest,
    build_validated_manifest,
    open_archive_path,
)
from app.plugins.tabular import (
    PROFILE_REPORT_KIND,
    PROFILE_REPORT_SCHEMA_VERSION,
    ProfileBuildRequest,
    build_tabular_profile_report,
    infer_tabular_profile,
)
from app.validation.contracts import (
    load_contract_pack,
    validate_contract_payload,
)
from tests.fixtures.demo_archive import build_demo_archive

_MANIFEST_REQUEST = BuildManifestRequest(
    dataset_id="dataset_demo",
    version_id="version_demo",
    parent_version_id="version_demo_parent",
    created_by_job_id="compute_run_profile",
    config_hash="sha256:" + "a" * 64,
)
_PROFILE_REQUEST = ProfileBuildRequest(
    dataset_id="dataset_demo",
    version_id="version_demo",
    parent_version_id="version_demo_parent",
    created_by_job_id="compute_run_profile",
    config_hash="sha256:" + "a" * 64,
    source_artifact_id="raw_archive:demo",
    source_system="transactions",
)


# ---------------------------------------------------------------------------
# Step 1: run profiler on demo transactions.csv
# ---------------------------------------------------------------------------


def test_profiler_runs_on_demo_archive_and_produces_artifact(
    tmp_path: Path,
) -> None:
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report
    assert isinstance(report, TabularProfileReport)
    assert report.row_count == 200
    assert report.column_count == 8
    assert report.profile_schema_version == "tabular_profile_report.v1"
    assert report.source_system == "transactions"
    assert result.artifact.artifact_kind == PROFILE_REPORT_KIND
    assert result.artifact.schema_version == PROFILE_REPORT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Step 2: target = is_fraud, group key = customer_id_hash, schema/nullability
# ---------------------------------------------------------------------------


def test_profiler_detects_target_group_key_and_pii_like(tmp_path: Path) -> None:
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report
    assert report.target_column == "is_fraud"
    assert "customer_id_hash" in report.group_key_columns
    assert "object_id" in report.id_columns
    # Demo transactions.csv has no PII-like column names.
    assert report.pii_like_columns == ()

    columns_by_name = {col.name: col for col in report.columns}
    target = columns_by_name["is_fraud"]
    assert target.role is ColumnRole.TARGET
    assert target.type is ColumnType.BOOLEAN

    customer_hash = columns_by_name["customer_id_hash"]
    assert customer_hash.role is ColumnRole.GROUP_KEY

    monthly_income = columns_by_name["monthly_income"]
    assert monthly_income.nullable is True
    assert monthly_income.null_count > 0
    assert 0.0 < monthly_income.null_ratio < 1.0
    assert monthly_income.type is ColumnType.NUMERIC_FLOAT

    leakage = columns_by_name["manual_review_flag"]
    assert leakage.role is ColumnRole.LEAKAGE_CANDIDATE
    assert leakage.type is ColumnType.BOOLEAN


def test_pii_like_column_names_are_detected_and_sample_value_redacted() -> None:
    rows = [
        {"object_id": "u1", "email": "alex@example.test", "phone": "+10000001234"},
        {"object_id": "u2", "email": "lena@example.test", "phone": "+10000005678"},
    ]
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "email", "phone"),
        request=_PROFILE_REQUEST,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_pii_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )

    pii_columns = {col.name for col in report.columns if col.role is ColumnRole.PII_LIKE}
    assert pii_columns == {"email", "phone"}
    for col in report.columns:
        if col.role is ColumnRole.PII_LIKE:
            assert col.sample_value is None
        else:
            assert col.sample_value is not None


# ---------------------------------------------------------------------------
# TASK-024: missingness diagnostics
# ---------------------------------------------------------------------------


def test_missingness_monthly_income_in_expected_range(tmp_path: Path) -> None:
    """Step 1+2: profiler computes missing_rate for monthly_income in demo data."""
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report
    assert report.missingness is not None
    missingness = report.missingness

    # Target column detected and has no missing values in demo data.
    assert missingness.target_column == "is_fraud"
    assert missingness.target_column_missing is False
    assert missingness.missing_target_count == 0

    # Segment column detected.
    assert missingness.segment_column == "customer_segment"

    # monthly_income has ~45 missing values (39 in-segment + 6 outside);
    # the duplicate-row injection can overwrite a small number of them, so
    # accept a small range around the expected_counts value.
    col_map = {cm.column: cm for cm in missingness.columns}
    income = col_map["monthly_income"]
    assert 35 <= income.missing_count <= 50
    assert income.total_count == 200
    assert 0.17 <= income.missing_rate <= 0.25

    # by_target: missingness conditioned on is_fraud values.
    assert income.by_target is not None
    assert income.by_target.group_column == "is_fraud"
    assert "0" in income.by_target.groups
    assert "1" in income.by_target.groups

    # by_segment: young_customers has higher missing-income ratio than
    # the regular_customers segment (the demo intentionally injects more
    # missingness in young_customers).
    assert income.by_segment is not None
    assert income.by_segment.group_column == "customer_segment"
    young = income.by_segment.groups["young_customers"]
    regular = income.by_segment.groups["regular_customers"]
    assert young.missing_ratio > regular.missing_ratio
    assert young.missing_ratio >= 0.4


def test_missing_target_produces_hard_blocker_candidate() -> None:
    """Step 3: fixture with missing target -> target_column_missing=True."""
    rows = [
        {"object_id": "r1", "is_fraud": "1", "amount": "100"},
        {"object_id": "r2", "is_fraud": "", "amount": "200"},  # missing target
        {"object_id": "r3", "is_fraud": "0", "amount": "300"},
    ]
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud", "amount"),
        request=_PROFILE_REQUEST,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_missing_target",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )

    assert report.missingness is not None
    assert report.missingness.target_column == "is_fraud"
    assert report.missingness.target_column_missing is True
    assert report.missingness.missing_target_count == 1


def test_missingness_no_segment_column() -> None:
    """When segment_column is None, by_segment is None for all columns."""
    rows = [
        {"object_id": "r1", "is_fraud": "1", "value": "10"},
        {"object_id": "r2", "is_fraud": "0", "value": ""},
    ]
    request = ProfileBuildRequest(
        dataset_id="dataset_demo",
        version_id="version_demo",
        parent_version_id="version_demo_parent",
        created_by_job_id="compute_run_profile",
        config_hash="sha256:" + "a" * 64,
        source_artifact_id="raw_archive:demo",
        source_system="transactions",
        segment_column=None,
    )
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud", "value"),
        request=request,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_no_segment",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )

    assert report.missingness is not None
    assert report.missingness.segment_column is None
    for cm in report.missingness.columns:
        assert cm.by_segment is None


# ---------------------------------------------------------------------------
# TASK-025: duplicate / outlier / class imbalance / leakage diagnostics
# ---------------------------------------------------------------------------


def test_duplicates_outliers_imbalance_leakage_on_demo(tmp_path: Path) -> None:
    """Step 1+2+3+4+5: profiler reports duplicate count, rare class ratio,
    leakage candidate reason code, effective number of samples, and
    BalanceScore decomposition on demo data."""
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report

    # Step 2: duplicate row pairs and rare class ratio.
    assert report.duplicates is not None
    assert report.duplicates.duplicate_pair_count >= 4
    assert len(report.duplicates.affected_object_ids) >= 4
    assert "object_id" not in report.duplicates.signature_columns
    assert report.duplicates.id_column == "object_id"

    # Class imbalance: rare class is fraud=1 with ratio 0.02.
    assert report.class_imbalance is not None
    ci = report.class_imbalance
    assert ci.target_column == "is_fraud"
    assert ci.rare_class_label == "1"
    assert ci.rare_class_count == 4
    assert abs(ci.rare_class_ratio - 0.02) < 1e-9
    # Step 5: minority_share, imbalance_ratio, BalanceScore decomposition.
    assert ci.minority_class_share == ci.rare_class_ratio
    assert abs(ci.imbalance_ratio - 49.0) < 1e-9  # 196/4
    assert ci.balance_score_formula == "1 / log(1 + imbalance_ratio)"
    expected_balance = 1.0 / math.log1p(49.0)
    assert abs(ci.balance_score - expected_balance) < 1e-6
    assert (
        ci.balance_score_alternative_formula == "min_c(n_c) / mean_c(n_c)"
    )
    assert abs(ci.balance_score_alternative - 4 / (200 / 2)) < 1e-9
    # Step 4: effective_number_of_samples for rare class follows the
    # E_n = (1 - β^n) / (1 - β) closed form.
    beta = ci.effective_number_beta
    assert 0.0 < beta < 1.0
    expected_rare = (1 - beta**ci.rare_class_count) / (1 - beta)
    assert abs(ci.effective_number_of_samples["1"] - expected_rare) < 1e-6
    # Effective n for the rare class must be < its raw count (down-weighted).
    assert ci.effective_number_of_samples["1"] < ci.rare_class_count

    # Step 3: leakage candidate reason code.
    assert report.leakage is not None
    leakage_columns = {c.column for c in report.leakage.candidates}
    assert "manual_review_flag" in leakage_columns
    manual_review = next(
        c for c in report.leakage.candidates if c.column == "manual_review_flag"
    )
    assert manual_review.reason_code in {
        "high_target_match_rate",
        "name_pattern_leakage_candidate",
    }


def test_duplicate_diagnostics_on_synthetic_rows() -> None:
    """Two rows with identical non-id payload yield a duplicate pair."""
    rows = [
        {"object_id": "r1", "is_fraud": "0", "amount": "100.00"},
        {"object_id": "r2", "is_fraud": "0", "amount": "100.00"},  # duplicate of r1
        {"object_id": "r3", "is_fraud": "1", "amount": "200.00"},
    ]
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud", "amount"),
        request=_PROFILE_REQUEST,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_dup_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )
    assert report.duplicates is not None
    assert report.duplicates.duplicate_pair_count == 1
    assert report.duplicates.duplicate_group_count == 1
    assert set(report.duplicates.affected_object_ids) == {"r1", "r2"}


def test_outlier_detection_on_synthetic_amount_column() -> None:
    """IQR-based outlier detection flags far-away values."""
    rows = [
        {"object_id": f"r{i}", "is_fraud": "0", "amount": str(100 + i)}
        for i in range(30)
    ]
    rows.append(
        {"object_id": "r_outlier", "is_fraud": "0", "amount": "100000"}
    )
    request = ProfileBuildRequest(
        dataset_id="dataset_demo",
        version_id="version_demo",
        parent_version_id="version_demo_parent",
        created_by_job_id="compute_run_profile",
        config_hash="sha256:" + "a" * 64,
        source_artifact_id="raw_archive:demo",
        source_system="transactions",
        outlier_columns=("amount",),
    )
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud", "amount"),
        request=request,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_outlier_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )
    assert report.outliers is not None
    amount_stats = next(c for c in report.outliers.columns if c.column == "amount")
    assert amount_stats.outlier_count >= 1
    assert "r_outlier" in amount_stats.affected_object_ids


def test_class_imbalance_decomposition_on_synthetic_target() -> None:
    """Imbalance decomposition matches the documented formulas."""
    # 9 zeros, 1 one — N=10, min=1, max=9, imbalance_ratio=9.
    rows = [{"object_id": f"r{i}", "is_fraud": "0"} for i in range(9)]
    rows.append({"object_id": "r9", "is_fraud": "1"})
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud"),
        request=_PROFILE_REQUEST,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_imbalance_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )
    assert report.class_imbalance is not None
    ci = report.class_imbalance
    assert ci.total_samples == 10
    assert ci.imbalance_ratio == 9.0
    assert abs(ci.minority_class_share - 0.1) < 1e-9
    expected_balance = 1.0 / math.log1p(9.0)
    assert abs(ci.balance_score - expected_balance) < 1e-9
    # Alternative: rare_count / mean_count = 1 / (10/2) = 0.2.
    assert abs(ci.balance_score_alternative - 0.2) < 1e-9
    beta = ci.effective_number_beta
    assert abs(
        ci.effective_number_of_samples["1"] - (1 - beta**1) / (1 - beta)
    ) < 1e-9


# ---------------------------------------------------------------------------
# TASK-026: business rule validation
# ---------------------------------------------------------------------------


def test_business_rule_amount_non_negative_passes_on_demo(tmp_path: Path) -> None:
    """Step 1+2+3: profiler runs amount>=0 rule on demo data and reports
    zero violations, with rules_config_hash present in the report."""
    from app.domain import BusinessRuleSeverity
    from app.plugins.tabular.rules import (
        BusinessRule,
        RuleFieldCheck,
        compute_rules_config_hash,
    )

    rule = BusinessRule(
        rule_id="amount_must_be_non_negative",
        description="transaction amount must be >= 0",
        severity=BusinessRuleSeverity.CRITICAL,
        columns=("amount",),
        checks=(RuleFieldCheck(field="amount", op="gte", value=0),),
        message="transaction amount cannot be negative",
    )
    expected_hash = compute_rules_config_hash((rule,))

    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)
    request = ProfileBuildRequest(
        dataset_id="dataset_demo",
        version_id="version_demo",
        parent_version_id="version_demo_parent",
        created_by_job_id="compute_run_profile",
        config_hash="sha256:" + "a" * 64,
        source_artifact_id="raw_archive:demo",
        source_system="transactions",
        business_rules=(rule,),
    )

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=request,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report
    assert report.business_rules is not None
    br = report.business_rules
    assert br.rules_version == "tabular_business_rules.v1"
    assert br.rules_config_hash == expected_hash
    assert len(br.rules) == 1
    summary = br.rules[0]
    assert summary.rule_id == "amount_must_be_non_negative"
    assert summary.severity is BusinessRuleSeverity.CRITICAL
    assert summary.evaluated_count == 200
    assert summary.violation_count == 0
    assert summary.pass_rate == 1.0
    assert br.blocker_candidate_rule_ids == ()


def test_critical_business_rule_violation_creates_blocker_candidate() -> None:
    """A critical rule with violations adds the rule_id to
    blocker_candidate_rule_ids (BLOCK_RULES_REVIEW signal)."""
    from app.domain import BusinessRuleSeverity
    from app.plugins.tabular.rules import BusinessRule, RuleFieldCheck

    rule = BusinessRule(
        rule_id="amount_must_be_non_negative",
        severity=BusinessRuleSeverity.CRITICAL,
        columns=("amount",),
        checks=(RuleFieldCheck(field="amount", op="gte", value=0),),
        message="transaction amount cannot be negative",
    )
    rows = [
        {"object_id": "r1", "is_fraud": "0", "amount": "100"},
        {"object_id": "r2", "is_fraud": "1", "amount": "-50"},
        {"object_id": "r3", "is_fraud": "0", "amount": "0"},
    ]
    request = ProfileBuildRequest(
        dataset_id="dataset_demo",
        version_id="version_demo",
        parent_version_id="version_demo_parent",
        created_by_job_id="compute_run_profile",
        config_hash="sha256:" + "a" * 64,
        source_artifact_id="raw_archive:demo",
        source_system="transactions",
        business_rules=(rule,),
    )
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud", "amount"),
        request=request,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_blocker_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )
    assert report.business_rules is not None
    br = report.business_rules
    summary = br.rules[0]
    assert summary.violation_count == 1
    assert summary.evaluated_count == 3
    assert abs(summary.pass_rate - 2 / 3) < 1e-9
    assert "amount_must_be_non_negative" in br.blocker_candidate_rule_ids
    # Sample violation includes the offending object_id and the column.
    sample_ids = {v.object_id for v in br.sample_violations}
    assert sample_ids == {"r2"}
    assert br.sample_violations[0].column == "amount"
    assert br.sample_violations[0].rule_id == "amount_must_be_non_negative"


def test_warning_business_rule_does_not_block() -> None:
    """A warning-severity rule with violations does NOT produce a blocker."""
    from app.domain import BusinessRuleSeverity
    from app.plugins.tabular.rules import BusinessRule, RuleFieldCheck

    rule = BusinessRule(
        rule_id="amount_under_million",
        severity=BusinessRuleSeverity.WARNING,
        columns=("amount",),
        checks=(RuleFieldCheck(field="amount", op="lt", value=1_000_000),),
    )
    rows = [
        {"object_id": "r1", "is_fraud": "0", "amount": "100"},
        {"object_id": "r2", "is_fraud": "0", "amount": "5000000"},
    ]
    request = ProfileBuildRequest(
        dataset_id="dataset_demo",
        version_id="version_demo",
        parent_version_id="version_demo_parent",
        created_by_job_id="compute_run_profile",
        config_hash="sha256:" + "a" * 64,
        source_artifact_id="raw_archive:demo",
        source_system="transactions",
        business_rules=(rule,),
    )
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "is_fraud", "amount"),
        request=request,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_warning_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )
    assert report.business_rules is not None
    br = report.business_rules
    assert br.rules[0].violation_count == 1
    assert br.blocker_candidate_rule_ids == ()


def test_rules_config_hash_is_deterministic() -> None:
    """Same rule list produces the same hash regardless of in-memory order."""
    from app.plugins.tabular.rules import (
        BusinessRule,
        RuleFieldCheck,
        compute_rules_config_hash,
    )

    rule_a = BusinessRule(
        rule_id="rule_a",
        checks=(RuleFieldCheck(field="amount", op="gte", value=0),),
    )
    rule_b = BusinessRule(
        rule_id="rule_b",
        checks=(RuleFieldCheck(field="monthly_income", op="gte", value=0),),
    )
    h1 = compute_rules_config_hash((rule_a, rule_b))
    h2 = compute_rules_config_hash((rule_a, rule_b))
    assert h1 == h2
    # Changing rule order changes the hash (rules_config_hash respects order).
    h3 = compute_rules_config_hash((rule_b, rule_a))
    assert h1 != h3


def test_no_business_rules_means_no_business_rules_report(tmp_path: Path) -> None:
    """When the request carries no rules, the report omits business_rules."""
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)
    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )
    assert result.profile_report.business_rules is None


# ---------------------------------------------------------------------------
# Step 3: artifact saved with hash and contract-compatible
# ---------------------------------------------------------------------------


def test_profile_artifact_is_contract_compatible_and_idempotent(
    tmp_path: Path,
) -> None:
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        first = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
            profile_id="tabular_profile_demo",
            generated_at=datetime(2026, 5, 20, tzinfo=UTC),
        )
    with open_archive_path(archive_path) as reader:
        second = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
            profile_id="tabular_profile_demo",
            generated_at=datetime(2026, 5, 20, tzinfo=UTC),
        )

    assert first.artifact.uri == second.artifact.uri
    assert first.artifact.hash == second.artifact.hash

    stored = storage.get(first.artifact.uri)
    assert stored.info.metadata["artifact-kind"] == "tabular_profile_report"
    assert stored.info.metadata["schema-version"] == "tabular_profile_report.v1"

    pack = load_contract_pack()
    payload = first.profile_report.model_dump(mode="json")
    validate_contract_payload(pack, "tabular_profile_report", payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_demo(
    tmp_path: Path,
) -> tuple[
    MinioObjectStorageAdapter,
    ArtifactRegistry,
    Path,
    Path,
]:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_test",
            project_id="project_test",
            dataset_id="dataset_demo",
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    return storage, registry, built.archive_path, built.archive_path


def _validated_manifest(
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    archive_path: Path,
) -> Any:
    with open_archive_path(archive_path) as reader:
        manifest_result = build_asset_manifest(
            reader, request=_MANIFEST_REQUEST, registry=registry
        )
    raw_artifact = manifest_result.manifest_artifact
    return build_validated_manifest(
        raw_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_MANIFEST_REQUEST.version_id,
        parent_version_id=_MANIFEST_REQUEST.parent_version_id,
        created_by_job_id=_MANIFEST_REQUEST.created_by_job_id,
        config_hash=_MANIFEST_REQUEST.config_hash,
        organization_id="org_test",
        project_id="project_test",
    ).validated_manifest


def _synthetic_manifest_artifact_ref() -> Any:
    """Return a synthetic ArtifactRef for PII-name unit tests (no archive)."""
    from app.domain import ArtifactLineage, ArtifactRef

    return ArtifactRef(
        artifact_id="validated_manifest:synthetic",
        kind="validated_manifest",
        uri="s3://dataforge-local/dataforge/org_test/project_test/dataset_demo/manifest.jsonl",
        hash="sha256:" + "f" * 64,
        media_type="application/jsonl",
        size_bytes=1024,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="version_demo",
            job_id="compute_run_profile",
            config_hash="sha256:" + "a" * 64,
            created_at=datetime(2026, 5, 20, tzinfo=UTC),
        ),
    )


class _InMemoryS3Client(S3CompatibleClient):
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(record["Body"])}
                for (bucket, key), record in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
