"""Bootstrap smoke tests for the initial compute-plane package layout."""


def test_compute_plane_packages_import() -> None:
    import app.adapters
    import app.api
    import app.domain
    import app.ingestion
    import app.kernel
    import app.orchestration
    import app.plugin_sdk
    import app.plugins
    import app.reports
    import app.telemetry
    import app.validation

    assert app.api is not None
