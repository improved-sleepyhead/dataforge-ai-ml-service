"""Export-time tabular and text/OCR writers (TASK-054).

These writers consume immutable candidate / source artifacts and
produce export-grade artifacts that the ``ExportPackage`` builder
references:

- ``write_tabular_export`` — Parquet primary and optional CSV outputs
  per overall candidate and per train/validation/test split.
- ``write_text_ocr_export`` — redacted JSONL for restricted text/OCR
  sources (raw PII never reaches the export artifact).

Writers never mutate raw artifacts. They register every output through
``ArtifactRegistry`` so the ExportPackage carries deterministic,
content-addressed hashes.
"""

from app.plugins.export.tabular_writer import (
    TABULAR_EXPORT_CSV_FORMAT,
    TABULAR_EXPORT_CSV_KIND,
    TABULAR_EXPORT_CSV_MEDIA_TYPE,
    TABULAR_EXPORT_CSV_SCHEMA_VERSION,
    TABULAR_EXPORT_PARQUET_FORMAT,
    TABULAR_EXPORT_PARQUET_KIND,
    TABULAR_EXPORT_PARQUET_MEDIA_TYPE,
    TABULAR_EXPORT_PARQUET_SCHEMA_VERSION,
    TabularExportArtifacts,
    TabularExportPerSplitArtifact,
    TabularExportRequest,
    TabularExportWriterError,
    write_tabular_export,
)
from app.plugins.export.text_ocr_writer import (
    TEXT_OCR_EXPORT_FORMAT,
    TEXT_OCR_EXPORT_KIND,
    TEXT_OCR_EXPORT_MEDIA_TYPE,
    TEXT_OCR_EXPORT_SCHEMA_VERSION,
    TextOcrExportArtifacts,
    TextOcrExportRequest,
    TextOcrExportWriterError,
    write_text_ocr_export,
)

__all__ = [
    "TABULAR_EXPORT_CSV_FORMAT",
    "TABULAR_EXPORT_CSV_KIND",
    "TABULAR_EXPORT_CSV_MEDIA_TYPE",
    "TABULAR_EXPORT_CSV_SCHEMA_VERSION",
    "TABULAR_EXPORT_PARQUET_FORMAT",
    "TABULAR_EXPORT_PARQUET_KIND",
    "TABULAR_EXPORT_PARQUET_MEDIA_TYPE",
    "TABULAR_EXPORT_PARQUET_SCHEMA_VERSION",
    "TEXT_OCR_EXPORT_FORMAT",
    "TEXT_OCR_EXPORT_KIND",
    "TEXT_OCR_EXPORT_MEDIA_TYPE",
    "TEXT_OCR_EXPORT_SCHEMA_VERSION",
    "TabularExportArtifacts",
    "TabularExportPerSplitArtifact",
    "TabularExportRequest",
    "TabularExportWriterError",
    "TextOcrExportArtifacts",
    "TextOcrExportRequest",
    "TextOcrExportWriterError",
    "write_tabular_export",
    "write_text_ocr_export",
]
