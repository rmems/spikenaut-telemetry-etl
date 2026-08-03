"""Spikenaut telemetry ETL.

Cleans and validates blockchain mining / GPU / HFT telemetry between
``rmems/Theseus-Quarry`` (Rust collectors) and the published Hugging Face dataset
``rmems/Spikenaut-SNN-Telemetry``.

Its reason for existing: the previous cleaning step silently emitted 813,973
identical empty records and 120,314 all-zero rows, then reported success. Every
module here is built so that failure is loud.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schemas import OUTPUT_SCHEMA_VERSION
from .validate import ValidationError, assert_publishable, check_all

__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "ValidationError",
    "assert_publishable",
    "check_all",
    "__version__",
]
