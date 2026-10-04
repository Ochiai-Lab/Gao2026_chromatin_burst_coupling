"""Explicit CSV-boundary compatibility for transcription-state labels."""

from __future__ import annotations

import pandas as pd


STATE_COLUMNS = ("state", "raw_state", "kind")
CANONICAL_LABELS = {"ON": "Active", "OFF": "Inactive"}
LEGACY_LABELS = {"Active": "ON", "Inactive": "OFF"}


def export_transcription_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with publication labels, preserving all other values."""
    result = frame.copy()
    for column in STATE_COLUMNS:
        if column in result.columns:
            result[column] = result[column].replace(CANONICAL_LABELS)
    return result


def read_legacy_state_csv(*args, **kwargs) -> pd.DataFrame:
    """Accept either label convention without changing legacy analysis logic."""
    frame = pd.read_csv(*args, **kwargs)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("State-aware CSV reads require a DataFrame, not a chunk iterator")
    for column in STATE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].replace(LEGACY_LABELS)
    return frame
