import pytest

from src.data_loader import parse_timestamps, read_raw


@pytest.fixture(scope="session")
def clean():
    """Cleaned dataframe built directly from the raw file."""
    return parse_timestamps(read_raw())
