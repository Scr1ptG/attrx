from pathlib import Path

import pytest

from attrx.schema import Schema

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="session")
def schema() -> Schema:
    return Schema.load()
