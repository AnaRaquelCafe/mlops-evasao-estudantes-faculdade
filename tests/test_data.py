import pandas as pd
import pytest
from src.evasao.data import detect_target_column, validate_dataset

def test_detect_target():
    df = pd.DataFrame({"feature": [1, 2], "Target": ["Dropout", "Graduate"]})
    assert detect_target_column(df) == "Target"

def test_validate_rejects_missing_target():
    df = pd.DataFrame({"Target": ["Dropout", None]})
    with pytest.raises(ValueError):
        validate_dataset(df, "Target")
