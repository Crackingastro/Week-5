import pytest
import joblib
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.pipline import create_features

MODEL_DIR = Path(__file__).parent.parent / "src"/"model"

@pytest.fixture(scope="module")
def pipeline():
    """Load the fitted preprocessing pipeline."""
    return joblib.load(MODEL_DIR / "pipeline.pkl")

def test_input_validation(pipeline):
    """
    Supplying a DataFrame missing required input columns should raise an error.
    """
    # completely wrong DataFrame
    bad = pd.DataFrame([{"foo": 1, "bar": 2}])
    with pytest.raises(Exception):
        pipeline.transform(bad)
