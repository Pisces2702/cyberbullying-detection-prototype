import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import clean_text

def test_clean_text():
    assert clean_text('Hello @User!!! #test') == 'hello user test'
