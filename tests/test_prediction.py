from src.predict import clean_text

def test_clean_text():
    assert clean_text('Hello @User!!! #test') == 'hello user test'
