def test_clean_text():
    raw_text = "<b>Looking</b> for 5+ years of C++ & Java!"
    expected = "looking for years of c java"
    assert clean_text(raw_text) == expected

def test_normalize():
    raw_text = "Seeking experience in NodeJS, ReactJS, and object-oriented design."
    expected = "seeking experience in node js, react, and object oriented design "
    assert normalize(raw_text) == expected

def test_auto_label_single_word():
    # Assuming SKILL_SET contains "python"
    tokens, tags = auto_label("expert in python programming")
    assert tokens == ["expert", "in", "python", "programming"]
    assert tags == ["O", "O", "B-SKILL", "O"]

def test_auto_label_multi_word():
    # Assuming SKILL_SET contains "machine learning"
    tokens, tags = auto_label("machine learning engineer")
    assert tags == ["B-SKILL", "I-SKILL", "O"]

def test_extract_skills_subword_stitching(monkeypatch):
    # Mock pipeline output for the word "kotlin" broken into "ko" and "##tlin"
    mock_pipeline_output = [
        {'entity': 'B-SKILL', 'word': 'ko', 'start': 10, 'end': 12},
        {'entity': 'I-SKILL', 'word': '##tlin', 'start': 12, 'end': 16}
    ]

    # You would mock `ner_pipe` here to return mock_pipeline_output
    extracted = extract_skills("I know kotlin")
    assert "kotlin" in extracted
    assert "ko" not in extracted
    assert "tlin" not in extracted

def test_extract_skills_ignores_single_characters():
    # Mock pipeline output returning a valid skill and a 1-character hallucination
    mock_pipeline_output = [
        {'entity': 'B-SKILL', 'word': 'java', 'start': 0, 'end': 4},
        {'entity': 'B-SKILL', 'word': 'c', 'start': 10, 'end': 11} # Should be ignored if length > 1 is enforced strictly, note: 'c' is in your dictionary, so you may need to reconsider the >1 rule if you want to extract 'C' or 'R'
    ]
    # Test logic ensures length > 1
    extracted = extract_skills("Mock text")
    assert "java" in extracted
    assert "c" not in extracted # Based on your current extract_skills implementation