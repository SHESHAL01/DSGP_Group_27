import numpy as np
from utils import build_user_vector


def test_build_user_vector():

    feature_names = ["python", "java", "sql", "docker"]

    user_skills = ["python", "docker"]

    vector = build_user_vector(user_skills, feature_names)

    expected = np.array([1, 0, 0, 1])

    assert np.array_equal(vector, expected)


def test_unknown_skill():

    feature_names = ["python", "java", "sql"]

    user_skills = ["python", "aws"]

    vector = build_user_vector(user_skills, feature_names)

    expected = np.array([1, 0, 0])

    assert np.array_equal(vector, expected)


