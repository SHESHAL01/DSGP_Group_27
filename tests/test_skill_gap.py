import numpy as np
from skill_gap import skill_gap_analysis


def test_skill_gap_analysis():

    feature_names = ["python", "java", "sql"]

    user_vector = np.array([1, 0, 0])

    role_vector = np.array([1, 0.8, 0.6])

    gaps = skill_gap_analysis(user_vector, role_vector, feature_names)

    missing_skills = [skill for skill, score in gaps]

    assert "java" in missing_skills
    assert "sql" in missing_skills

