import numpy as np
import pandas as pd

def compute_role_skill_vectors(dataset, feature_names, target_column):

    role_vectors = {}

    roles = dataset[target_column].unique()

    for role in roles:

        role_data = dataset[dataset[target_column] == role]

        role_vectors[role] = role_data[feature_names].mean().values

    return role_vectors


def skill_gap_analysis(user_vector, role_vector, feature_names):

    gaps = role_vector - user_vector

    missing_skills = []

    for i, value in enumerate(gaps):

        if value > 0.3:   # threshold

            missing_skills.append(
                (feature_names[i], round(value,3))
            )

    missing_skills.sort(key=lambda x: x[1], reverse=True)

    return missing_skills
