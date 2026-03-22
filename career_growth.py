from employability import predict_employability


def simulate_career_growth(model, user_vector, new_skills, feature_names, role_vectors):

    updated_vector = user_vector.copy()

    for skill in new_skills:

        if skill in feature_names:
            idx = feature_names.index(skill)
            updated_vector[idx] = 1

    simulated_scores = {}

    for role in role_vectors.keys():

        score, _, _ = predict_employability(
            model,
            updated_vector,
            role,
            role_vectors,
            feature_names
        )

        simulated_scores[role] = score / 100

    return simulated_scores
