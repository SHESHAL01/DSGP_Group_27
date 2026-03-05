def simulate_career_growth(model, user_vector, new_skills, feature_names):

    updated_vector = user_vector.copy()

    for skill in new_skills:

        if skill in feature_names:

            idx = feature_names.index(skill)
            updated_vector[idx] = 1

    probs = model.predict_proba([updated_vector])[0]
    classes = model.classes_

    role_probs = dict(zip(classes, probs))

    return role_probs
