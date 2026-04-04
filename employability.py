import numpy as np

def predict_employability(model, user_vector, preferred_role, role_vectors, feature_names, threshold=40):

    role_vector = role_vectors[preferred_role]

    # Filter irrelevant skills

    filtered_vector = np.array(user_vector) * (np.array(role_vector) > 0.3)

    # ===============================
    # Preferred role prediction
    # ===============================

    probs_filtered = model.predict_proba([filtered_vector])[0]
    classes = model.classes_

    role_probs_filtered = dict(zip(classes, probs_filtered))

    model_score = role_probs_filtered.get(preferred_role, 0)

    # ===============================
    # Skill match score
    # ===============================

    relevant_skills = np.sum(role_vector > 0.3)

    matched_skills = np.sum(
        (filtered_vector == 1) & (np.array(role_vector) > 0.3)
    )

    if relevant_skills > 0:
        skill_match_score = matched_skills / relevant_skills
    else:
        skill_match_score = 0

    final_score = (0.7 * skill_match_score + 0.3 * model_score) * 100

    # ===============================
    # Alternative roles
    # ===============================

    alternative_roles = []

    if final_score < threshold:

        role_scores = []

        for role in classes:
            role_vector = role_vectors[role]

            # filter skills relevant to that role
            filtered_vector = np.array(user_vector) * (np.array(role_vector) > 0.3)

            probs = model.predict_proba([filtered_vector])[0]
            role_prob = dict(zip(classes, probs)).get(role, 0)

            role_scores.append((role, role_prob))

        sorted_roles = sorted(role_scores, key=lambda x: x[1], reverse=True)

        for role, prob in sorted_roles:

            if role != preferred_role:
                alternative_roles.append((role, prob * 100))

            if len(alternative_roles) == 2:
                break

    return final_score, role_probs_filtered, alternative_roles


