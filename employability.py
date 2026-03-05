import numpy as np

def predict_employability(model, user_vector, preferred_role, threshold=40):

    probs = model.predict_proba([user_vector])[0]
    classes = model.classes_

    role_probs = dict(zip(classes, probs))

    # Employability score for preferred role
    if preferred_role in role_probs:
        employability_score = role_probs[preferred_role] * 100
    else:
        employability_score = 0

    alternative_roles = []

    # If score is low, suggest alternatives
    if employability_score < threshold:

        # Sort roles by probability
        sorted_roles = sorted(role_probs.items(), key=lambda x: x[1], reverse=True)

        for role, prob in sorted_roles:
            if role != preferred_role:
                alternative_roles.append((role, prob * 100))

            if len(alternative_roles) == 2:
                break

    return employability_score, role_probs, alternative_roles
