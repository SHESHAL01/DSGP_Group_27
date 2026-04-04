import numpy as np

def build_user_vector(user_skills, feature_names):

    # Normalize dataset feature names
    normalized_features = {
        f.replace("_", " ").lower(): f
        for f in feature_names
    }

    vector = np.zeros(len(feature_names))

    valid_skills = []

    for skill in user_skills:

        normalized_skill = skill.lower().strip()

        if normalized_skill in normalized_features:

            real_feature = normalized_features[normalized_skill]

            idx = feature_names.index(real_feature)

            vector[idx] = 1
            valid_skills.append(real_feature)

    return vector, valid_skills

def print_probabilities(class_probs):

    print("\nRole Probabilities:")
    print("-"*30)

    for role, prob in class_probs.items():
        print(f"{role}: {round(prob*100,2)}%")
