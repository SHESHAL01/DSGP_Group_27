import numpy as np

def build_user_vector(user_skills, feature_names):

    vector = np.zeros(len(feature_names))

    for skill in user_skills:
        if skill in feature_names:
            idx = feature_names.index(skill)
            vector[idx] = 1

    return vector


def print_probabilities(class_probs):

    print("\nRole Probabilities:")
    print("-"*30)

    for role, prob in class_probs.items():
        print(f"{role}: {round(prob*100,2)}%")
