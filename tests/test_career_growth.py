import numpy as np
from sklearn.ensemble import RandomForestClassifier
from career_growth import simulate_career_growth


def test_simulate_career_growth():

    X = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    y = ["role1","role2","role3"]

    model = RandomForestClassifier()
    model.fit(X,y)

    user_vector = np.array([1,0,0])

    role_vectors = {
        "role1": np.array([1,0,0]),
        "role2": np.array([0,1,0]),
        "role3": np.array([0,0,1])
    }

    result = simulate_career_growth(
        model,
        user_vector,
        ["java"],
        ["python","java","aws"],
        role_vectors
    )

    assert isinstance(result, dict)
