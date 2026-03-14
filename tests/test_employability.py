import numpy as np
from sklearn.ensemble import RandomForestClassifier
from employability import predict_employability


def test_predict_employability():

    # dummy model
    X = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    y = ["data_scientist","software_engineer","cloud_engineer"]

    model = RandomForestClassifier()
    model.fit(X,y)

    user_vector = np.array([1,0,0])

    role_vectors = {
        "data_scientist": np.array([1,0.5,0]),
        "software_engineer": np.array([0,1,0.5]),
        "cloud_engineer": np.array([0,0,1])
    }

    score, probs, alternatives = predict_employability(
        model,
        user_vector,
        "data_scientist",
        role_vectors,
        ["python","java","aws"]
    )

    assert score >= 0
    assert isinstance(probs, dict)
