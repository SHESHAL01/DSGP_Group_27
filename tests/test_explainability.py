import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainability import explain_prediction


def test_explain_prediction():

    X = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    y = ["a","b","c"]

    model = RandomForestClassifier()
    model.fit(X,y)

    features = ["python","java","aws"]

    result = explain_prediction(model, features, top_n=2)

    assert len(result) == 2
