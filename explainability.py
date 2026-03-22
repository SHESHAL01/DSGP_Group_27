def explain_prediction(model, feature_names, top_n=10):

    importances = model.feature_importances_

    features = list(zip(feature_names, importances))

    ranked = sorted(
        features,
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]
