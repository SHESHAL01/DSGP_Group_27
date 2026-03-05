import joblib

MODEL_PATH = "saved_models/best_rf_model.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model
