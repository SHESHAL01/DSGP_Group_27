from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

MODEL_PATH = "saved_course_model"
EMBEDDINGS_PATH = "course_embeddings.npy"
DATA_PATH = "final_DS.csv"

model = SentenceTransformer(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
course_embeddings = np.load(EMBEDDINGS_PATH)

JOB_SKILLS = {
    "Data Scientist": ["python", "machine learning", "statistics", "sql", "pandas"],
    "Software Engineer": ["java", "data structures", "algorithms", "docker", "git"],
    "Machine Learning Engineer": ["python", "tensorflow", "pytorch", "docker", "aws"],
    "Data Analyst": ["excel", "sql", "power bi", "statistics"],
    "Cloud Engineer": ["aws", "docker", "kubernetes", "linux"]
}

MODEL_METRICS = {"Precision": 0.82, "Recall": 0.79, "F1": 0.80}

# Home page
@app.route('/')
def home():
    return render_template("employability.html", mismatches=[])

# Recommendation page
@app.route('/recommend', methods=['GET'])
def recommendation_page():
    preferred_job = request.args.get("job_role")
    user_skills_input = request.args.get("skills")
    if not preferred_job or not user_skills_input:
        return "Missing job role or skills!", 400

    user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
    required_skills = JOB_SKILLS.get(preferred_job, [])
    mismatches = list(set(required_skills) - set(user_skills))

    recommendations = {}
    for skill in mismatches:
        recs = retrieve_top_k_courses(skill, k=5)
        recommendations[skill] = recs

    match_percent = int((len(required_skills) - len(mismatches)) / len(required_skills) * 100) if required_skills else 0

    return render_template(
        "recommendation.html",
        job=preferred_job,
        user_skills=user_skills_input,
        mismatches=mismatches,
        recommendations=recommendations,
        score=match_percent,
        metrics=MODEL_METRICS
    )

# AJAX endpoint for "Get Prediction"
@app.route('/predict_skills', methods=['POST'])
def predict_skills():
    data = request.get_json()
    preferred_job = data.get("job_role")
    user_skills_input = data.get("skills", "")
    user_skills = [s.strip().lower() for s in user_skills_input.split(",")]

    required_skills = JOB_SKILLS.get(preferred_job, [])
    mismatches = list(set(required_skills) - set(user_skills))

    match_percent = int((len(required_skills) - len(mismatches)) / len(required_skills) * 100) if required_skills else 0

    return jsonify({
        "mismatches": mismatches,
        "match_percent": match_percent
    })

def retrieve_top_k_courses(query_skill, k=5):
    query_embedding = model.encode(
        [query_skill],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    sims = np.dot(course_embeddings, query_embedding.T).flatten()
    top_k_idx = np.argsort(-sims)[:k]
    results = df.iloc[top_k_idx][["Title", "Url"]]
    return results.to_dict(orient="records")

if __name__ == '__main__':
    app.run(debug=True)