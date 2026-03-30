from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

from utils import build_user_vector
from employability import predict_employability
from skill_gap import compute_role_skill_vectors, skill_gap_analysis
from career_growth import simulate_career_growth
from explainability import explain_prediction


app = Flask(__name__)

# ===============================
# Load dataset
# ===============================

dataset = pd.read_csv("encoded_skills_dataset.csv")

feature_names = dataset.drop(columns=["title"]).columns.tolist()

# ===============================
# Load models
# ===============================

best_rf_model = joblib.load("saved_models/best_rf_model.pkl")

xgb_model = joblib.load("saved_models/best_xgb_model.pkl")
gb_model = joblib.load("saved_models/best_gb_model.pkl")

# ===============================
# Model accuracies
# ===============================

import joblib

rf_model = joblib.load("saved_models/best_rf_model.pkl")
xgb_model = joblib.load("saved_models/best_xgb_model.pkl")
gb_model = joblib.load("saved_models/best_gb_model.pkl")

metrics_other = joblib.load("saved_models/model_metrics.pkl")
metrics_rf = joblib.load("saved_models/model_metrics_rf.pkl")

model_metrics = {}
model_metrics.update(metrics_other)
model_metrics.update(metrics_rf)

model_scores = []

for model_name, values in model_metrics.items():
    model_scores.append(
        (model_name, round(values["accuracy"], 3))
    )

top_models = sorted(
    model_scores,
    key=lambda x: x[1],
    reverse=True
)[:3]


rf_precision = round(model_metrics["Random Forest"]["precision"],3)
rf_recall = round(model_metrics["Random Forest"]["recall"],3)
rf_f1 = round(model_metrics["Random Forest"]["f1"],3)

# ===============================
# Precompute role skill vectors
# ===============================

role_vectors = compute_role_skill_vectors(
    dataset,
    feature_names,
    "title"
)

# ===============================
# HOME PAGE
# ===============================

@app.route("/")
def home():

    return render_template(
        "employability.html",
        score=0,
        status="Waiting for input",
        important_skills=[],
        alternative_roles=[],
        missing_skills=[],
        lower1=0,
        full1=0,
        lower2=0,
        full2=0,
        top_models=top_models,
        precision=rf_precision,
        recall=rf_recall,
        f1=rf_f1,


    )


# ===============================
# PREDICTION
# ===============================

@app.route("/predict", methods=["POST"])
def predict():

    preferred_role = request.form["preferred_role"]

    skills_text = request.form["skills"]

    user_skills = [s.strip() for s in skills_text.split(",")]

    # Build vector
    user_vector, valid_skills = build_user_vector(user_skills, feature_names)
    if len(valid_skills) == 0:
        return render_template(
            "employability.html",
            score=0,
            status="Invalid Skills Entered",
            important_skills=[],
            alternative_roles=[],
            missing_skills=[],
            simulations=[],
            top_models=top_models,
            precision=rf_precision,
            recall=rf_recall,
            f1=rf_f1
        )

    # ===============================
    # Employability prediction
    # ===============================

    score, role_probs, alternative_roles = predict_employability(
        rf_model,
        user_vector,
        preferred_role,
        role_vectors,
        feature_names
    )

    # Round alternative role scores
    alternative_roles = [
        (role, round(prob, 2)) for role, prob in alternative_roles
    ]

    score = round(score, 2)

    # ===============================
    # Status
    # ===============================

    if score > 75:
        status = "Highly Employable"

    elif score > 50:
        status = "Moderately Employable"

    else:
        status = "Needs Skill Improvement"

    # ===============================
    # Skill Gap Analysis
    # ===============================

    role_vector = role_vectors[preferred_role]

    missing_skills = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )[:6]

    # keep only skill names
    missing_skills = [
        skill.replace("_", " ").title()
        for skill, _ in missing_skills
    ]
    # ===============================
    # Explainable AI
    # ===============================

    import math

    important_skills = []

    role_vector = role_vectors[preferred_role]

    for skill in valid_skills:

        if skill in feature_names:
            idx = feature_names.index(skill)

            role_value = role_vector[idx]

            # importance = how important this skill is for the role
            importance = role_value

            important_skills.append((skill, importance))

    # Sort by importance
    important_skills = sorted(
        important_skills,
        key=lambda x: x[1],
        reverse=True
    )[:4]
    # Log scaling for visualization
    scaled_skills = []

    if important_skills:

        max_importance = max([imp for _, imp in important_skills])

        for skill, imp in important_skills:
            scaled = (imp / max_importance) * 85

            scaled_skills.append(
                (skill.replace("_", " ").title(), round(scaled, 2))
            )

    important_skills = scaled_skills

    # ===============================
    # Career growth simulation
    # ===============================

    from itertools import combinations

    missing_skills_full = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )

    missing_skills_only = [skill for skill, _ in missing_skills_full]

    simulations = []

    skill_sets = []

    skill_sets += list(combinations(missing_skills_only, 1))
    skill_sets += list(combinations(missing_skills_only, 2))

    skill_sets = skill_sets[:4]

    for skill_set in skill_sets:
        sim_result = simulate_career_growth(
            rf_model,
            user_vector,
            list(skill_set),
            feature_names,
            role_vectors
        )

        new_score = round(sim_result.get(preferred_role, 0) * 100, 2)

        simulations.append({
            "skills": " + ".join(skill.replace("_", " ").title()for skill in skill_set),
            "old_score": score,
            "new_score": new_score
        })

    return render_template(
        "employability.html",
        score=score,
        status=status,
        important_skills=important_skills,
        alternative_roles=alternative_roles,
        missing_skills=missing_skills,
        simulations=simulations,
        lower1=score,
        lower2=score,
        top_models=top_models,
        precision=rf_precision,
        recall=rf_recall,
        f1=rf_f1
    )


if __name__ == "__main__":
    app.run(debug=True)

