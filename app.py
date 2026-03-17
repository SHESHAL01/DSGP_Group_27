from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from utils import build_user_vector
from employability import predict_employability
from skill_gap import compute_role_skill_vectors, skill_gap_analysis
from career_growth import simulate_career_growth
from explainability import explain_prediction
from recommend_courses import get_course_recommendations


app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'

# ===============================
# Load dataset and models
# ===============================

dataset = pd.read_csv("encoded_skills_dataset.csv")
feature_names = dataset.drop(columns=["title"]).columns.tolist()

# Load models
rf_model = joblib.load("saved_models/best_rf_model.pkl")
xgb_model = joblib.load("saved_models/best_xgb_model.pkl")
gb_model = joblib.load("saved_models/best_gb_model.pkl")

# ===============================
# Model accuracies
# ===============================

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
# Database functions
# ===============================

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_login TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_activities
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  activity_type TEXT NOT NULL,
                  job_role TEXT,
                  skills TEXT,
                  score FLOAT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

init_db()

def log_activity(user_id, activity_type, job_role=None, skills=None, score=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''INSERT INTO user_activities (user_id, activity_type, job_role, skills, score)
                 VALUES (?, ?, ?, ?, ?)''',
              (user_id, activity_type, job_role, skills, score))
    conn.commit()
    conn.close()

def get_user_activities(user_id, limit=50):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT activity_type, job_role, skills, score, created_at 
                 FROM user_activities 
                 WHERE user_id = ? 
                 ORDER BY created_at DESC 
                 LIMIT ?''', (user_id, limit))
    activities = c.fetchall()
    conn.close()
    return activities

# ===============================
# AUTHENTICATION ROUTES
# ===============================

@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('skillsync_home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, name, email, password FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            session['user_email'] = user[2]
            
            c.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                     (datetime.now(), user[0]))
            c.execute('''INSERT INTO user_activities (user_id, activity_type)
                         VALUES (?, ?)''', (user[0], 'login'))
            
            conn.commit()
            conn.close()
            
            flash('Logged in successfully!', 'success')
            return redirect(url_for('skillsync_home'))
        else:
            conn.close()
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        try:
            c.execute('''INSERT INTO users (name, email, password, created_at)
                         VALUES (?, ?, ?, ?)''',
                     (name, email, hashed_password, datetime.now()))
            user_id = c.lastrowid
            
            c.execute('''INSERT INTO user_activities (user_id, activity_type)
                         VALUES (?, ?)''', (user_id, 'register'))
            
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.IntegrityError:
            conn.close()
            flash('Email already exists. Please use a different email.', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        log_activity(session['user_id'], 'logout')
    
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please log in to view your profile', 'error')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('SELECT name, email, created_at, last_login FROM users WHERE id = ?', 
             (session['user_id'],))
    user = c.fetchone()
    
    activities_raw = get_user_activities(session['user_id'])
    
    activities = []
    for act in activities_raw:
        activity = {
            'activity_type': act[0],
            'job_role': act[1] if act[1] else 'N/A',
            'skills': act[2] if act[2] else 'N/A',
            'score': act[3] if act[3] else 'N/A',
            'created_at': datetime.fromisoformat(act[4]) if isinstance(act[4], str) else act[4]
        }
        activities.append(activity)
    
    conn.close()
    
    user_dict = {
        'name': user[0],
        'email': user[1],
        'created_at': datetime.fromisoformat(user[2]) if isinstance(user[2], str) else user[2],
        'last_login': datetime.fromisoformat(user[3]) if isinstance(user[3], str) else user[3]
    }
    
    return render_template('profile.html', user=user_dict, activities=activities)

# ===============================
# MAIN ROUTES
# ===============================

@app.route("/home")
def skillsync_home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template(
        "skillsync.html",
        user={'name': session['user_name']}
    )

@app.route("/employability")
def employability_page():
    if 'user_id' not in session:
        flash('Please log in to access the employability predictor', 'error')
        return redirect(url_for('login'))
    
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
        user={'name': session['user_name']}
    )

@app.route("/predict", methods=["POST"])
def predict():
    if 'user_id' not in session:
        return {'error': 'Please log in'}, 401
    
    preferred_role = request.form["preferred_role"]
    skills_text = request.form["skills"]
    user_skills = [s.strip() for s in skills_text.split(",")]

    # Build vector
    user_vector = build_user_vector(user_skills, feature_names)

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

    alternative_roles = [
        (role, round(prob, 2)) for role, prob in alternative_roles
    ]

    score = round(score, 2)

    # Status
    if score > 75:
        status = "Highly Employable"
    elif score > 50:
        status = "Moderately Employable"
    else:
        status = "Needs Skill Improvement"

    # ===============================
    # Skill Gap Analysis (USING YOUR ACTUAL MODEL)
    # ===============================

    role_vector = role_vectors[preferred_role]

    missing_skills_full = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )
    
    missing_skills = missing_skills_full[:6]
    missing_skills_display = [skill.replace("_", " ").title() for skill, _ in missing_skills]
    
    # Store the raw missing skills (with underscores) in session for recommendation page
    session['last_missing_skills'] = [skill for skill, _ in missing_skills_full[:10]]
    session['last_job_role'] = preferred_role
    session['last_user_skills'] = skills_text

    # Log the prediction activity
    log_activity(
        session['user_id'],
        'prediction',
        preferred_role,
        skills_text,
        score
    )

    return render_template(
        "employability.html",
        score=score,
        status=status,
        important_skills=[],
        alternative_roles=alternative_roles,
        missing_skills=missing_skills_display,
        lower1=score,
        lower2=score,
        top_models=top_models,
        precision=rf_precision,
        recall=rf_recall,
        f1=rf_f1,
        user={'name': session['user_name']}
    )

@app.route("/recommend", methods=['GET'])
def recommendation_page():
    if 'user_id' not in session:
        flash('Please log in to access recommendations', 'error')
        return redirect(url_for('login'))
    
    # Get parameters from URL (coming from employability page)
    preferred_job = request.args.get("job_role")
    user_skills_input = request.args.get("skills", "")
    
    # If no URL parameters, try to use session data
    if not preferred_job or not user_skills_input:
        preferred_job = session.get('last_job_role', '')
        user_skills_input = session.get('last_user_skills', '')
    
    if not preferred_job or not user_skills_input:
        return render_template(
            "recommendation.html",
            job="",
            user_skills="",
            mismatches=[],
            recommendations={},
            score=0,
            metrics={"Precision": rf_precision, "Recall": rf_recall, "F1": rf_f1},
            user={'name': session['user_name']}
        )
    
    # Get missing skills from session (these are the REAL skill gaps from your model)
    missing_skills = session.get('last_missing_skills', [])
    
    if not missing_skills:
        # If session doesn't have missing skills, recalculate them
        user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
        user_vector = build_user_vector(user_skills, feature_names)
        role_vector = role_vectors[preferred_job]
        
        missing_skills_full = skill_gap_analysis(
            user_vector,
            role_vector,
            feature_names
        )
        missing_skills = [skill for skill, _ in missing_skills_full[:10]]
    
    # Calculate match percentage (just for display)
    match_percent = 0
    if missing_skills:
        # This is a rough estimate - you could calculate this differently
        role_vector = role_vectors[preferred_job]
        total_relevant = sum(1 for v in role_vector if v > 0.3)
        match_percent = int((total_relevant - len(missing_skills)) / total_relevant * 100) if total_relevant > 0 else 0
    
    # Get course recommendations for missing skills (using your final_DS.csv)
    from recommend_courses import get_course_recommendations
    recommendations = get_course_recommendations(missing_skills, preferred_job)
    
    # Clean skill names for display
    mismatches_display = [skill.replace('_', ' ').title() for skill in missing_skills]
    
    # Log recommendation activity
    log_activity(
        session['user_id'],
        'recommendation',
        preferred_job,
        user_skills_input
    )
    
    return render_template(
        "recommendation.html",
        job=preferred_job,
        user_skills=user_skills_input,
        mismatches=mismatches_display,
        recommendations=recommendations,
        score=match_percent,
        metrics={"Precision": rf_precision, "Recall": rf_recall, "F1": rf_f1},
        user={'name': session['user_name']}
    )

# AJAX endpoint for "Get Prediction" (used by employability.html)
@app.route('/predict_skills', methods=['POST'])
def predict_skills():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    data = request.get_json()
    preferred_job = data.get("job_role")
    user_skills_input = data.get("skills", "")
    user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
    
    # Use your actual model to get skill gaps
    user_vector = build_user_vector(user_skills, feature_names)
    role_vector = role_vectors[preferred_job]
    
    missing_skills_full = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )
    
    missing_skills_display = [skill.replace('_', ' ').title() for skill, _ in missing_skills_full[:6]]
    
    # Store in session for recommendation page
    session['last_missing_skills'] = [skill for skill, _ in missing_skills_full[:10]]
    session['last_job_role'] = preferred_job
    session['last_user_skills'] = user_skills_input
    
    # Calculate a simple match percentage (optional)
    total_relevant = sum(1 for v in role_vector if v > 0.3)
    match_percent = int((total_relevant - len(missing_skills_full)) / total_relevant * 100) if total_relevant > 0 else 0
    
    return jsonify({
        "mismatches": missing_skills_display,
        "match_percent": match_percent
    })
@app.route("/education_alignment")
def education_alignment():
    if 'user_id' not in session:
        flash('Please log in to view education alignment', 'error')
        return redirect(url_for('login'))
    
    return render_template("index.html", user={'name': session['user_name']})

if __name__ == "__main__":
    app.run(debug=True)