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
    
    # Enable foreign keys
    c.execute('PRAGMA foreign_keys = ON')
    
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
                  FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)''')
    
    conn.commit()
    conn.close()

init_db()

def log_activity(user_id, activity_type, job_role=None, skills=None, score=None):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''INSERT INTO user_activities (user_id, activity_type, job_role, skills, score, created_at)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (user_id, activity_type, job_role, skills, score, datetime.now()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging activity: {e}")

def get_user_activities(user_id, limit=50):
    try:
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
    except Exception as e:
        print(f"Error fetching activities: {e}")
        return []

def parse_datetime_safe(dt_value):
    """Safely parse datetime from string or return as is"""
    if dt_value is None:
        return None
    if isinstance(dt_value, datetime):
        return dt_value
    if isinstance(dt_value, str):
        try:
            # Try different datetime formats
            return datetime.strptime(dt_value, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                return datetime.strptime(dt_value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    return datetime.fromisoformat(dt_value)
                except ValueError:
                    return datetime.now()
    return datetime.now()

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
            c.execute('''INSERT INTO user_activities (user_id, activity_type, created_at)
                         VALUES (?, ?, ?)''', (user[0], 'login', datetime.now()))
            
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
            
            c.execute('''INSERT INTO user_activities (user_id, activity_type, created_at)
                         VALUES (?, ?, ?)''', (user_id, 'register', datetime.now()))
            
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
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Get user info
        c.execute('SELECT name, email, created_at, last_login FROM users WHERE id = ?', 
                 (session['user_id'],))
        user = c.fetchone()
        
        if not user:
            conn.close()
            flash('User not found', 'error')
            return redirect(url_for('logout'))
        
        # Get activities
        activities_raw = get_user_activities(session['user_id'])
        
        activities = []
        for act in activities_raw:
            activity = {
                'activity_type': act[0],
                'job_role': act[1] if act[1] else 'N/A',
                'skills': act[2] if act[2] else 'N/A',
                'score': act[3] if act[3] else 'N/A',
                'created_at': parse_datetime_safe(act[4])
            }
            activities.append(activity)
        
        conn.close()
        
        # Parse user datetimes
        user_dict = {
            'name': user[0],
            'email': user[1],
            'created_at': parse_datetime_safe(user[2]),
            'last_login': parse_datetime_safe(user[3])
        }
        
        return render_template('profile.html', user=user_dict, activities=activities)
        
    except Exception as e:
        print(f"Error in profile route: {e}")
        flash('An error occurred while loading your profile', 'error')
        return redirect(url_for('skillsync_home'))

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

    default_important_skills = []    
    return render_template(
        "employability.html",
        score=0,
        status="Waiting for input",
        important_skills=default_important_skills,
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
    # Skill Gap Analysis
    # ===============================

    role_vector = role_vectors[preferred_role]

    missing_skills_full = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )
    
    missing_skills = missing_skills_full[:6]
    missing_skills_display = [skill.replace("_", " ").title() for skill, _ in missing_skills]
    
    # ===============================
    # Calculate Important Skills (Skill Importance)
    # ===============================
    
    # Get the top important skills for the preferred role
    # This identifies which skills are most critical for this role
    important_skills_data = []
    
    # Get the role vector and sort by importance (highest values first)
    role_vector_with_names = [(feature_names[i], role_vector[i]) for i in range(len(feature_names))]
    # Filter out skills with zero importance and sort by importance
    important_skills_sorted = sorted(
        [(skill, round(importance * 100, 1)) for skill, importance in role_vector_with_names if importance > 0.3],
        key=lambda x: x[1], 
        reverse=True
    )[:8]  # Get top 8 important skills
    
    important_skills_data = important_skills_sorted
    
    # Store in session
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
        important_skills=important_skills_data,  # Now passing actual data
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
    
    # Get parameters from URL
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
    
    # Get missing skills from session
    missing_skills = session.get('last_missing_skills', [])
    
    if not missing_skills:
        # Recalculate if needed
        user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
        user_vector = build_user_vector(user_skills, feature_names)
        role_vector = role_vectors[preferred_job]
        
        missing_skills_full = skill_gap_analysis(
            user_vector,
            role_vector,
            feature_names
        )
        missing_skills = [skill for skill, _ in missing_skills_full[:10]]
    
    # Calculate match percentage
    match_percent = 0
    if missing_skills:
        role_vector = role_vectors[preferred_job]
        total_relevant = sum(1 for v in role_vector if v > 0.3)
        match_percent = int((total_relevant - len(missing_skills)) / total_relevant * 100) if total_relevant > 0 else 0
    
    # Get course recommendations
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

@app.route('/predict_skills', methods=['POST'])
def predict_skills():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    data = request.get_json()
    preferred_job = data.get("job_role")
    user_skills_input = data.get("skills", "")
    user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
    
    # Use actual model to get skill gaps
    user_vector = build_user_vector(user_skills, feature_names)
    role_vector = role_vectors[preferred_job]
    
    missing_skills_full = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )
    
    missing_skills_display = [skill.replace('_', ' ').title() for skill, _ in missing_skills_full[:6]]
    
    # Store in session
    session['last_missing_skills'] = [skill for skill, _ in missing_skills_full[:10]]
    session['last_job_role'] = preferred_job
    session['last_user_skills'] = user_skills_input
    
    # Calculate match percentage
    total_relevant = sum(1 for v in role_vector if v > 0.3)
    match_percent = int((total_relevant - len(missing_skills_full)) / total_relevant * 100) if total_relevant > 0 else 0
    
    return jsonify({
        "mismatches": missing_skills_display,
        "match_percent": match_percent
    })

@app.route("/market_demand", methods=["POST"])
def market_demand():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    data = request.get_json()
    role = data.get("role", "").lower()
    
    # This is sample data - you should replace this with actual data from your database or API
    # You can fetch this data from a database, CSV file, or external API
    
    # Sample market demand data for different roles
    market_data = {
        "software engineer": [
            {"name": "Python", "demand": 85},
            {"name": "JavaScript", "demand": 82},
            {"name": "React", "demand": 78},
            {"name": "Java", "demand": 75},
            {"name": "SQL", "demand": 70},
            {"name": "AWS", "demand": 65},
            {"name": "Docker", "demand": 60},
            {"name": "Git", "demand": 55}
        ],
        "data scientist": [
            {"name": "Python", "demand": 90},
            {"name": "Machine Learning", "demand": 85},
            {"name": "SQL", "demand": 75},
            {"name": "Statistics", "demand": 70},
            {"name": "TensorFlow", "demand": 65},
            {"name": "R", "demand": 60},
            {"name": "Data Visualization", "demand": 55},
            {"name": "Deep Learning", "demand": 50}
        ],
        "data analyst": [
            {"name": "SQL", "demand": 88},
            {"name": "Excel", "demand": 85},
            {"name": "Python", "demand": 75},
            {"name": "Tableau", "demand": 70},
            {"name": "Power BI", "demand": 68},
            {"name": "Statistics", "demand": 65},
            {"name": "Data Visualization", "demand": 60},
            {"name": "R", "demand": 55}
        ],
        "devops engineer": [
            {"name": "Docker", "demand": 85},
            {"name": "Kubernetes", "demand": 82},
            {"name": "AWS", "demand": 80},
            {"name": "CI/CD", "demand": 78},
            {"name": "Jenkins", "demand": 75},
            {"name": "Linux", "demand": 70},
            {"name": "Terraform", "demand": 65},
            {"name": "Ansible", "demand": 60}
        ],
        "qa engineer": [
            {"name": "Selenium", "demand": 80},
            {"name": "Test Automation", "demand": 78},
            {"name": "JUnit", "demand": 70},
            {"name": "Python", "demand": 65},
            {"name": "Java", "demand": 60},
            {"name": "JMeter", "demand": 55},
            {"name": "Cucumber", "demand": 50},
            {"name": "TestNG", "demand": 45}
        ]
    }
    
    # Get data for the requested role, or provide default data
    skills = market_data.get(role, [
        {"name": "Python", "demand": 80},
        {"name": "SQL", "demand": 75},
        {"name": "JavaScript", "demand": 70},
        {"name": "Communication", "demand": 65},
        {"name": "Problem Solving", "demand": 60},
        {"name": "Teamwork", "demand": 55}
    ])
    
    # Log the activity
    log_activity(
        session['user_id'],
        'market_analysis',
        role,
        None,
        None
    )
    
    return jsonify({"skills": skills})

@app.route("/market-demand")
def market_demand_page():
    if 'user_id' not in session:
        flash('Please log in to access market demand analysis', 'error')
        return redirect(url_for('login'))
    
    return render_template("marketdemand.html", user={'name': session['user_name']})

@app.route("/education_alignment")
def education_alignment():
    if 'user_id' not in session:
        flash('Please log in to view education alignment', 'error')
        return redirect(url_for('login'))
    
    return render_template("index.html", user={'name': session['user_name']})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(debug=True)