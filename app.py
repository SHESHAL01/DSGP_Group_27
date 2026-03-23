from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import ast
from collections import Counter
from utils import build_user_vector
from employability import predict_employability
from skill_gap import compute_role_skill_vectors, skill_gap_analysis
from career_growth import simulate_career_growth
from explainability import explain_prediction
from recommend_courses import get_course_recommendations
import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from EduIndPipeline with error handling
try:
    from EduIndPipeline import (
        clean_skills_cell,
        market_demand_skills,
        jaccard_relavance,
        cosine_relavance,
        DataAnalysis,
        test_Sbert
    )
    logger.info("Successfully imported EduIndPipeline modules")
except ImportError as e:
    logger.error(f"Error importing from EduIndPipeline: {e}")
    logger.error(traceback.format_exc())
    # Define fallback functions
    def clean_skills_cell(cell): 
        return cell if pd.notna(cell) else ""
    
    def market_demand_skills(): 
        try:
            market_df = pd.read_csv("extracted_skills.csv")
            market_skills = set()
            for raw in market_df["extracted_skills"].dropna():
                try:
                    items = ast.literal_eval(raw)
                    market_skills.update({str(t).strip().lower() for t in items if str(t).strip()})
                except:
                    pass
            return market_skills
        except:
            return set()
    
    def jaccard_relavance(df, market_skills):
        return pd.DataFrame({"University": ["No Data"], "Jaccard_Score": [0.0], "Relevance_%": ["0.00%"]})
    
    def cosine_relavance(df, market_df):
        return pd.DataFrame({"University": ["No Data"], "Relevance_Score_%": [0.0]})
    
    def DataAnalysis(df):
        print("Data analysis function not available")
    
    def test_Sbert(df):
        print("SBERT training function not available")

# ======================================
# Load Skills From CSV For Market Demand
# ======================================
try:
    skills_df = pd.read_csv("extracted_skills.csv")
    logger.info("Successfully loaded extracted_skills.csv")
except Exception as e:
    logger.error(f"Error loading extracted_skills.csv: {e}")
    skills_df = pd.DataFrame()

SKILL_DICTIONARY = [
    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "go", "rust",
    "php", "ruby", "swift", "kotlin", "r", "matlab", "scala", "perl", "bash",
    "powershell", "objective-c", "groovy", "dart", "lua", "haskell",

    # Web Development
    "html", "css", "sass", "less", "bootstrap", "tailwind css",
    "react", "angular", "vue", "next.js", "nuxt.js", "svelte",
    "node.js", "express.js", "nestjs",
    "django", "flask", "fastapi",
    "spring", "spring boot",
    "laravel", "codeigniter",
    "asp.net", "asp.net core",
    "graphql", "rest api", "soap",

    # Databases
    "mysql", "postgresql", "oracle", "sql server", "sqlite",
    "mongodb", "cassandra", "couchdb", "redis", "dynamodb",
    "firebase", "neo4j", "elasticsearch",
    "nosql", "sql", "pl/sql",

    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp",
    "ec2", "s3", "lambda", "cloudformation",
    "docker", "kubernetes", "helm",
    "terraform", "ansible", "chef", "puppet",
    "jenkins", "gitlab ci", "github actions", "circleci",
    "linux", "unix",
    "nginx", "apache",
    "devops", "site reliability engineering", "sre",

    # Data Science & Machine Learning
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "computer vision",
    "data science", "data analysis", "data engineering",
    "pandas", "numpy", "scipy", "scikit-learn",
    "tensorflow", "keras", "pytorch",
    "xgboost", "lightgbm",
    "opencv",
    "statistics", "linear regression", "logistic regression",
    "clustering", "classification", "time series",

    # Big Data
    "hadoop", "spark", "pyspark", "kafka", "flink",
    "hive", "pig", "hbase", "airflow",
    "data warehousing", "etl",

    # Mobile Development
    "android", "ios",
    "react native", "flutter", "xamarin",
    "android studio", "xcode",

    # Cybersecurity
    "cybersecurity", "information security",
    "penetration testing", "ethical hacking",
    "network security", "application security",
    "cryptography", "siem", "soc",
    "firewalls", "ids", "ips",
    "owasp", "iam",

    # Networking
    "tcp/ip", "udp", "dns", "dhcp",
    "http", "https",
    "routing", "switching",
    "vpn", "lan", "wan",
    "ccna", "ccnp",

    # Operating Systems
    "windows", "linux", "macos",
    "red hat", "ubuntu", "debian", "centos",

    # Software Engineering
    "object oriented programming", "oop",
    "design patterns",
    "clean code", "solid principles",
    "data structures", "algorithms",
    "microservices", "monolithic architecture",
    "event driven architecture",

    # Testing & QA
    "unit testing", "integration testing", "system testing",
    "selenium", "cypress", "playwright",
    "junit", "pytest", "testng",
    "automation testing", "manual testing",

    # Version Control & Tools
    "git", "github", "gitlab", "bitbucket",
    "jira", "confluence",
    "postman", "swagger",

    # UI / UX
    "figma", "adobe xd", "sketch",
    "ui design", "ux design",
    "wireframing", "prototyping",

    # ERP / CRM / Enterprise
    "sap", "oracle erp", "salesforce",
    "workday", "servicenow",

    # Methodologies
    "agile", "scrum", "kanban",
    "waterfall", "devsecops",

    # Misc / Emerging
    "blockchain", "web3", "smart contracts",
    "solidity",
    "internet of things", "iot",
    "robotic process automation", "rpa",
    "computer graphics", "game development",
    "unity", "unreal engine"
]
SKILL_SET = set([s.lower() for s in SKILL_DICTIONARY])

# Replace the process_skills function (around line 250-275) with this:

import re
from difflib import get_close_matches
from collections import Counter

def process_skills():
    if skills_df.empty:
        return {}
    
    # Create a set of normalized skill names for matching
    skill_set_lower = set([s.lower() for s in SKILL_SET])
    
    # Create a mapping for common abbreviations and variations
    skill_mappings = {
        'aw': 'aws',
        'ku': 'kubernetes',
        'dock': 'docker',
        'gi': 'git',
        'gith': 'git',
        'reactjs': 'react',
        'graphq': 'graphql',
        'postgresq': 'postgresql',
        'mysq': 'mysql',
        'nosq': 'nosql',
        'mongodb': 'mongodb',
        'kubernet': 'kubernetes',
        'devops': 'devops',
        'python': 'python',
        'java': 'java',
        'javascript': 'javascript',
        'typescript': 'typescript',
        'html': 'html',
        'css': 'css',
        'sql': 'sql',
        'agile': 'agile',
        'scrum': 'scrum',
        'unit': 'unit testing',
        'integration': 'integration testing',
        'testing': 'testing',
        'micro': 'microservices',
        'aw': 'aws',
        'gcp': 'google cloud',
        'azure': 'azure',
        'spring': 'spring',
        'boot': 'spring boot',
        'django': 'django',
        'flask': 'flask',
        'react': 'react',
        'angular': 'angular',
        'vue': 'vue.js',
        'vu': 'vue.js',
        'node': 'node.js',
        'docker': 'docker',
        'kubernetes': 'kubernetes',
        'jenkins': 'jenkins',
        'git': 'git',
        'github': 'github',
        'gitlab': 'gitlab',
        'tensorflow': 'tensorflow',
        'pytorch': 'pytorch',
        'scikit': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'machine': 'machine learning',
        'deep': 'deep learning',
        'ai': 'artificial intelligence',
        'data': 'data analysis',
        'science': 'data science',
        'cyber': 'cybersecurity',
        'security': 'cybersecurity',
        'cloud': 'cloud computing',
        'block': 'blockchain',
        'kotl': 'kotlin',
        'scala': 'scala',
        'rust': 'rust',
        'go': 'golang',
        'php': 'php',
        'ruby': 'ruby',
        'swift': 'swift',
        'flutter': 'flutter',
        'android': 'android development',
        'ios': 'ios development',
        'fig': 'figma',
        'ui': 'ui design',
        'ux': 'ux design'
    }
    
    def normalize_skill(skill):
        skill = skill.strip().lower()
        
        # Remove common suffixes and clean up
        skill = re.sub(r'[^a-z0-9\s]', '', skill)
        
        # Check direct mapping
        if skill in skill_mappings:
            return skill_mappings[skill]
        
        # Check for partial matches in SKILL_SET
        for key, value in skill_mappings.items():
            if skill in key or key in skill:
                return value
        
        # Try fuzzy matching for close matches
        close_matches = get_close_matches(skill, skill_set_lower, n=1, cutoff=0.6)
        if close_matches:
            return close_matches[0]
        
        # Check if skill is in SKILL_SET
        if skill in skill_set_lower:
            return skill
        
        # Check for skill being part of a longer skill name
        for std_skill in skill_set_lower:
            if skill in std_skill or std_skill in skill:
                return std_skill
        
        return skill  # Return original if no match found
    
    role_skills = {}
    for _, row in skills_df.iterrows():
        role = str(row.get("role", "")).strip().lower()
        
        # Clean role name - extract main role
        if '(' in role:
            role = role.split('(')[0].strip()
        if ',' in role:
            role = role.split(',')[0].strip()
        if '-' in role:
            role = role.split('-')[0].strip()
        
        # Map common role variations
        role_mappings = {
            'software engineer': 'software engineer',
            'senior software engineer': 'software engineer',
            'staff software engineer': 'software engineer',
            'lead software engineer': 'software engineer',
            'software developer': 'software engineer',
            'backend engineer': 'software engineer',
            'frontend engineer': 'software engineer',
            'full stack engineer': 'software engineer',
            'qa engineer': 'qa engineer',
            'quality assurance': 'qa engineer',
            'software quality assurance': 'qa engineer',
            'test engineer': 'qa engineer',
            'cybersecurity engineer': 'cybersecurity engineer',
            'security engineer': 'cybersecurity engineer',
            'data engineer': 'data engineer',
            'data scientist': 'data scientist',
            'data analyst': 'data analyst',
            'devops engineer': 'devops engineer',
            'site reliability engineer': 'devops engineer',
            'embedded engineer': 'embedded engineer',
            'database administrator': 'database administrator',
            'dba': 'database administrator'
        }
        
        role = role_mappings.get(role, role)
        
        try:
            skills_list = ast.literal_eval(row.get("extracted_skills", "[]"))
        except:
            continue

        cleaned_skills = []
        for s in skills_list:
            if isinstance(s, str) and s.strip():
                normalized = normalize_skill(s.strip())
                # Only include if it's a meaningful skill (at least 2 chars or in mappings)
                if len(normalized) > 2 or normalized in skill_mappings.values():
                    cleaned_skills.append(normalized)

        if role not in role_skills:
            role_skills[role] = []
        
        role_skills[role].extend(cleaned_skills)

    # Aggregate and get top skills for each role
    top_skills = {}
    for role, skills in role_skills.items():
        if skills:  # Only include roles that have skills
            counter = Counter(skills)
            # Remove duplicates and get top 10
            top_skills[role] = [skill for skill, _ in counter.most_common(10)]
    
    return top_skills

TOP_SKILLS = process_skills()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'

# ===============================
# Load dataset and models
# ===============================
try:
    dataset = pd.read_csv("encoded_skills_dataset.csv")
    feature_names = dataset.drop(columns=["title"]).columns.tolist()
    logger.info("Successfully loaded encoded_skills_dataset.csv")
except Exception as e:
    logger.error(f"Error loading encoded_skills_dataset.csv: {e}")
    dataset = pd.DataFrame()
    feature_names = []

# Load models
try:
    rf_model = joblib.load("saved_models/best_rf_model.pkl")
    xgb_model = joblib.load("saved_models/best_xgb_model.pkl")
    gb_model = joblib.load("saved_models/best_gb_model.pkl")
    logger.info("Successfully loaded models")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    rf_model = xgb_model = gb_model = None

# ===============================
# Model accuracies
# ===============================
model_metrics = {}
try:
    metrics_other = joblib.load("saved_models/model_metrics.pkl")
    metrics_rf = joblib.load("saved_models/model_metrics_rf.pkl")
    model_metrics.update(metrics_other)
    model_metrics.update(metrics_rf)
except Exception as e:
    logger.error(f"Error loading model metrics: {e}")

model_scores = []
for model_name, values in model_metrics.items():
    model_scores.append(
        (model_name, round(values.get("accuracy", 0), 3))
    )

top_models = sorted(
    model_scores,
    key=lambda x: x[1],
    reverse=True
)[:3]

rf_precision = round(model_metrics.get("Random Forest", {}).get("precision", 0), 3)
rf_recall = round(model_metrics.get("Random Forest", {}).get("recall", 0), 3)
rf_f1 = round(model_metrics.get("Random Forest", {}).get("f1", 0), 3)

# ===============================
# Precompute role skill vectors
# ===============================
role_vectors = {}
if not dataset.empty:
    try:
        role_vectors = compute_role_skill_vectors(
            dataset,
            feature_names,
            "title"
        )
    except Exception as e:
        logger.error(f"Error computing role vectors: {e}")

# ===============================
# Database functions
# ===============================

def init_db():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

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
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

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
        logger.error(f"Error logging activity: {e}")

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
        logger.error(f"Error fetching activities: {e}")
        return []

def parse_datetime_safe(dt_value):
    """Safely parse datetime from string or return as is"""
    if dt_value is None:
        return None
    if isinstance(dt_value, datetime):
        return dt_value
    if isinstance(dt_value, str):
        try:
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

        c.execute('SELECT name, email, created_at, last_login FROM users WHERE id = ?',
                 (session['user_id'],))
        user = c.fetchone()

        if not user:
            conn.close()
            flash('User not found', 'error')
            return redirect(url_for('logout'))

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

        user_dict = {
            'name': user[0],
            'email': user[1],
            'created_at': parse_datetime_safe(user[2]),
            'last_login': parse_datetime_safe(user[3])
        }

        return render_template('profile.html', user=user_dict, activities=activities)

    except Exception as e:
        logger.error(f"Error in profile route: {e}")
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

    if not dataset.empty and feature_names and rf_model:
        user_vector = build_user_vector(user_skills, feature_names)
        score, role_probs, alternative_roles = predict_employability(
            rf_model,
            user_vector,
            preferred_role,
            role_vectors,
            feature_names
        )
        alternative_roles = [(role, round(prob, 2)) for role, prob in alternative_roles]
        score = round(score, 2)

        if score > 75:
            status = "Highly Employable"
        elif score > 50:
            status = "Moderately Employable"
        else:
            status = "Needs Skill Improvement"

        role_vector = role_vectors.get(preferred_role, [0] * len(feature_names))
        missing_skills_full = skill_gap_analysis(
            user_vector,
            role_vector,
            feature_names
        )
        missing_skills = missing_skills_full[:6]
        missing_skills_display = [skill.replace("_", " ").title() for skill, _ in missing_skills]

        important_skills_data = []
        role_vector_with_names = [(feature_names[i], role_vector[i]) for i in range(len(feature_names))]
        important_skills_sorted = sorted(
            [(skill, round(importance * 100, 1)) for skill, importance in role_vector_with_names if importance > 0.3],
            key=lambda x: x[1],
            reverse=True
        )[:8]
        important_skills_data = important_skills_sorted

        session['last_missing_skills'] = [skill for skill, _ in missing_skills_full[:10]]
        session['last_job_role'] = preferred_role
        session['last_user_skills'] = skills_text

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
            important_skills=important_skills_data,
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
    else:
        flash('Models not loaded properly. Please check system configuration.', 'error')
        return redirect(url_for('employability_page'))

@app.route("/recommend", methods=['GET'])
def recommendation_page():
    if 'user_id' not in session:
        flash('Please log in to access recommendations', 'error')
        return redirect(url_for('login'))

    preferred_job = request.args.get("job_role")
    user_skills_input = request.args.get("skills", "")

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

    missing_skills = session.get('last_missing_skills', [])

    if not missing_skills and not dataset.empty and feature_names:
        user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
        user_vector = build_user_vector(user_skills, feature_names)
        role_vector = role_vectors.get(preferred_job, [0] * len(feature_names))

        missing_skills_full = skill_gap_analysis(
            user_vector,
            role_vector,
            feature_names
        )
        missing_skills = [skill for skill, _ in missing_skills_full[:10]]

    match_percent = 0
    if missing_skills and not dataset.empty and feature_names:
        role_vector = role_vectors.get(preferred_job, [0] * len(feature_names))
        total_relevant = sum(1 for v in role_vector if v > 0.3)
        match_percent = int((total_relevant - len(missing_skills)) / total_relevant * 100) if total_relevant > 0 else 0

    recommendations = get_course_recommendations(missing_skills, preferred_job)
    mismatches_display = [skill.replace('_', ' ').title() for skill in missing_skills]

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

    if dataset.empty or not feature_names:
        return jsonify({"error": "System not properly configured"}), 500

    user_vector = build_user_vector(user_skills, feature_names)
    role_vector = role_vectors.get(preferred_job, [0] * len(feature_names))

    missing_skills_full = skill_gap_analysis(
        user_vector,
        role_vector,
        feature_names
    )

    missing_skills_display = [skill.replace('_', ' ').title() for skill, _ in missing_skills_full[:6]]

    session['last_missing_skills'] = [skill for skill, _ in missing_skills_full[:10]]
    session['last_job_role'] = preferred_job
    session['last_user_skills'] = user_skills_input

    total_relevant = sum(1 for v in role_vector if v > 0.3)
    match_percent = int((total_relevant - len(missing_skills_full)) / total_relevant * 100) if total_relevant > 0 else 0

    return jsonify({
        "mismatches": missing_skills_display,
        "match_percent": match_percent
    })

# Replace the market_demand route (around line 830-850) with this:

@app.route("/market_demand", methods=["POST"])
def market_demand():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in"}), 401

    data = request.get_json()
    role = str(data.get("role", "")).strip().lower()
    
    # Try exact match first
    skills = TOP_SKILLS.get(role, [])
    
    # If no exact match, try fuzzy matching on role names
    if not skills:
        # Look for roles that contain this role
        matching_roles = [r for r in TOP_SKILLS.keys() if role in r or r in role]
        if matching_roles:
            # Aggregate skills from all matching roles
            all_skills = []
            for matching_role in matching_roles:
                all_skills.extend(TOP_SKILLS[matching_role])
            
            # Count frequencies and get top 10
            if all_skills:
                counter = Counter(all_skills)
                skills = [skill for skill, _ in counter.most_common(10)]
    
    # If still no skills, provide some default based on role
    if not skills:
        default_skills_map = {
            'software': ['python', 'java', 'javascript', 'react', 'sql', 'git', 'aws', 'docker', 'spring boot', 'agile'],
            'qa': ['testing', 'selenium', 'python', 'java', 'automation', 'jira', 'agile', 'sql', 'jenkins', 'cypress'],
            'security': ['network security', 'python', 'penetration testing', 'firewalls', 'cryptography', 'linux', 'cloud security', 'incident response', 'ethical hacking', 'siem'],
            'data': ['python', 'sql', 'spark', 'hadoop', 'aws', 'etl', 'data warehousing', 'machine learning', 'pandas', 'scala'],
            'devops': ['docker', 'kubernetes', 'aws', 'jenkins', 'linux', 'terraform', 'python', 'git', 'ansible', 'ci/cd'],
            'database': ['sql', 'oracle', 'mysql', 'postgresql', 'mongodb', 'backup recovery', 'performance tuning', 'linux', 'data modeling', 'security'],
            'embedded': ['c', 'c++', 'python', 'linux', 'rtos', 'microcontrollers', 'arm', 'iot', 'embedded systems', 'debugging']
        }
        
        for key, default_skills in default_skills_map.items():
            if key in role:
                skills = default_skills
                break
        
        # If still no skills, use a generic message
        if not skills:
            return jsonify({"skills": [], "message": f"No specific skills found for {role}. Please try another role."})
    
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

# ===============================
# EDUCATION-INDUSTRY PIPELINE API
# ===============================

_edu_analysis_cache = {"data": None, "timestamp": None}

def _run_edu_pipeline(force_refresh=False):
    global _edu_analysis_cache

    if not force_refresh and _edu_analysis_cache["data"] is not None:
        return _edu_analysis_cache["data"]

    try:
        logger.info("Running education pipeline...")
        
        if not os.path.exists("Data.csv"):
            logger.error("Data.csv not found")
            return {
                "current_university": {"name": "No Data", "score": 0},
                "market_benchmark": 0,
                "university_scores": [],
                "insights": [{"type": "error", "title": "Data File Missing", 
                             "description": "Data.csv file not found. Please ensure the curriculum data file is present."}],
                "cosine_available": False
            }
        
        curriculum_df = pd.read_csv("Data.csv")
        logger.info(f"Loaded Data.csv with {len(curriculum_df)} rows")
        
        curriculum_df["Skills"] = curriculum_df["Skills"].apply(clean_skills_cell)
        
        try:
            market_skills = market_demand_skills()
            logger.info(f"Loaded {len(market_skills)} market skills")
        except Exception as e:
            logger.error(f"Error loading market skills: {e}")
            market_skills = set()
        
        try:
            market_df = pd.read_csv("extracted_skills.csv")
            logger.info("Loaded extracted_skills.csv")
        except Exception as e:
            logger.error(f"Error loading extracted_skills.csv: {e}")
            market_df = pd.DataFrame()
        
        jaccard_df = jaccard_relavance(curriculum_df, market_skills)
        jaccard_df["Jaccard_Pct"] = jaccard_df["Jaccard_Score"].apply(
            lambda s: round(float(s) * 100, 2)
        )
        
        cosine_available = os.path.isdir("custom_it_curriculum_model")
        logger.info(f"Cosine model available: {cosine_available}")
        
        if cosine_available and not market_df.empty:
            try:
                cosine_df = cosine_relavance(curriculum_df, market_df)
                cosine_lookup = {
                    str(row["University"]): round(float(row["Relevance_Score_%"]), 2)
                    for _, row in cosine_df.iterrows()
                }
            except Exception as e:
                logger.error(f"Error running cosine relevance: {e}")
                cosine_lookup = {}
        else:
            cosine_lookup = {}
        
        university_scores = []
        for _, row in jaccard_df.iterrows():
            uni_name = str(row["University"])
            university_scores.append({
                "university": uni_name,
                "jaccard": round(float(row["Jaccard_Pct"]), 2),
                "cosine": cosine_lookup.get(uni_name, None),
            })
        
        sort_key = "cosine" if cosine_available else "jaccard"
        university_scores.sort(key=lambda u: u[sort_key] or 0, reverse=True)
        
        if university_scores:
            top = university_scores[0]
            current_university = {
                "name": top["university"],
                "jaccard": top["jaccard"],
                "cosine": top["cosine"],
                "score": top["cosine"] if top["cosine"] is not None else top["jaccard"],
            }
            
            top2 = [u["cosine"] or u["jaccard"] for u in university_scores[:2]]
            market_benchmark = round(sum(top2) / len(top2), 2) if top2 else 0.0
        else:
            current_university = {"name": "No Data", "score": 0}
            market_benchmark = 0
        
        insights = _generate_insights(jaccard_df, market_skills, "Jaccard_Pct")
        
        result = {
            "current_university": current_university,
            "market_benchmark": market_benchmark,
            "university_scores": university_scores,
            "insights": insights,
            "cosine_available": cosine_available,
        }
        
        _edu_analysis_cache["data"] = result
        _edu_analysis_cache["timestamp"] = datetime.now().isoformat()
        logger.info("Education pipeline completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in education pipeline: {e}")
        logger.error(traceback.format_exc())
        return {
            "current_university": {"name": "Error", "score": 0},
            "market_benchmark": 0,
            "university_scores": [],
            "insights": [{"type": "error", "title": "Pipeline Error", 
                         "description": f"An error occurred: {str(e)}"}],
            "cosine_available": False
        }

def _generate_insights(results_df, market_skills, score_col):
    """Derive structured insight cards from pipeline output."""
    insights = []
    
    try:
        per_uni_matched = {}
        for _, row in results_df.iterrows():
            raw = str(row.get("Matched_Skills", ""))
            matched = {s.strip().lower() for s in raw.split(",") if s.strip()}
            per_uni_matched[str(row["University"])] = matched
        
        all_matched = set().union(*per_uni_matched.values()) if per_uni_matched else set()
        
        import re
        def _tokenise(phrase):
            return {w.lower() for w in re.findall(r"[a-zA-Z0-9#+.]+", phrase)}
        
        global_gaps = [
            phrase for phrase in market_skills
            if not _tokenise(phrase) & all_matched
        ]
        global_gaps.sort()
        
        if len(per_uni_matched) > 1:
            universal_strengths = set.intersection(*per_uni_matched.values())
        else:
            universal_strengths = next(iter(per_uni_matched.values()), set())
        
        if global_gaps:
            sample = [g.title() for g in global_gaps[:5]]
            insights.append({
                "type": "skill_gap",
                "title": "Missing Skills Across All Curricula",
                "description": (
                    f"The following market-demanded skills are absent from every university "
                    f"curriculum analysed: {', '.join(sample)}. "
                    f"A total of {len(global_gaps)} such gaps were identified."
                ),
            })
        
        if len(results_df) > 1:
            lowest_row = results_df.iloc[-1]
            lowest_name = str(lowest_row["University"])
            lowest_gaps = [
                phrase for phrase in market_skills
                if not _tokenise(phrase) & per_uni_matched.get(lowest_name, set())
            ]
            lowest_gaps.sort()
            if lowest_gaps:
                sample = [g.title() for g in lowest_gaps[:4]]
                insights.append({
                    "type": "skill_gap",
                    "title": f"Largest Alignment Gap: {lowest_name}",
                    "description": (
                        f"{lowest_name} has the lowest curriculum relevance score "
                        f"({round(float(lowest_row[score_col]), 1)}%). "
                        f"Key missing areas include: {', '.join(sample)}."
                    ),
                })
        
        if universal_strengths:
            sample = sorted(universal_strengths)[:5]
            insights.append({
                "type": "strength",
                "title": "Universally Strong Curriculum Areas",
                "description": (
                    f"All universities show consistent coverage in: "
                    f"{', '.join(s.title() for s in sample)}. "
                    f"These topics align well with current industry demand."
                ),
            })
        
        if len(results_df) > 0:
            top_name = str(results_df.iloc[0]["University"])
            top_matched = per_uni_matched.get(top_name, set())
            others_union = set().union(
                *[v for k, v in per_uni_matched.items() if k != top_name]
            ) if len(per_uni_matched) > 1 else set()
            unique_to_top = top_matched - others_union
            if unique_to_top:
                sample = sorted(unique_to_top)[:4]
                insights.append({
                    "type": "strength",
                    "title": f"Best Performer: {top_name}",
                    "description": (
                        f"{top_name} leads the ranking with exclusive coverage of: "
                        f"{', '.join(s.title() for s in sample)}, "
                        f"giving it a competitive edge over peers."
                    ),
                })
        
        avg_score = results_df[score_col].mean()
        insights.append({
            "type": "recommendation",
            "title": "Prioritise High-Impact Missing Skills",
            "description": (
                f"Average curriculum alignment currently sits at {avg_score:.1f}%. "
                f"Introducing modules focused on the identified gap areas could "
                f"lift the average score by an estimated 10–15 percentage points."
            ),
        })
        
        if global_gaps:
            sample = [g.title() for g in global_gaps[:3]]
            insights.append({
                "type": "recommendation",
                "title": "Recommended New Curriculum Modules",
                "description": (
                    f"Consider adding dedicated modules for: {', '.join(sample)}. "
                    f"These skills are consistently in demand by industry but "
                    f"underrepresented across all current course offerings."
                ),
            })
    
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        insights.append({
            "type": "error",
            "title": "Analysis Error",
            "description": "Unable to generate insights due to an internal error."
        })
    
    return insights

@app.route("/api/analysis")
def api_analysis():
    """
    JSON endpoint consumed by script.js.
    
    Query params
    ------------
    refresh=true  — bypass cache and re-run the full pipeline.
    """
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Authentication required"}), 401

    force_refresh = request.args.get("refresh", "false").lower() == "true"

    try:
        result = _run_edu_pipeline(force_refresh=force_refresh)
        return jsonify({"status": "ok", "data": result})
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return jsonify({"status": "error", "message": f"Data file not found: {e}"}), 500
    except Exception as e:
        logger.error(f"[/api/analysis] Pipeline error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {e}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    # Print startup information
    print("\n" + "="*60)
    print("SkillSync Application Starting...")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Data.csv exists: {os.path.exists('Data.csv')}")
    print(f"extracted_skills.csv exists: {os.path.exists('extracted_skills.csv')}")
    print(f"custom_it_curriculum_model exists: {os.path.exists('custom_it_curriculum_model')}")
    print(f"Model files exist: {os.path.exists('saved_models/best_rf_model.pkl')}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)