import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Model paths
MODEL_PATH = "saved_course_model"
EMBEDDINGS_PATH = "course_embeddings.npy"
DATA_PATH = "final_DS.csv"

# Global variables for lazy loading
_model = None
_df = None
_embeddings = None

def load_course_data():
    """Load course data from final_DS.csv"""
    global _df
    if _df is None:
        try:
            _df = pd.read_csv(DATA_PATH)
            print(f"Loaded {len(_df)} courses from {DATA_PATH}")
        except Exception as e:
            print(f"Error loading course data: {e}")
            _df = pd.DataFrame(columns=['Title', 'Url'])
    return _df

def load_embeddings():
    """Load precomputed course embeddings"""
    global _embeddings
    if _embeddings is None:
        try:
            if os.path.exists(EMBEDDINGS_PATH):
                _embeddings = np.load(EMBEDDINGS_PATH)
                print(f"Loaded embeddings for {_embeddings.shape[0]} courses")
            else:
                print(f"Embeddings file not found at {EMBEDDINGS_PATH}")
                _embeddings = np.array([])
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            _embeddings = np.array([])
    return _embeddings

def get_model():
    """Load the sentence transformer model"""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_PATH)
            print("Loaded sentence transformer model")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a default model if custom model not found
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Using fallback model: all-MiniLM-L6-v2")
    return _model

def retrieve_top_k_courses(query_skill, k=5):
    """Get top k courses for a skill using semantic search"""
    df = load_course_data()
    embeddings = load_embeddings()
    
    if df.empty or len(embeddings) == 0:
        return []
    
    model = get_model()
    
    # Clean the skill name (remove underscores)
    clean_query = query_skill.replace('_', ' ')
    
    # Encode the query skill
    query_embedding = model.encode(
        [clean_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Calculate similarities
    sims = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top k indices
    top_k_idx = np.argsort(-sims)[:k]
    
    # Return results as list of dictionaries
    results = df.iloc[top_k_idx][["Title", "Url"]].to_dict(orient="records")
    return results

def get_course_recommendations(missing_skills, job_role, top_n=5):
    """
    Get course recommendations based on missing skills
    
    Args:
        missing_skills: List of missing skill names (from your model)
        job_role: The target job role
        top_n: Number of recommendations per skill
    
    Returns:
        Dictionary with skill as key and list of course recommendations
    """
    recommendations = {}
    
    if not missing_skills:
        return recommendations
    
    # Limit to top 5 missing skills
    for skill in missing_skills[:5]:
        # Clean the skill name for display
        clean_skill = skill.replace('_', ' ').title()
        
        # Get course recommendations
        recs = retrieve_top_k_courses(skill, k=top_n)
        
        if recs:
            recommendations[clean_skill] = recs
    
    return recommendations