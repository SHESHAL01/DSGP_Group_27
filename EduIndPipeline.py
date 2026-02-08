import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
from tqdm.notebook import tqdm
from itertools import product

def DataAnalysis(df):
    print("DATASET OVERVIEW")
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Records: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print("\nColumn Names and Types:")
    print(df.dtypes)
    
    print("\nSAMPLE DATA")
    print(df.head())
    
    print("\nSTATISTICAL SUMMARY")
    print(df.describe(include='all'))
    
    print("\nMISSING VALUES ANALYSIS")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\nUNIQUE VALUES PER COLUMN")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    print("\nUNIVERSITY DISTRIBUTION")
    print(df['University'].value_counts())
    
    print("\nDEGREE PROGRAM DISTRIBUTION")
    print(df['Degree Program'].value_counts())
    
    print("\nYEAR DISTRIBUTION")
    print(df['Year'].value_counts().sort_index())

def clean_data(df):
    # Removing duplicates and Nan values
    df = df.dropna()
    df = df.drop_duplicates(keep='first')

    # Fixing inconsistent data
    df['Degree Program'] = df['Degree Program'].str.strip().str.lower().str.capitalize()
    df['Course Name'] = df['Course Name'].str.strip().str.lower().str.capitalize()

def word_count_analysis(df):
    print("COURSE NAME ANALYSIS")
    course_name_lengths = df['Course Name'].str.len()
    print(f"Average Course Name Length: {course_name_lengths.mean():.2f} characters")
    print(f"Min Length: {course_name_lengths.min()}")
    print(f"Max Length: {course_name_lengths.max()}")

    word_counts = df['Course Name'].str.split().str.len()
    print(f"\nAverage Words in Course Name: {word_counts.mean():.2f}")

def feature_extraction(df):
    df_features = df.copy()

    print("\nENCODING CATEGORICAL VARIABLES")

    le_university = LabelEncoder()
    le_degree = LabelEncoder()

    df_features['University_Encoded'] = le_university.fit_transform(df_features['University'])
    df_features['Degree_Encoded'] = le_degree.fit_transform(df_features['Degree Program'])

    return df_features

def skill_extraction(df):
    # Load and Process CSV
    # flatten the CSV into a single list of unique, lowercase skills
    skill_df = pd.read_csv('skill_Data.csv')
    master_skill_set = set()

    for row in skill_df['Skills']:
        if isinstance(row, str):
            # Split each row by comma, strip whitespace, and convert to lowercase
            skills = [s.strip().lower() for s in row.split(',')]
            master_skill_set.update(skills)

    print(f"Loaded {len(master_skill_set)} unique skills from the CSV dictionary.")
    print(master_skill_set)
    return master_skill_set

def initialize_model():
    # Convert set to list for indexing
    master_skill_set = skill_extraction()
    master_skill_list = list(master_skill_set)
    print(f"Indexing {len(master_skill_list)} skills for mapping...")

    # Initialize tqdm for pandas
    tqdm.pandas(desc="Mapping Skills")

    # Load SBERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load Word2Vec
    print("Loading Word2Vec model (this may take a moment)...")
    w2v_model = api.load("glove-wiki-gigaword-50")

# --- SCORING FUNCTIONS ---
def get_sbert_scores(course_text, skill_list):
    """Returns top matches using Sentence-BERT cosine similarity"""
    # Encode the single course and all skills
    course_emb = sbert_model.encode(course_text, convert_to_tensor=True)
    skill_embs = sbert_model.encode(skill_list, convert_to_tensor=True)

    # Calculate cosine similarity
    scores = util.cos_sim(course_emb, skill_embs)[0].cpu().numpy()
    return scores

def get_jaccard_scores(course_text, skill_list):
    """Returns scores based on word overlap (Set Intersection / Union)"""
    course_tokens = set(word_tokenize(course_text.lower()))
    scores = []

    for skill in skill_list:
        skill_tokens = set(word_tokenize(skill.lower()))
        if not skill_tokens:
            scores.append(0.0)
            continue

        # Jaccard Formula
        intersection = course_tokens.intersection(skill_tokens)
        union = course_tokens.union(skill_tokens)
        score = len(intersection) / len(union) if len(union) > 0 else 0.0
        scores.append(score)

    return np.array(scores)

def get_w2v_scores(course_text, skill_list):
    """Returns scores using averaged Word2Vec embeddings"""
    def get_avg_vector(text):
        tokens = word_tokenize(text.lower())
        vectors = [w2v_model[word] for word in tokens if word in w2v_model]
        if not vectors:
            return np.zeros(50) # Return zero vector if no words found
        return np.mean(vectors, axis=0)

    course_vec = get_avg_vector(course_text)
    scores = []

    for skill in skill_list:
        skill_vec = get_avg_vector(skill)
        # Calculate Cosine Similarity between vectors
        if np.linalg.norm(course_vec) == 0 or np.linalg.norm(skill_vec) == 0:
            scores.append(0.0)
        else:
            sim = cosine_similarity([course_vec], [skill_vec])[0][0]
            scores.append(sim)

    return np.array(scores)

# --- EVALUATION METRICS ---
def calculate_ranking_metrics(predicted, actual, k=5):
    """
    Calculates ranking metrics for a single prediction.
    """
    actual_set = set([s.lower().strip() for s in actual])
    pred_k = [p.lower().strip() for p in predicted[:k]]

    hits = [1 if p in actual_set else 0 for p in pred_k]
    num_hits = sum(hits)

    # 1. Precision@K
    pk = num_hits / k

    # 2. Recall@K
    rk = num_hits / len(actual_set) if len(actual_set) > 0 else 0

    # 3. F1@K
    f1k = (2 * pk * rk) / (pk + rk) if (pk + rk) > 0 else 0

    # 4. Reciprocal Rank (RR) for MRR
    rr = 0
    for i, h in enumerate(hits):
        if h == 1:
            rr = 1 / (i + 1)
            break

    # 5. Average Precision (AP) for MAP
    ap = 0
    running_hits = 0
    for i, h in enumerate(hits):
        if h == 1:
            running_hits += 1
            ap += running_hits / (i + 1)
    ap = ap / len(actual_set) if len(actual_set) > 0 else 0

    return pk, rk, f1k, rr, ap

# --- HYBRID SCORING WITH WEIGHTS ---
def get_hybrid_top_skills(course_name, master_skill_list, weights, top_k=5):
    w_s, w_w, w_j = weights

    # Use your existing functions (ensure these are defined in your notebook)
    sbert_scores = get_sbert_scores(course_name, master_skill_list)
    jaccard_scores = get_jaccard_scores(course_name, master_skill_list) # Use your Jaccard logic
    w2v_scores = get_w2v_scores(course_name, master_skill_list)

    # Min-Max Normalization to make scores comparable (0 to 1)
    def norm(s):
        return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)

    final_scores = (w_s * norm(sbert_scores)) + \
                   (w_w * norm(w2v_scores)) + \
                   (w_j * norm(jaccard_scores))

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    return [master_skill_list[i] for i in top_indices]


def main():
    data = pd.read_csv('data.csv')
    df = pd.DataFrame(data)
    df.head()
    print("------------------")
    print("Before Cleaning")
    print("------------------")
    DataAnalysis(df)
    print("------------------")
    print("\nAfter Cleaning")
    print("------------------")
    clean_data(df)
    DataAnalysis(df)
    word_count_analysis(df)
    df = feature_extraction(df)


if __name__ == '__main__':
    main ()