import numpy as np
import pandas as pd
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from Skill_Extraction import skill_extraction

# Convert set to list for indexing
master_skill_set = skill_extraction()
master_skill_list = list(master_skill_set)
print(f"Indexing {len(master_skill_list)} skills for mapping...")

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
    """Returns scores based on word overlap"""
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

        if np.linalg.norm(course_vec) == 0 or np.linalg.norm(skill_vec) == 0:
            scores.append(0.0)
        else:
            sim = cosine_similarity([course_vec], [skill_vec])[0][0]
            scores.append(sim)

    return np.array(scores)

def map_skills_hybrid(course_name, top_k=5):
    # Get scores from all methods
    scores_sbert = get_sbert_scores(course_name, master_skill_list)
    scores_jaccard = get_jaccard_scores(course_name, master_skill_list)
    scores_w2v = get_w2v_scores(course_name, master_skill_list)

    # Normalize Scores
    def normalize(scores):
        if np.max(scores) == np.min(scores): return scores
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    norm_sbert = normalize(scores_sbert)
    norm_w2v = normalize(scores_w2v)

    # Weighted Ensemble Score
    final_scores = (0.7 * norm_sbert) + (0.2 * norm_w2v) + (0.1 * scores_jaccard)

    # Get Top K Indices
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    # Return the actual skill names
    return [master_skill_list[i] for i in top_indices]