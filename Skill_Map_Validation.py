import numpy as np
import pandas as pd
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
from itertools import product

# Convert set to list for indexing
master_skill_list = list(master_skill_set)
print(f"Indexing {len(master_skill_list)} skills for mapping...")

# Define Ground Truth
ground_truth_test = {
    "Professional english": ["communication", "writing", "presentation skills", "business english"],
    "Introductory statistics": ["data analysis", "probability", "statistics", "mathematics"],
    "Principles of management": ["leadership", "planning", "management", "organizational behavior"],
    "Computer system organization": ["hardware", "computer architecture", "assembly language", "os"],
    "Database management systems": ["sql", "mysql", "database design", "data modeling"]
}
# Load SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Word2Vec
print("Loading Word2Vec model (this may take a moment)...")
w2v_model = api.load("glove-wiki-gigaword-50")

def Bi_encoder_Map():
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
        """
        Combines SBERT, W2V, and Jaccard with specific weights.
        Weights: (w_sbert, w_w2v, w_jaccard)
        """
        w_s, w_w, w_j = weights

        # Use your existing functions
        sbert_scores = get_sbert_scores(course_name, master_skill_list)
        jaccard_scores = get_jaccard_scores(course_name, master_skill_list)
        w2v_scores = get_w2v_scores(course_name, master_skill_list)

        # Min-Max Normalization to make scores comparable (0 to 1)
        def norm(s):
            return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)

        final_scores = (w_s * norm(sbert_scores)) + \
                       (w_w * norm(w2v_scores)) + \
                       (w_j * norm(jaccard_scores))

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        return [master_skill_list[i] for i in top_indices]

    # Grid Search Cross-Validation for Weights
    def run_weight_optimization(master_skill_list, k=5):
        # Test weights in increments of 0.2
        weight_steps = np.linspace(0, 1, 6)
        best_map = -1
        best_w = None
        all_results = []

        print("Optimizing weights...")
        for w_s, w_w, w_j in product(weight_steps, weight_steps, weight_steps):
            if not np.isclose(w_s + w_w + w_j, 1.0): continue

            metrics = []
            for course, actual in ground_truth_test.items():
                preds = get_hybrid_top_skills(course, master_skill_list, (w_s, w_w, w_j), top_k=k)
                metrics.append(calculate_ranking_metrics(preds, actual, k=k))

            # Calculate Means
            avg_metrics = np.mean(metrics, axis=0)
            res = {
                'weights': (w_s, w_w, w_j),
                'P@K': avg_metrics[0], 'R@K': avg_metrics[1],
                'F1@K': avg_metrics[2], 'MRR': avg_metrics[3], 'MAP': avg_metrics[4]
            }
            all_results.append(res)

            if avg_metrics[4] > best_map: # Optimizing for MAP
                best_map = avg_metrics[4]
                best_w = (w_s, w_w, w_j)

        return pd.DataFrame(all_results), best_w

    # EXECUTE OPTIMIZATION
    results_df, optimized_weights = run_weight_optimization(master_skill_list)

    print(f"\nOptimization Complete!")
    print(f"Best Weights (SBERT, W2V, Jaccard): {optimized_weights}")
    print("\nTop 5 Weight Combinations by MAP:")
    print(results_df.sort_values('MAP', ascending=False).head(5))