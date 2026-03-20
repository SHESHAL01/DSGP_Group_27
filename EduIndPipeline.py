import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, evaluation, util
from sentence_transformers.datasets import DenoisingAutoEncoderDataset

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


def mock_market_demand():
    # Test "Market Demand Analyzer" output
    skill_df = pd.read_csv('skill_Data.csv')
    market_demand_skills_set = set()

    for row in skill_df['Skills']:
        if isinstance(row, str):
            # Split each row by comma, strip whitespace, and convert to lowercase
            skills = [s.strip().lower() for s in row.split(',')]
            market_demand_skills_set.update(skills)

    market_demand_skills = list(market_demand_skills_set)
    return market_demand_skills

def sbert_model(df,market_df):
    # ─────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    # ─────────────────────────────────────────────
    # PREPARE TRAINING CORPUS
    # ─────────────────────────────────────────────
    df = df.dropna(subset=['Skills'])
    #market_df = market_df.dropna(subset=['Skills'])

    all_sentences = df['Skills'].tolist()
    print(f"[INFO] Total skill rows loaded: {len(all_sentences)}")

    # ─────────────────────────────────────────────
    # MILESTONE 2 — TRAIN / TEST SPLIT  (80 / 20)
    # ─────────────────────────────────────────────
    train_sentences, test_sentences = train_test_split(
        all_sentences,
        test_size=0.20,
        random_state=42
    )
    print(f"[INFO] Train sentences : {len(train_sentences)}")
    print(f"[INFO] Test  sentences : {len(test_sentences)}")

    # ─────────────────────────────────────────────
    # BUILD THE BASE MODEL
    # ─────────────────────────────────────────────
    model_name = 'bert-base-uncased'

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode='mean'  # Mean pooling is standard for TSDAE
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # ─────────────────────────────────────────────
    # TRAINING DATA  (TSDAE — reconstruction task)
    # ─────────────────────────────────────────────
    train_dataset = DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=False  # Separate weights → better TSDAE quality
    )

    # ─────────────────────────────────────────────
    # MILESTONE 1 — BUILT-IN EVALUATOR
    #   Strategy: corrupt the held-out test sentences
    #   exactly as TSDAE does (random token deletion),
    #   then measure cosine-similarity between the
    #   encoder's (noisy) and (clean) embeddings.
    #   A well-trained model should produce similar
    #   vectors for both, so the score rises with
    #   training quality.  The best checkpoint is
    #   saved automatically via save_best_model=True.
    # ─────────────────────────────────────────────
    def corrupt_sentences(sentences: list[str], del_ratio: float = 0.60) -> list[str]:
        """
        Mimic TSDAE's default corruption: randomly delete ~60 % of tokens.
        Used to create (noisy, clean) evaluation pairs from the test split.
        """
        rng = np.random.default_rng(seed=42)
        corrupted = []
        for sent in sentences:
            words = sent.split()
            if len(words) <= 1:
                corrupted.append(sent)
                continue
            kept = [w for w in words if rng.random() > del_ratio]
            if not kept:  # Guarantee at least one token
                kept = [words[rng.integers(len(words))]]
            corrupted.append(' '.join(kept))
        return corrupted

    test_noisy = corrupt_sentences(test_sentences)
    test_labels = [1.0] * len(test_sentences)  # Every pair should score ≈ 1.0

    # EmbeddingSimilarityEvaluator computes Pearson / Spearman / cosine-similarity
    # between sentence pairs and returns a scalar score after every evaluation step.
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=test_noisy,
        sentences2=test_sentences,
        scores=test_labels,
        name='tsdae-test-eval',
        show_progress_bar=False
    )

    # ─────────────────────────────────────────────
    # FINE-TUNING  — AUTO BEST-EPOCH SELECTION
    #   • evaluation_steps = steps_per_epoch  → evaluator runs once per epoch
    #   • save_best_model = True              → only the best-scoring checkpoint
    #                                           is kept in output_path
    #   Set MAX_EPOCHS to 5 (or higher on GPU);
    #   the evaluator will crown the best epoch automatically.
    # ─────────────────────────────────────────────
    MAX_EPOCHS = 5
    steps_per_epoch = len(train_dataloader)

    print(f"\n[INFO] Starting TSDAE fine-tuning — up to {MAX_EPOCHS} epochs ...")
    print(f"[INFO] Evaluator will run every {steps_per_epoch} steps (once per epoch).\n")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=MAX_EPOCHS,
        evaluator=evaluator,
        evaluation_steps=steps_per_epoch,  # ← one eval per full epoch
        save_best_model=True,  # ← only best epoch is persisted
        show_progress_bar=True,
        output_path='custom_it_curriculum_model'
    )

    print("\n[INFO] Fine-tuning complete.")
    print("[INFO] Best epoch model saved to 'custom_it_curriculum_model'.")
    print("[INFO] Check tsdae-test-eval_results.csv for per-epoch scores.")


def Similarity_Measures(market_demand_skills):
    # Initialize Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Market Demand Data
    market_text = ", ".join(market_demand_skills)
    print(market_text)
    market_embeddings = model.encode(market_text, convert_to_tensor=True)

    # Similarity Model Test_Data
    df_features = pd.read_csv('Data.csv')

    def get_scores(course_skills):
        if not course_skills:
            return pd.Series([0.0, 0.0, 0.0, 0.0])

        course_text = ", ".join(course_skills)

        # --- A. Cosine Similarity (SBERT) ---
        course_emb = model.encode(course_text, convert_to_tensor=True)
        cos_score = util.cos_sim(course_emb, market_embeddings).item()

        # --- B. Jaccard Similarity ---
        set_c = set([s.lower() for s in course_skills])
        set_m = set([s.lower() for s in market_demand_skills])
        intersection = len(set_c.intersection(set_m))
        union = len(set_c.union(set_m))
        jaccard_score = (intersection / union if union > 0 else 0.0) * 100

        # --- C. Levenshtein / Fuzzy Match --
        fuzzy_score = fuzz.token_sort_ratio(course_text, market_text) / 100.0

        return pd.Series([cos_score, jaccard_score, fuzzy_score])

    # APPLYING TO DATASET
    print("COMPARING SKILLS USING MULTIPLE MODELS...")

    # Create columns for each method
    method_cols = ['Cosine_Score', 'Jaccard_Score', 'Fuzzy_Score']
    df_features[method_cols] = df_features['Skills'].apply(get_scores)

    # CALCULATING UNIVERSITY RELEVANCE SCORES
    uni_relevance_report = df_features.groupby('University')[method_cols].mean() * 100

    uni_relevance_report.columns = [
        'Cosine_Relevance_%',
        'Jaccard_Relevance_%',
        'Fuzzy_Relevance_%'
    ]

    df_uni_score = uni_relevance_report.reset_index()

    print("\n--- FINAL CURRICULUM RELEVANCE SCORES PER UNIVERSITY ---")
    print(df_uni_score.to_markdown(index=False))
    return df_uni_score


def plot_charts(df_uni_score, df_features, market_demand_skills):
    df_uni_score.plot(x='University', y=['Cosine_Relevance_%', 'Jaccard_Relevance_%', 'Fuzzy_Relevance_%'],
                      kind='bar', figsize=(12, 6))

    plt.title("Curriculum Relevance Comparison by Similarity Method")
    plt.ylabel("Alignment Score (%)")
    plt.legend(title="ML Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Insight: Identify Skill Gaps
    all_extracted_skills = [skill for sublist in df_features['Skills'] for skill in sublist]
    taught_skills_unique = set([s.lower() for s in all_extracted_skills])
    market_skills_unique = set([s.lower() for s in market_demand_skills])

    # Skills in market but NOT in curriculum
    skill_gap = market_skills_unique - taught_skills_unique

    print("\n--- INSIGHTS & RECOMMENDATIONS ---")
    print(f"Identified Skill Gaps: {list(skill_gap)}")
    if skill_gap:
        print(f"Recommendation: Consider adding modules for {', '.join(list(skill_gap)[:3])} to improve alignment")

def main():
    data = pd.read_csv('data.csv')

    df = pd.DataFrame(data)
    #df.head()
    #DataAnalysis(df)
    #clean_data(df)
    #DataAnalysis(df)
    #word_count_analysis(df)
    #df = feature_extraction(df)
    mock_data = mock_market_demand()

    sbert_model(df,mock_data)

    #uni_score = Similarity_Measures(mock_data)
    #plot_charts(uni_score, df, mock_data)


if __name__ == '__main__':
    main()