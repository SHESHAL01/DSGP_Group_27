import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, evaluation, util
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
import torch
from tqdm import tqdm
from torch.optim import AdamW

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

def test_Sbert(df):
    # ─────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # ─────────────────────────────────────────────
    # PREPARE DATA  —  80 / 20 SPLIT
    # ─────────────────────────────────────────────
    df = df.dropna(subset=['Skills'])
    #market_df = market_df.dropna(subset=['Skills'])

    all_sentences = df['Skills'].tolist()

    train_sentences, val_sentences = train_test_split(
        all_sentences,
        test_size=0.20,
        random_state=42
    )
    print(f"[INFO] Train: {len(train_sentences)} | Validation: {len(val_sentences)}")

    # ─────────────────────────────────────────────
    # BUILD THE SENTENCE TRANSFORMER MODEL
    # ─────────────────────────────────────────────
    model_name = 'bert-base-uncased'

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode='mean'
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.to(device)

    # ─────────────────────────────────────────────
    # BUILD DATALOADERS
    #   smart_batching_collate converts InputExample
    #   objects into (features, labels) tensor tuples
    #   that the loss function consumes directly.
    # ─────────────────────────────────────────────
    BATCH_SIZE = 8

    train_dataloader = DataLoader(
        DenoisingAutoEncoderDataset(train_sentences),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=model.smart_batching_collate  # ← critical: formats batches correctly
    )

    val_dataloader = DataLoader(
        DenoisingAutoEncoderDataset(val_sentences),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=model.smart_batching_collate
    )

    # ─────────────────────────────────────────────
    # BUILD THE TSDAE LOSS
    #   The decoder lives inside the loss object, so
    #   both model + loss_fn must be moved to device
    #   and their parameters included in the optimizer.
    # ─────────────────────────────────────────────
    loss_fn = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=False
    )
    loss_fn.to(device)

    # ─────────────────────────────────────────────
    # OPTIMIZER
    #   Combine encoder (model) + decoder (loss_fn)
    #   parameters so both are updated each step.
    # ─────────────────────────────────────────────
    optimizer = AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=3e-5,
        weight_decay=0.01
    )

    # ─────────────────────────────────────────────
    # CUSTOM TRAINING LOOP
    #   Tracks Training Loss AND Validation Loss
    #   after every epoch so you can spot:
    #     • Underfitting  → both losses stay high
    #     • Optimal       → both decrease and level together
    #     • Overfitting   → train↓ but val turns back up
    # ─────────────────────────────────────────────
    MAX_EPOCHS = 5
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\n{'=' * 60}")
    print("       TSDAE TRAINING  —  LOSS TRACKING")
    print(f"{'=' * 60}\n")

    for epoch in range(1, MAX_EPOCHS + 1):

        # ── TRAINING PHASE ──────────────────────────────────────────
        model.train()
        loss_fn.train()
        epoch_train_loss = 0.0

        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]",
            leave=False
        )
        for features, labels in train_bar:
            # Move each feature dict's tensors to device
            features = [{k: v.to(device) for k, v in f.items()} for f in features]

            optimizer.zero_grad()
            loss = loss_fn(features, labels)  # TSDAE reconstruction loss
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train)

        # ── VALIDATION PHASE ─────────────────────────────────────────
        #   model.eval() + torch.no_grad() → no gradients computed,
        #   loss is purely a measurement of reconstruction quality.
        model.eval()
        loss_fn.eval()
        epoch_val_loss = 0.0

        val_bar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch}/{MAX_EPOCHS} [Val  ]",
            leave=False
        )
        with torch.no_grad():
            for features, labels in val_bar:
                features = [{k: v.to(device) for k, v in f.items()} for f in features]
                loss = loss_fn(features, labels)
                epoch_val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val)

        # ── EPOCH SUMMARY ────────────────────────────────────────────
        gap = avg_val - avg_train
        if avg_train > 1.5 and avg_val > 1.5:
            diagnosis = "⚠️  UNDERFITTING  — both losses high"
        elif gap > 0.3:
            diagnosis = "🔴  OVERFITTING   — val loss diverging from train"
        elif gap > 0.1:
            diagnosis = "🟡  WATCH         — small gap, monitor next epoch"
        else:
            diagnosis = "✅  OPTIMAL       — losses tracking together"

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}  |  "
            f"Train Loss: {avg_train:.4f}  |  "
            f"Val Loss: {avg_val:.4f}  |  "
            f"Gap: {gap:+.4f}  |  {diagnosis}"
        )

        # Save best checkpoint (lowest validation loss)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            model.save('custom_it_curriculum_model')
            print(f"           └─ 💾 New best model saved (val loss: {best_val_loss:.4f})")

    # ─────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Training complete.  Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: custom_it_curriculum_model/")
    print(f"{'=' * 60}")

    # ─────────────────────────────────────────────
    # PLOT  —  Training Loss vs Validation Loss
    #   Visual diagnosis of fit condition
    # ─────────────────────────────────────────────
    epochs_range = range(1, MAX_EPOCHS + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, train_losses, marker='o', linewidth=2,
             color='steelblue', label='Training Loss')
    plt.plot(epochs_range, val_losses, marker='s', linewidth=2,
             color='tomato', label='Validation Loss', linestyle='--')

    # Mark the best epoch
    plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5,
                label=f'Best Epoch ({best_epoch})')

    plt.title('TSDAE Training vs Validation Loss\n(Underfitting / Optimal / Overfitting Diagnosis)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Reconstruction Loss', fontsize=11)
    plt.xticks(epochs_range)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.35)

    # Annotation guide
    plt.figtext(
        0.13, 0.01,
        "Both high → Underfit  |  "
        "Both decrease together → Optimal  |  "
        "Val rises while Train drops → Overfit",
        fontsize=8, color='dimgray'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] Loss curve saved to 'loss_curve.png'")
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
    mock_data = mock_market_demand()

    sbert_model(df,mock_data)

    #sbert_model(df,mock_data)
    test_Sbert(df)
    #uni_score = Similarity_Measures(mock_data)
    #plot_charts(uni_score, df, mock_data)

if __name__ == '__main__':
    main()