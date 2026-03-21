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
import re
import ast

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


# STOPWORDS
STOPWORDS = {
    # Domain-specific filler (course catalog language)
    "introduction", "intro", "fundamentals", "fundamental",
    "overview", "basics", "basic", "advanced",
    # Common English stopwords
    "a", "an", "the", "and", "but", "or", "nor", "if",
    "to", "of", "in", "for", "on", "at", "by", "from",
    "with", "into", "through", "via", "using", "about", "as",
    "until", "while", "during", "before", "after", "above",
    "below", "up", "down", "out", "over", "under", "again",
    "its", "their", "our", "your", "my",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "it", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "them",
    "so", "than", "too", "very", "just", "only", "same",
    "each", "more", "most", "other", "some", "such", "no",
    "not", "both", "own", "between", "here", "there",
    "when", "where", "how", "all", "few", "then", "once",
}

# HELPER: Is a slash-token a protected compound abbreviation
def is_compound_abbreviation(token: str) -> bool:
    """Return True if all slash-parts are uppercase-only abbreviations."""
    parts = token.split("/")
    pattern = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)*$")
    return all(pattern.match(p) for p in parts)

# HELPER: Split by comma
def split_comma_outside_parens(text: str) -> list:
    """Split text on commas that sit outside of any parentheses."""
    parts, current, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


# Expand parentheses
def expand_parentheses(text: str) -> list:
    """
    Remove parentheses and extract their content.

    Examples
    --------
    'Version control (Git)'
        → ['Version control', 'Git']
    'OOP principles (encapsulation, inheritance, polymorphism, abstraction)'
        → ['OOP principles', 'encapsulation', 'inheritance', 'polymorphism', 'abstraction']
    'Data structures (lists/dicts/tuples)'
        → ['Data structures', 'lists', 'dicts', 'tuples']
    'SQL (DDL/DML/DCL)'
        → ['SQL', 'DDL', 'DML', 'DCL']
    'Software testing (unit, integration, system)'
        → ['Software testing', 'unit', 'integration', 'system']
    """
    results = []
    remaining = text

    while "(" in remaining and ")" in remaining:
        open_i  = remaining.index("(")
        close_i = remaining.index(")")

        before = remaining[:open_i].strip().rstrip(",").strip()
        inside = remaining[open_i + 1 : close_i].strip()
        after  = remaining[close_i + 1 :].strip().lstrip(",").strip()

        if before:
            results.append(before)

        # Inside: split by comma first, then by slash (always — they are alternatives)
        for chunk in re.split(r",", inside):
            chunk = chunk.strip()
            if not chunk:
                continue
            for part in chunk.split("/"):
                part = part.strip()
                if part:
                    results.append(part)

        remaining = after

    if remaining.strip():
        results.append(remaining.strip())

    return [r for r in results if r]


# Handle slashes OUTSIDE parentheses
def split_slash(skill: str) -> list:
    """
    Split a skill on non-compound slashes.

    Examples
    --------
    'Agile/Scrum'              → ['Agile', 'Scrum']
    'File I/O'                 → ['File I/O']            (I/O is compound)
    'CI/CD Pipeline'           → ['CI/CD Pipeline']      (CI/CD is compound)
    'OSI/TCP-IP model'         → ['OSI/TCP-IP model']    (both uppercase)
    'AWS/Azure/GCP'            → ['AWS', 'Azure', 'GCP'] (Azure has lowercase)
    'Sorting/searching algos'  → ['Sorting', 'searching algos']
    """
    if "/" not in skill:
        return [skill]

    def replace_slash_token(match):
        token = match.group(0)
        if is_compound_abbreviation(token):
            return token                # keep intact
        return token.replace("/", "§")  # mark for splitting

    processed = re.sub(r"\S+/\S*", replace_slash_token, skill)

    if "§" not in processed:
        return [skill]

    return [p.strip() for p in processed.split("§") if p.strip()]


# Remove stopwords from a single skill phrase
def remove_stopwords(skill: str) -> str:
    words = skill.split()
    cleaned = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(cleaned)


# MAIN CLEANING FUNCTION
def clean_skills_cell(cell) -> str:
    """Full cleaning pipeline for one cell in the Skills column."""
    if pd.isna(cell):
        return cell

    # Split on commas that are OUTSIDE parentheses (fixes comma-inside-parens bug)
    raw_skills = split_comma_outside_parens(cell)

    final_skills = []
    for skill in raw_skills:
        skill = skill.strip()
        if not skill:
            continue

        # 1. Expand parentheses → list of sub-skills
        expanded = expand_parentheses(skill)

        for part in expanded:

            # 2. Split on non-compound slashes → possibly multiple sub-skills
            split_parts = split_slash(part)

            for sp in split_parts:

                # 3. Remove stopwords
                sp = remove_stopwords(sp)

                # 4. Lowercase + collapse extra whitespace
                sp = re.sub(r"\s+", " ", sp).strip().lower()

                if sp:
                    final_skills.append(sp)

    return ", ".join(final_skills)

def parse_market_skills(raw: str) -> set:
    """
    Parse the string-encoded list
    """
    try:
        items = ast.literal_eval(raw)
        return {str(t).strip().lower() for t in items if str(t).strip()}
    except (ValueError, SyntaxError):
        raw = raw.strip("[]").replace("'", "").replace('"', "")
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

def market_demand_skills():
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
    MAX_EPOCHS = 10
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

def phrase_to_tokens(phrase: str) -> set:
    """
    Split a skill phrase into individual word tokens.
    """
    phrase = phrase.lower()
    return {word for word in re.findall(r"[a-z]+", phrase)}

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard = |A ∩ B| / |A ∪ B|"""
    if not set_a or not set_b:
        return 0.0
    return round(len(set_a & set_b) / len(set_a | set_b), 4)

def jaccard_relavance(df, market_skills):

    # ── Aggregate per-university skill tokens
    university_skills = {}

    for _, row in df.iterrows():
        uni = str(row["University"]).strip()
        raw = str(row["Skills"])

        # Split each comma-separated phrase into individual word tokens
        tokens = set()
        for phrase in raw.split(","):
            tokens |= phrase_to_tokens(phrase.strip())

        university_skills.setdefault(uni, set())
        university_skills[uni] |= tokens  # set union → no duplicates

    print(f"Universities found: {list(university_skills.keys())}\n")

    # ── Jaccard similarity per university

    results = []
    for uni, uni_tokens in university_skills.items():
        score = jaccard_similarity(uni_tokens, market_skills)
        common = uni_tokens & market_skills
        results.append({
            "University": uni,
            "Curriculum_Tokens": len(uni_tokens),
            "Market_Tokens": len(market_skills),
            "Common_Tokens": len(common),
            "Jaccard_Score": score,
            "Relevance_%": f"{score * 100:.2f}%",
            "Matched_Skills": ", ".join(sorted(common))
        })

    results_df = (pd.DataFrame(results)
                  .sort_values("Jaccard_Score", ascending=False)
                  .reset_index(drop=True))

    # Print results

    print("=" * 60)
    print("  UNIVERSITY CURRICULUM RELEVANCE  (Jaccard Similarity)")
    print("=" * 60)

    for _, row in results_df.iterrows():
        print(f"\n  University         : {row['University']}")
        print(f"  Jaccard Score      : {row['Jaccard_Score']:.4f}  ({row['Relevance_%']})")
        print(f"  Curriculum tokens  : {row['Curriculum_Tokens']}")
        print(f"  Matched with market: {row['Common_Tokens']}")
        print(f"  Matched skills     : {row['Matched_Skills'][:120]}")


def Similarity_Measures(market_demand_skills):
    # Initialize Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Market Demand Data
    market_text = ", ".join(market_demand_skills)
    print(market_text)
    market_embeddings = model.encode(market_text, convert_to_tensor=True)

    # Similarity Model Test_Data
    df_features = pd.read_csv('Output.csv')

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
    df = pd.read_csv("Data.csv")

    df["Skills"] = df["Skills"].apply(clean_skills_cell)
    #DataAnalysis(df)
    market_data = market_demand_skills()

    #test_Sbert(df)
    jaccard_relavance(df,market_data)
    #uni_score = Similarity_Measures(mock_data)
    #plot_charts(uni_score, df, mock_data)

if __name__ == '__main__':
    main()