import random
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    f1_score
)

# ------------------- LOAD DATA -------------------
df = pd.read_csv(
    "C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv"
)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------- PREPROCESSING -------------------
def parse_skills(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except:
        if pd.isna(s):
            return []
        return [tok.strip().lower() for tok in str(s).split(",") if tok.strip()]
    return []

df["skills_list"] = df["Skills"].fillna("[]").apply(parse_skills)

df["combined_text"] = df.apply(
    lambda r: f"{str(r.get('Title',''))} | {' '.join(r['skills_list'])}",
    axis=1
)

# ------------------- TRAIN / TEST SPLIT -------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# ------------------- CREATE TRAIN PAIRS -------------------
def create_training_pairs(data, negative_prob=0.05, max_pairs=100):

    examples = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            skills_i = set(data.iloc[i]["skills_list"])
            skills_j = set(data.iloc[j]["skills_list"])

            shared = skills_i & skills_j

            if shared:
                examples.append(
                    InputExample(
                        texts=[data.iloc[i]["combined_text"],
                               data.iloc[j]["combined_text"]],
                        label=1.0
                    )
                )
            elif random.random() < negative_prob:
                examples.append(
                    InputExample(
                        texts=[data.iloc[i]["combined_text"],
                               data.iloc[j]["combined_text"]],
                        label=0.0
                    )
                )

    if len(examples) > max_pairs:
        examples = random.sample(examples, max_pairs)

    return examples

# ------------------- FINE-TUNE MODEL -------------------
def fine_tune_model(train_df, epochs=3, warmup_ratio=0.1):

    train_examples = create_training_pairs(train_df)

    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )

    train_loss = losses.CosineSimilarityLoss(model)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )

    return model

# ------------------- EVALUATION FUNCTION -------------------
def evaluate_model(model, test_df, sample_size=3000):

    texts = test_df["combined_text"].tolist()
    skills = test_df["skills_list"].tolist()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    y_true = []
    y_scores = []

    n = len(texts)

    for _ in range(sample_size):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        if i == j:
            continue

        label = 1 if set(skills[i]) & set(skills[j]) else 0
        score = np.dot(embeddings[i], embeddings[j])

        y_true.append(label)
        y_scores.append(score)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Best F1 threshold
    thresholds = np.linspace(0, 1, 50)
    best_f1 = 0
    best_threshold = 0

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in y_scores]
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return {
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "embeddings": embeddings,
        "skills": skills
    }


# ------------------- RECALL@K -------------------
def recall_at_k(embeddings, skills, k=5):

    correct = 0
    total = len(embeddings)

    for i in range(total):
        sims = np.dot(embeddings, embeddings[i])
        ranked = np.argsort(-sims)

        top_k = ranked[1:k+1]

        if any(set(skills[i]) & set(skills[j]) for j in top_k):
            correct += 1

    return correct / total

# ------------------- BASELINE (NO FINE-TUNING) -------------------
print("\nEvaluating Pretrained Model (Baseline)...")
baseline_model = SentenceTransformer(BASE_MODEL)
baseline_results = evaluate_model(baseline_model, test_df)

# ------------------- FINE-TUNED MODEL -------------------
print("\nFine-tuning Model...")
fine_model = fine_tune_model(train_df, epochs=3)

print("\nEvaluating Fine-Tuned Model...")
fine_results = evaluate_model(fine_model, test_df)

# ------------------- RECALL@5 -------------------
baseline_r5 = recall_at_k(
    baseline_results["embeddings"],
    baseline_results["skills"],
    k=5
)

fine_r5 = recall_at_k(
    fine_results["embeddings"],
    fine_results["skills"],
    k=5
)

# ------------------- RESULTS -------------------
print("\n================= FINAL COMPARISON =================")
print("Baseline AP:", round(baseline_results["ap"], 4))
print("Fine-tuned AP:", round(fine_results["ap"], 4))
print()

print("Baseline ROC-AUC:", round(baseline_results["roc_auc"], 4))
print("Fine-tuned ROC-AUC:", round(fine_results["roc_auc"], 4))
print()

print("Baseline Best F1:", round(baseline_results["best_f1"], 4))
print("Fine-tuned Best F1:", round(fine_results["best_f1"], 4))
print()

print("Baseline Recall@5:", round(baseline_r5, 4))
print("Fine-tuned Recall@5:", round(fine_r5, 4))
print("====================================================")

# ------------------- PR CURVE PLOT -------------------
plt.figure()
plt.plot(
    baseline_results["recall"],
    baseline_results["precision"],
    label=f"Baseline AP={baseline_results['ap']:.3f}"
)

plt.plot(
    fine_results["recall"],
    fine_results["precision"],
    label=f"Fine-Tuned AP={fine_results['ap']:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.show()

# ------------------- ROC CURVE PLOT -------------------
plt.figure()

# Baseline ROC
plt.plot(
    baseline_results["fpr"],
    baseline_results["tpr"],
    label=f"Baseline ROC-AUC = {baseline_results['roc_auc']:.3f}"
)

# Fine-tuned ROC
plt.plot(
    fine_results["fpr"],
    fine_results["tpr"],
    label=f"Fine-Tuned ROC-AUC = {fine_results['roc_auc']:.3f}"
)

# Diagonal line (random classifier)
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
