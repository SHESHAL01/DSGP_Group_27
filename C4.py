import random
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import os

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ------------------- CONFIG -------------------
DATA_PATH = "C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_SAVE_PATH = "saved_course_model"
EMBEDDINGS_SAVE_PATH = "course_embeddings.npy"

EPOCHS = 4
WARMUP_RATIO = 0.1
BATCH_SIZE = 32

# ------------------- LOAD DATA -------------------
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")
except Exception as e:
    raise Exception(f"Error reading CSV file: {e}")

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
    lambda r: f"{str(r.get('Title',''))}. Skills: {' '.join(r['skills_list'])}",
    axis=1
)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ------------------- CREATE POSITIVE PAIRS -------------------
def create_positive_pairs(data, min_overlap=2):
    examples = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            skills_i = set(data.iloc[i]["skills_list"])
            skills_j = set(data.iloc[j]["skills_list"])
            overlap = len(skills_i & skills_j)
            if overlap >= min_overlap:
                examples.append(
                    InputExample(
                        texts=[
                            data.iloc[i]["combined_text"],
                            data.iloc[j]["combined_text"]
                        ]
                    )
                )
    return examples

# ------------------- TRAIN OR LOAD MODEL -------------------
try:
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading saved model...")
        model = SentenceTransformer(MODEL_SAVE_PATH)
    else:
        print("Training model...")
        train_examples = create_positive_pairs(train_df, min_overlap=2)
        print("Total training pairs:", len(train_examples))

        model = SentenceTransformer(BASE_MODEL)

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=BATCH_SIZE
        )

        train_loss = losses.MultipleNegativesRankingLoss(model)

        total_steps = len(train_dataloader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=warmup_steps,
            show_progress_bar=True
        )

        model.save(MODEL_SAVE_PATH)
        print("Model saved successfully!")
except Exception as e:
    raise Exception(f"Error in model training/loading: {e}")

# ------------------- RETRIEVAL EVALUATION -------------------
def evaluate_model_with_metrics(model, test_df, sample_size=20000):
    try:
        texts = test_df["combined_text"].tolist()
        skills = test_df["skills_list"].tolist()

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        y_true = []
        y_scores = []
        n = len(texts)

        positives, negatives = [], []

        while len(positives) < sample_size // 2 or len(negatives) < sample_size // 2:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if i == j:
                continue

            label = 1 if set(skills[i]) & set(skills[j]) else 0
            score = np.dot(embeddings[i], embeddings[j])

            if label == 1 and len(positives) < sample_size // 2:
                positives.append((label, score))
            elif label == 0 and len(negatives) < sample_size // 2:
                negatives.append((label, score))

        combined = positives + negatives
        random.shuffle(combined)

        for label, score in combined:
            y_true.append(label)
            y_scores.append(score)

        ap = average_precision_score(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        thresholds = np.linspace(-1, 1, 100)
        best_f1 = 0
        best_threshold = 0
        for t in thresholds:
            preds = [1 if s >= t else 0 for s in y_scores]
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        final_preds = [1 if s >= best_threshold else 0 for s in y_scores]

        precision = precision_score(y_true, final_preds)
        recall = recall_score(y_true, final_preds)

        cm = confusion_matrix(y_true, final_preds)

        print("\n====== Evaluation Results ======")
        print(f"Average Precision (AP): {ap:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Best F1: {best_f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Best Threshold: {best_threshold:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, final_preds))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

        return {
            "AP": ap,
            "ROC-AUC": roc_auc,
            "F1": best_f1,
            "Precision": precision,
            "Recall": recall
        }
    except Exception as e:
        raise Exception(f"Error during evaluation: {e}")

try:
    evaluate_model_with_metrics(model, test_df)
except Exception as e:
    print(e)

# ------------------- BUILD OR LOAD EMBEDDINGS -------------------
try:
    if os.path.exists(EMBEDDINGS_SAVE_PATH):
        print("Loading saved embeddings...")
        course_embeddings = np.load(EMBEDDINGS_SAVE_PATH)
    else:
        print("Generating course embeddings...")
        course_embeddings = model.encode(
            df["combined_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        np.save(EMBEDDINGS_SAVE_PATH, course_embeddings)
        print("Embeddings saved!")
except Exception as e:
    raise Exception(f"Error in embeddings processing: {e}")

# ------------------- RETRIEVAL FUNCTION -------------------
def retrieve_top_k_courses(model, df, embeddings, query_skill, k=5):
    try:
        query_embedding = model.encode(
            [query_skill],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        sims = np.dot(embeddings, query_embedding.T).flatten()
        top_k_idx = np.argsort(-sims)[:k]
        return df.iloc[top_k_idx][["Title", "Url"]]
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return pd.DataFrame(columns=["Title", "Url"])

# ------------------- EXAMPLE QUERY -------------------
skill_query = "Maths"

top_courses = retrieve_top_k_courses(
    model,
    df,
    course_embeddings,
    skill_query,
    k=5
)

print("\nTop 5 Courses for:", skill_query)
print(top_courses)