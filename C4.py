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
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------- CONFIG -------------------
DATA_PATH = "C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_SAVE_PATH = "saved_course_model"
EMBEDDINGS_SAVE_PATH = "course_embeddings.npy"

EPOCHS = 4          # your optimal value
WARMUP_RATIO = 0.15  # your optimal value
BATCH_SIZE = 16

# ------------------- LOAD DATA -------------------
df = pd.read_csv(DATA_PATH)

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

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ------------------- CREATE TRAIN PAIRS -------------------
def create_training_pairs(data, negative_prob=0.05, max_pairs=1024000):
    examples = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            skills_i = set(data.iloc[i]["skills_list"])
            skills_j = set(data.iloc[j]["skills_list"])

            if skills_i & skills_j:
                label = 1.0
            elif random.random() < negative_prob:
                label = 0.0
            else:
                continue

            examples.append(
                InputExample(
                    texts=[data.iloc[i]["combined_text"],
                           data.iloc[j]["combined_text"]],
                    label=label
                )
            )

    if len(examples) > max_pairs:
        examples = random.sample(examples, max_pairs)

    return examples

# ------------------- TRAIN OR LOAD MODEL -------------------
if os.path.exists(MODEL_SAVE_PATH):
    print("Loading saved model...")
    model = SentenceTransformer(MODEL_SAVE_PATH)
else:
    print("Training model...")
    train_examples = create_training_pairs(train_df)

    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    train_loss = losses.CosineSimilarityLoss(model)

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

# ------------------- EVALUATION -------------------
def evaluate_model(model, test_df, sample_size=3000):

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

    for _ in range(sample_size):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        if i == j:
            continue

        label = 1 if set(skills[i]) & set(skills[j]) else 0
        score = np.dot(embeddings[i], embeddings[j])

        y_true.append(label)
        y_scores.append(score)

    ap = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Best threshold
    thresholds = np.linspace(0, 1, 50)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in y_scores]
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    final_preds = [1 if s >= best_threshold else 0 for s in y_scores]
    cm = confusion_matrix(y_true, final_preds)

    print("\nAP:", ap)
    print("ROC-AUC:", roc_auc)
    print("Best F1:", best_f1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

evaluate_model(model, test_df)

