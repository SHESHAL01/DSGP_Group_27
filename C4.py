import random
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

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
    except Exception:
        if pd.isna(s):
            return []
        return [tok.strip().lower() for tok in str(s).split(",") if tok.strip()]
    return []

df["skills_list"] = df["Skills"].fillna("[]").apply(parse_skills)

df["combined_text"] = df.apply(
    lambda r: f"{str(r.get('Title',''))} | {' '.join(r['skills_list'])}",
    axis=1
)

texts = df["combined_text"].tolist()
skills = df["skills_list"].tolist()

# ------------------- EXPERIMENT SETTINGS -------------------
EXPERIMENTS = [
    {"epochs": 1, "warmup_ratio": 0.05},
    {"epochs": 2, "warmup_ratio": 0.05},
    {"epochs": 3, "warmup_ratio": 0.05},
]

MAX_PAIRS = 100000
BATCH_SIZE = 16
NEGATIVE_SAMPLE_PROB = 0.05

# ------------------- FINE-TUNING FUNCTION -------------------
def fine_tune_model(epochs, warmup_ratio):

    train_examples = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            skills_i = set(df.loc[i, "skills_list"])
            skills_j = set(df.loc[j, "skills_list"])

            shared = skills_i & skills_j

            if shared:
                train_examples.append(
                    InputExample(
                        texts=[df.loc[i, "combined_text"],
                               df.loc[j, "combined_text"]],
                        label=1.0
                    )
                )
            elif random.random() < NEGATIVE_SAMPLE_PROB:
                train_examples.append(
                    InputExample(
                        texts=[df.loc[i, "combined_text"],
                               df.loc[j, "combined_text"]],
                        label=0.0
                    )
                )

    if len(train_examples) > MAX_PAIRS:
        train_examples = random.sample(train_examples, MAX_PAIRS)

    print(f"Training with {len(train_examples)} pairs")

    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
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

    return model, warmup_steps

# ------------------- PR CURVE FUNCTION -------------------
def get_pr_curve_data(model, texts, skills, sample_size=3000):

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

        shared = set(skills[i]) & set(skills[j])
        label = 1 if shared else 0

        score = np.dot(embeddings[i], embeddings[j])

        y_true.append(label)
        y_scores.append(score)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    return precision, recall, ap_score

# ------------------- RUN EXPERIMENTS -------------------
experiment_results = {}

for exp in EXPERIMENTS:

    epochs = exp["epochs"]
    warmup_ratio = exp["warmup_ratio"]

    print("\n=================================")
    print(f"Running Experiment: Epochs={epochs}")
    print("=================================")

    model, warmup_steps = fine_tune_model(epochs, warmup_ratio)

    precision, recall, ap_score = get_pr_curve_data(model, texts, skills)

    experiment_results[f"E{epochs}"] = {
        "precision": precision,
        "recall": recall,
        "ap": ap_score,
        "warmup": warmup_steps
    }

    print(f"AP Score: {ap_score:.4f}")
    print(f"Warmup Steps: {warmup_steps}")

# ------------------- PLOT ALL PR CURVES -------------------
plt.figure()

for label, data in experiment_results.items():
    plt.plot(
        data["recall"],
        data["precision"],
        label=f"{label} | AP={data['ap']:.3f}"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.show()

# ------------------- PRINT BEST MODEL -------------------
best_model_label = max(
    experiment_results,
    key=lambda x: experiment_results[x]["ap"]
)

print("\n=================================")
print(f"Best Model: {best_model_label}")
print(f"Best AP: {experiment_results[best_model_label]['ap']:.4f}")
print("=================================")
