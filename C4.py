import random

import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

df = pd.read_csv("C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv")

print("rows:", len(df))
print("columns:", df.columns.tolist())

#Preprocess text
def parse_skills(s):
    # safe parse if it's like "['python','r']"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip().lower() for x in parsed if str(x).strip()!='']
    except Exception:
        # fallback: split on comma
        if pd.isna(s): return []
        return [tok.strip().lower() for tok in str(s).split(',') if tok.strip()!='']
    return []

df['skills_list'] = df['Skills'].fillna('[]').apply(parse_skills)
# combined textual field (title + skills)
df['combined_text'] = df.apply(lambda r: (str(r.get('Title','')) + ' | ' + ' '.join(r['skills_list'])), axis=1)

models_to_test = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-mpnet-base-v2'
]

texts = df["combined_text"].tolist()
skills = df["skills_list"].tolist()

def evaluate_model(model, texts, skills, k=5):
    print("\nEvaluating model")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    nn = NearestNeighbors(metric="cosine", n_neighbors=k + 1)
    nn.fit(embeddings)

    hits = 0
    for i in range(len(texts)):
        _, indices = nn.kneighbors([embeddings[i]])
        retrieved = indices[0][1:]
        query_skills = set(skills[i])

        for idx in retrieved:
            if query_skills & set(skills[idx]):
                hits += 1
                break

    return hits / len(texts)

results = {}

results = {}

for model_name in models_to_test:
    print("\nEvaluating BEFORE fine-tuning:", model_name)
    model = SentenceTransformer(model_name)   # 👈 ADD THIS LINE HERE
    recall = evaluate_model(model, texts, skills, k=5)  # 👈 CHANGE THIS LINE
    results[model_name] = recall


# -------- SHOW RESULTS --------
results_df = pd.DataFrame.from_dict(
    results, orient="index", columns=["Recall@5"]
)

train_examples = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        skills_i = set(df.loc[i, "skills_list"])
        skills_j = set(df.loc[j, "skills_list"])

        shared = skills_i & skills_j

        # Positive pair
        if len(shared) > 0:
            train_examples.append(
                InputExample(
                    texts=[df.loc[i, "combined_text"], df.loc[j, "combined_text"]],
                    label=1.0
                )
            )

        # Negative pair (randomly sample to avoid imbalance)
        elif random.random() < 0.05:
            train_examples.append(
                InputExample(
                    texts=[df.loc[i, "combined_text"], df.loc[j, "combined_text"]],
                    label=0.0
                )
            )



# FINE-TUNE EACH MODEL & EVALUATE AFTER TRAINING
final_results = []

for model_name in models_to_test:
    print("\n======================================")
    print("Fine-tuning:", model_name)
    print("======================================")

    # Load base model
    model = SentenceTransformer(model_name)

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )

    # Loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        show_progress_bar=True
    )

    # Evaluate after fine-tuning
    recall_after = evaluate_model(model, texts, skills, k=5)

    final_results.append({
        "Model": model_name,
        "Recall@5 (Before)": results[model_name],
        "Recall@5 (After)": recall_after,
        "Improvement": recall_after - results[model_name]
    })

# -------- SHOW FINAL COMPARISON --------
final_df = pd.DataFrame(final_results)

print("\nMODEL COMPARISON RESULTS (Before vs After Fine-Tuning)")
print(final_df.sort_values("Recall@5 (After)", ascending=False))