import random
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import os
import numpy as np

df = pd.read_csv("C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv")
MODEL_PATH = "models/minilm_l6_fine_tuned"

# print("rows:", len(df))
# print("columns:", df.columns.tolist())

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

# for model_name in models_to_test:
#     print("\nEvaluating Model:", model_name)
#     model = SentenceTransformer(model_name)
#     recall = evaluate_model(model, texts, skills, k=5)
#     print("Recall Score:", recall)

final_model = 'sentence-transformers/all-MiniLM-L6-v2'

def fine_tune_model():
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
                        texts=[
                            df.loc[i, "combined_text"],
                            df.loc[j, "combined_text"]
                        ],
                        label=1.0
                    )
                )

            # Negative pair (random sampling to avoid imbalance)
            elif random.random() < 0.05:
                train_examples.append(
                    InputExample(
                        texts=[
                            df.loc[i, "combined_text"],
                            df.loc[j, "combined_text"]
                        ],
                        label=0.0
                    )
                )

    # -------- LIMIT TRAINING PAIRS (IMPORTANT) --------
    MAX_PAIRS = 2000

    if len(train_examples) > MAX_PAIRS:
        train_examples = random.sample(train_examples, MAX_PAIRS)




    print("\n======================================")
    print("Fine-tuning:", final_model)
    print("======================================")

    # Load base model
    finetuned_model = SentenceTransformer(final_model)

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )

    # Loss
    train_loss = losses.CosineSimilarityLoss(finetuned_model)

    # Fine-tune
    finetuned_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        show_progress_bar=True
    )
    finetuned_model.save("models/minilm_l6_fine_tuned")
    return finetuned_model

# Evaluate after fine-tuning
print("\nEvaluating AFTER fine-tuning:", final_model)
# -------- LOAD OR TRAIN FINE-TUNED MODEL --------
if os.path.exists(MODEL_PATH):
    print("Loading fine-tuned model from disk...")
    fine_tuned_model = SentenceTransformer(MODEL_PATH)
else:
    print("Fine-tuned model not found. Training now...")
    fine_tuned_model = fine_tune_model()

# -------- EVALUATE FINE-TUNED MODEL --------
print("\nEvaluating AFTER fine-tuning:", final_model)
recall_after = evaluate_model(fine_tuned_model, texts, skills, k=5)
print("Recall Score after fine-tuning:", recall_after)

chosen_model = SentenceTransformer("models/minilm_l6_fine_tuned")

def build_embedding_index(chosen_model, texts):
    embeddings = chosen_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    np.save("course_embeddings.npy", embeddings)
    return embeddings

if os.path.exists("job_embeddings.npy"):
    job_embeddings = np.load("job_embeddings.npy")
else:
    job_embeddings = build_embedding_index(fine_tuned_model, texts)
