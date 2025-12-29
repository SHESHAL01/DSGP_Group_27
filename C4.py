import os
import random

import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, InputExample

df = pd.read_csv("C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv")

print("rows:", len(df))
print("columns:", df.columns.tolist())
print(df.head())

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

models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-mpnet-base-v2'
]

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

print("Training pairs:", len(train_examples))
