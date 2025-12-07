import os
import pandas as pd
import ast

df = pd.read_csv("C:\\Users\\User\\Downloads\\IIT\\Year 2\\DSGP\\final_DS.csv")

# Quick sanity
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
# lower-case & simple cleaning (you can add lemmatization later)
df['combined_text'] = df['combined_text'].str.replace(r'\s+',' ', regex=True).str.strip().str.lower()
