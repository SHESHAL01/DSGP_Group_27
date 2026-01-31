import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('DATASETS/combined_skills_dataset.csv')

df['all_skills'] = df['all_skills'].apply(ast.literal_eval)

X = df['all_skills']
y = df['title']

mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)
feature_names = mlb.classes_

X_encoded_df = pd.DataFrame(
    X_encoded,
    columns=feature_names
)

X_encoded_df['title'] = y.values
X_encoded_df.to_csv('encoded_skills_dataset.csv', index=False)
print("Saved encoded_skills_dataset.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('encoded_skills_dataset.csv')
print(df.head())
