import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import seaborn as sns
import matplotlib.pyplot as plt

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


def clean_data(df):
    # Removing duplicates and Nan values
    df = df.dropna()
    df = df.drop_duplicates(keep='first')

    # Fixing inconsistent data
    df['Degree Program'] = df['Degree Program'].str.strip().str.lower().str.capitalize()
    df['Course Name'] = df['Course Name'].str.strip().str.lower().str.capitalize()


def word_count_analysis(df):
    print("COURSE NAME ANALYSIS")
    course_name_lengths = df['Course Name'].str.len()
    print(f"Average Course Name Length: {course_name_lengths.mean():.2f} characters")
    print(f"Min Length: {course_name_lengths.min()}")
    print(f"Max Length: {course_name_lengths.max()}")

    word_counts = df['Course Name'].str.split().str.len()
    print(f"\nAverage Words in Course Name: {word_counts.mean():.2f}")


def feature_extraction(df):
    df_features = df.copy()

    print("\nENCODING CATEGORICAL VARIABLES")

    le_university = LabelEncoder()
    le_degree = LabelEncoder()

    df_features['University_Encoded'] = le_university.fit_transform(df_features['University'])
    df_features['Degree_Encoded'] = le_degree.fit_transform(df_features['Degree Program'])

    return df_features


def mock_market_demand():
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


def Similarity_Measures(market_demand_skills):
    # Initialize Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Market Demand Data
    market_text = ", ".join(market_demand_skills)
    print(market_text)
    market_embeddings = model.encode(market_text, convert_to_tensor=True)

    # Similarity Model Test_Data
    df_features = pd.read_csv('Data.csv')

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
    data = pd.read_csv('data.csv')
    df = pd.DataFrame(data)
    df.head()
    print("------------------")
    print("Before Cleaning")
    print("------------------")
    DataAnalysis(df)
    print("------------------")
    print("\nAfter Cleaning")
    print("------------------")
    clean_data(df)
    DataAnalysis(df)
    word_count_analysis(df)
    df = feature_extraction(df)
    mock_data = mock_market_demand()
    uni_score = Similarity_Measures(mock_data)
    plot_charts(uni_score, df, mock_data)


if __name__ == '__main__':
    main()