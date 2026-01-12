import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
df.head()

def DataAnalysis():
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

print("Before Cleaning")
print("------------------")
DataAnalysis()
