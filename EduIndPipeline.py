import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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


if __name__ == '__main__':
    main ()