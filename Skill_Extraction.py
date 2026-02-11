import pandas as pd

def skill_extraction(df):
    # Load and Process CSV
    # flatten the CSV into a single list of unique, lowercase skills
    skill_df = pd.read_csv('skill_Data.csv')
    master_skill_set = set()

    for row in skill_df['Skills']:
        if isinstance(row, str):
            # Split each row by comma, strip whitespace, and convert to lowercase
            skills = [s.strip().lower() for s in row.split(',')]
            master_skill_set.update(skills)

    print(f"Loaded {len(master_skill_set)} unique skills from the CSV dictionary.")
    print(master_skill_set)
    return master_skill_set