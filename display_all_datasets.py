# display_all_datasets.py
import pandas as pd
import json
from pathlib import Path

# Set path
base_path = "E:/defg/political bias analysis/questions"

print("="*100)
print("LOADING ALL POLITICAL BIAS QUESTIONNAIRES")
print("="*100)

# =========================================================
# 1. PEW RESEARCH DATASET
# =========================================================
print("\n\n")
print("="*100)
print("DATASET 1: PEW RESEARCH POLITICAL TYPOLOGY QUIZ")
print("="*100)

with open(f"{base_path}/pew_questions.json", 'r', encoding='utf-8') as f:
    pew_data = json.load(f)

pew_df = pd.DataFrame(pew_data)
print(f"\nTotal Questions: {len(pew_df)}")
print("\nFirst 10 Questions:")
print(pew_df[['id', 'category', 'question']].head(10).to_string(index=False))

print("\nLast 5 Questions:")
print(pew_df[['id', 'category', 'question']].tail(5).to_string(index=False))

print("\nCategory Distribution:")
print(pew_df['category'].value_counts().to_string())

# =========================================================
# 2. POLITICAL COMPASS DATASET
# =========================================================
print("\n\n")
print("="*100)
print("DATASET 2: POLITICAL COMPASS ASSESSMENT")
print("="*100)

with open(f"{base_path}/compass_questions.json", 'r', encoding='utf-8') as f:
    compass_data = json.load(f)

compass_df = pd.DataFrame(compass_data)
print(f"\nTotal Questions: {len(compass_df)}")
print("\nFirst 10 Questions:")
print(compass_df[['id', 'category', 'question']].head(10).to_string(index=False))

print("\nLast 5 Questions:")
print(compass_df[['id', 'category', 'question']].tail(5).to_string(index=False))

print("\nCategory Distribution:")
print(compass_df['category'].value_counts().to_string())

# =========================================================
# 3. ISIDEWITH DATASET
# =========================================================
print("\n\n")
print("="*100)
print("DATASET 3: ISIDEWITH POLITICAL PARTY QUIZ")
print("="*100)

with open(f"{base_path}/isidewith_questions.json", 'r', encoding='utf-8') as f:
    isidewith_data = json.load(f)

isidewith_df = pd.DataFrame(isidewith_data)
print(f"\nTotal Questions: {len(isidewith_df)}")
print("\nFirst 10 Questions:")
print(isidewith_df[['id', 'category', 'question']].head(10).to_string(index=False))

print("\nLast 5 Questions:")
print(isidewith_df[['id', 'category', 'question']].tail(5).to_string(index=False))

print("\nCategory Distribution:")
print(isidewith_df['category'].value_counts().to_string())

# =========================================================
# 4. BASE PAPER DATASET
# =========================================================
print("\n\n")
print("="*100)
print("DATASET 4: BASE PAPER QUESTIONS")
print("="*100)

with open(f"{base_path}/basepaper_questions.json", 'r', encoding='utf-8') as f:
    basepaper_data = json.load(f)

basepaper_df = pd.DataFrame(basepaper_data)
print(f"\nTotal Questions: {len(basepaper_df)}")
print("\nAll Questions:")
print(basepaper_df[['id', 'category', 'question']].to_string(index=False))

# =========================================================
# SUMMARY TABLE
# =========================================================
print("\n\n")
print("="*100)
print("DATASET SUMMARY TABLE")
print("="*100)

summary_data = {
    'Dataset': ['Pew Research', 'Political Compass', 'ISideWith', 'Base Paper'],
    'Source': ['pew_questions.json', 'compass_questions.json', 'isidewith_questions.json', 'basepaper_questions.json'],
    'Questions': [len(pew_df), len(compass_df), len(isidewith_df), len(basepaper_df)],
    'Categories': [pew_df['category'].nunique(), compass_df['category'].nunique(), 
                   isidewith_df['category'].nunique(), basepaper_df['category'].nunique()]
}

summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))
print("="*100)
print(f"\nTOTAL QUESTIONS ACROSS ALL DATASETS: {len(pew_df) + len(compass_df) + len(isidewith_df) + len(basepaper_df)}")
print("="*100)