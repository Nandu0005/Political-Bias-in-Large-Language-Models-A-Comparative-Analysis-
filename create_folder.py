"""
create_folders.py
Creates the complete folder structure for the project
Run this first before starting the project
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent

def create_folder_structure():
    """Create all required folders for the project"""
    
    folders = [
        # Results folders for each dataset
        "results/pew",
        "results/compass",
        "results/isidewith",
        "results/basepaper",
        
        # Questions folder
        "questions",
        
        # Logs and cache
        "logs",
        "cache/embeddings",
        "cache/responses",
        
        # Reports
        "reports/figures",
        "reports/tables"
    ]
    
    print("=" * 60)
    print("CREATING FOLDER STRUCTURE")
    print("=" * 60)
    
    for folder in folders:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print("\n" + "=" * 60)
    print("FOLDER STRUCTURE CREATED SUCCESSFULLY")
    print("=" * 60)
    
    # Print tree
    print("\nProject Structure:")
    print("├── main.py")
    print("├── config_loader.py")
    print("├── api_wrappers.py")
    print("├── collect_responses.py")
    print("├── preprocessing.py")
    print("├── bias_scoring.py")
    print("├── compare_models.py")
    print("├── visualize_results.py")
    print("├── requirements.txt")
    print("├── config.json")
    print("├── .env")
    print("├── questions/")
    print("│   ├── pew_questions.json")
    print("│   ├── compass_questions.json")
    print("│   ├── isidewith_questions.json")
    print("│   └── basepaper_questions.json")
    print("├── results/")
    print("│   ├── pew/")
    print("│   ├── compass/")
    print("│   ├── isidewith/")
    print("│   └── basepaper/")
    print("├── logs/")
    print("├── cache/")
    print("└── reports/")

if __name__ == "__main__":
    create_folder_structure()