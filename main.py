import sys
import json
import joblib
import pandas as pd
import os

from src.assessment_engine import full_wine_assessment

def print_usage():
    print("""
Vineyard Quality Assessment System
---------------------------------

This project provides a web interface for wine quality prediction.

To use the web interface, run:
    streamlit run app.py

This will start a local web server where you can input wine parameters
and get quality predictions interactively.

For development or direct API usage, you can use the assessment functions
in src/assessment_engine.py directly.
    """)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_usage()
        return

    print("=" * 60)
    print("Vineyard Quality Assessment System")
    print("=" * 60)
    print("\nThis project is designed to be used through the web interface.")
    print("\nTo start the web interface, run:")
    print("    streamlit run app.py")
    print("\nFor more options, run:")
    print("    python main.py --help")

if __name__ == "__main__":
    main()
