import os
import pandas as pd
import requests


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Download dataset if not present and load into DataFrame
# ------------------------------------------------------------------------------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "wine_quality.csv")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    with open(CSV_PATH, "x") as f:
        print("Downloading wine_quality.csv...")
        response = requests.get(url)
        response.raise_for_status()
        f.write(response.text)
        print("Download complete.")
except FileExistsError:
    print("wine_quality.csv already exists. Skipping download.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")
    url_combined = "https://raw.githubusercontent.com/anirudha027/Wine-Quality-Prediction/main/winequality.csv"
    try:
        with open(CSV_PATH, "w") as f:
            print("Attempting to download combined winequality.csv...")
            response = requests.get(url_combined)
            response.raise_for_status()
            f.write(response.text)
            print("Combined winequality.csv download complete.")
    except requests.exceptions.RequestException as e_combined:
        print(f"Error downloading combined file: {e_combined}")
        print("Please ensure dataset URL is accessible.")
df = pd.read_csv(CSV_PATH, sep=";")


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Log dataset load success message
# ------------------------------------------------------------------------------------------------------------------------------------------------

print("\nDataset loaded successfully")
print(df.head())
