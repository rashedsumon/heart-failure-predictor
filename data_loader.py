import os
import shutil
import kagglehub
import pandas as pd

def load_heart_data():
    """
    Downloads the heart failure dataset from Kaggle via kagglehub 
    and returns it as a Pandas DataFrame.
    """
    # 1. Download latest version of the dataset
    raw_path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")
    
    # 2. Locate the target CSV file
    csv_filename = "heart_failure_clinical_records_dataset.csv"
    source_file = os.path.join(raw_path, csv_filename)
    
    # 3. Optional: Bring a copy to the local working directory for easier access
    local_file = os.path.join(os.getcwd(), csv_filename)
    if os.path.exists(source_file) and not os.path.exists(local_file):
        shutil.copy(source_file, local_file)
        
    # 4. Load and return the DataFrame
    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
    else:
        df = pd.read_csv(source_file)
        
    return df

if __name__ == "__main__":
    # Test script execution
    print("Testing dataset download...")
    data = load_heart_data()
    print("Dataset loaded successfully! Shape:", data.shape)
    print("Columns available:", data.columns.tolist())