
import os
import pandas as pd

RAW_DATA_DIR = "../Data/"
PROCESSED_DATA_DIR = "../Data/processed/"

def load_data(data_dir, label):
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  
            df['label'] = label 
            data.append(df)
    return pd.concat(data, ignore_index=True)


def main():
    #there should be something that automates the creation of the file directories
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    dyslexic_data = load_data(os.path.join(RAW_DATA_DIR, "Dyslexic"), label=1)
    control_data = load_data(os.path.join(RAW_DATA_DIR, "Control"), label=0)
    combined_data = pd.concat([dyslexic_data, control_data], ignore_index=True)
    combined_data.to_csv(os.path.join(PROCESSED_DATA_DIR, "combined_raw_data.csv"), index=False)
    print("Combined raw data saved to combined_raw_data.csv")

if __name__ == "__main__":
    #The main function runs when life is f***ing you up
    main()
