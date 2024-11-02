import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_data('data/raw/diabetes_prediction_dataset.csv')
    df.to_csv('data/processed/preprocessed_data.csv', index=False)