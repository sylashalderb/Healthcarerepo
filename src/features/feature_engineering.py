import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    # Check for missing values and one-hot encode 'gender' and 'smoking_history'
    # Perform feature scaling and create interaction features here
    
    # Apply one-hot encoding to the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')
    
    # Filter out rows where 'smoking_history' is 'No Info'
    df = df[df['smoking_history'] != 'No Info']
    
    # Apply one-hot encoding to the 'smoking_history' column
    df = pd.get_dummies(df, columns=['smoking_history'], prefix='smoking_history')

    # Standard Scaling on numerical columns
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    standard_scaler = StandardScaler()
    df[numerical_cols] = standard_scaler.fit_transform(df[numerical_cols])

    # Create interaction features
    df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']
    df['diabetes_age'] = df['diabetes'] * df['age']

    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    processed_df = feature_engineering(df)
    processed_df.to_csv('data/processed/final_data.csv', index=False)
