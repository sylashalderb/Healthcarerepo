import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    sns.set(style='whitegrid')

    # Count Plot
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='diabetes')
    plt.title('Distribution of Diabetes Cases')
    plt.xlabel('Diabetes (0: No, 1: Yes)')
    plt.ylabel('Count')
    plt.show()

    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

    # Distribution of Numerical Features
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    plt.figure(figsize=(16, 12))

    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, hue='diabetes', multiple='stack', kde=True)
        plt.title(f'Distribution of {col} by Diabetes Status')
        plt.xlabel(col)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data/processed/final_data.csv')
    visualize_data(df)
