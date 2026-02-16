import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
import joblib


def main():
    # 1) Load Data
    file_path = "Energy_Efficiency.xlsx"   # Path to the dataset
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    print("First 5 rows of the data:")
    print(df.head(), "\n")

    print("Statistical summary of the data:")
    print(df.describe(), "\n")

    # Check for missing values
    print("Missing values per column:")
    print(df.isna().sum(), "\n")

    # 2) Exploratory Data Analysis (EDA)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap - Energy Efficiency Data")
    plt.tight_layout()
    plt.savefig("plot_correlation_heatmap.png", dpi=300)
    plt.close()

    # Scatter Plot: Heating Load & Relative Compactness
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Relative_Compactness",
        y="Heating_Load"
    )
    plt.title("Heating Load vs Relative Compactness")
    plt.xlabel("Relative Compactness")
    plt.ylabel("Heating Load")
    plt.tight_layout()
    plt.savefig("plot_scatter_relative_compactness_vs_heating.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
