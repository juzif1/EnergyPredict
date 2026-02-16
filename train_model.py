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

    # 3) Prepare Input Features and Target
    # All building design features used to predict Heating Load
    feature_cols = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Distribution",
    ]

    X = df[feature_cols]  # Input features
    y = df["Heating_Load"] # Target variable (what we want to predict)

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=27
    )

    # 4) Build Random Forest Model 
    model = RandomForestRegressor(
        n_estimators=200,   # Number of trees
        max_depth=None,     # No maximum depth (trees grow fully)
        random_state=27,    # For reproducibility
        n_jobs=-1,          # Use all CPU cores
    )

    # Train the model
    model.fit(X_train, y_train)

    # 5) Evaluate Model Performance 
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    print("Model Performance on Test Set:")
    print(f"R2  : {r2:.3f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")

    # 6) Save the Trained Model
    model_filename = "energy_heating_model.pkl"
    joblib.dump(model, model_filename)
    print(f"\nModel saved as: {model_filename}")

if __name__ == "__main__":
    main()