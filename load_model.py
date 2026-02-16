import pandas as pd
import joblib


def main():
    # 1) Load the Saved Model
    model_filename = "energy_heating_model.pkl"
    try:
        model = joblib.load(model_filename)
        print(f"Successfully loaded model from: {model_filename}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found. Please run train_model.py first.")
        return

    # 2) Prediction Example (New Building)
    example_building = pd.DataFrame(
        {
            "Relative_Compactness": [0.90],
            "Surface_Area": [563.5],
            "Wall_Area": [318.5],
            "Roof_Area": [122.5],
            "Overall_Height": [7.0],
            "Orientation": [2],
            "Glazing_Area": [0.10],
            "Glazing_Distribution": [3],
        }
    )

    # Make prediction
    try:
        example_pred = model.predict(example_building)[0]
        print("\nNew Building Example:")
        print(example_building)
        print(f"\nPredicted Heating Load = {example_pred:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
