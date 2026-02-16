# EnergyPredict

A machine learning project that uses the Random Forest algorithm to predict heating loads in buildings, supporting energy-efficient design decisions.

## Project Overview

EnergyPredict is designed to predict the heating loads of buildings based on their design characteristics. By analyzing building features such as compactness, surface area, orientation, and glazing properties, the model can help architects and engineers make informed decisions about energy efficiency during the design phase.

## Features

- **Data Loading & Preprocessing**: Loads energy efficiency data from Excel files
- **Exploratory Data Analysis**: Generates correlation heatmaps and scatter plots to understand data relationships
- **Machine Learning Model**: Implements a Random Forest Regressor with 200 trees for robust predictions
- **Model Evaluation**: Provides comprehensive performance metrics (R², MSE, RMSE, MAE)
- **Model Persistence**: Saves trained models for future use

## Dataset

The project uses an "Energy_Efficiency.xlsx" dataset containing building design characteristics:

- **Relative Compactness**: Ratio of surface area to volume
- **Surface Area**: Total surface area of the building
- **Wall Area**: Total wall area
- **Roof Area**: Total roof area
- **Overall Height**: Height of the building
- **Orientation**: Building orientation (degrees)
- **Glazing Area**: Total area of glazed surfaces
- **Glazing Distribution**: How glazing is distributed
- **Heating Load**: Target variable (what the model predicts)

## Requirements

The project requires Python 3.x and the following packages:

```
pandas
seaborn
matplotlib
scikit-learn
joblib
openpyxl
```

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the Random Forest model:

```bash
python train_model.py
```

This script will:
1. Load the energy efficiency dataset from `Energy_Efficiency.xlsx`
2. Display data statistics and missing value analysis
3. Generate exploratory visualizations:
   - `plot_correlation_heatmap.png`: Correlation matrix heatmap
   - `plot_scatter_relative_compactness_vs_heating.png`: Scatter plot of relative compactness vs heating load
4. Train a Random Forest model with 200 trees on 80% of the data
5. Evaluate performance on the test set (20% of data)
6. Save the trained model as `energy_heating_model.pkl`

### Loading and Using the Trained Model

Use the `load_model.py` script to load and use the pre-trained model for predictions on new building designs.

## Model Details

### Random Forest Configuration

- **Number of Trees (n_estimators)**: 200
- **Max Depth**: None (trees grow fully)
- **Random State**: 27 (ensures reproducibility)
- **Parallel Processing**: Uses all available CPU cores

### Train/Test Split

- Training Set: 80% of data
- Test Set: 20% of data
- Random State: 27 (for reproducible splits)

## Evaluation Metrics

The model performance is evaluated using:

- **R² Score**: Coefficient of determination (how well the model fits the data)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

## Output Files

After running `train_model.py`, the following files are generated:

- `energy_heating_model.pkl`: Trained Random Forest model (serialized with joblib)
- `plot_correlation_heatmap.png`: Correlation matrix visualization
- `plot_scatter_relative_compactness_vs_heating.png`: Feature-target relationship visualization

## Project Structure

```
EnergyPredict/
├── train_model.py              # Main training script
├── load_model.py               # Model loading and prediction script
├── requirements.txt            # Project dependencies
├── README.md                   # This file
├── LICENSE                     # License information
└── Energy_Efficiency.xlsx      # Input dataset (not included)
```

## Notes

- The dataset file `Energy_Efficiency.xlsx` must be in the same directory as the scripts
- Ensure all dependencies are installed before running the scripts
- The model uses random_state=27 for reproducibility; results should be consistent when using the same data
