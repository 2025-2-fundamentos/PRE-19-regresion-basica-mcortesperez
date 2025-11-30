# Copilot Instructions for PRE-19: Basic Regression

## Project Overview

This is a supervised learning homework assignment focused on **regression using scikit-learn**, specifically training a Multi-Layer Perceptron (MLP) neural network on the Auto MPG dataset. The task is to build a model that predicts MPG from car features with MSE < 7.745 on the test set.

## Architecture & Data Flow

1. **Input**: `files/input/auto_mpg.csv` (398 car records with features: Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin)
2. **Data Preprocessing**: 
   - Drop NaN values
   - Encode categorical "Origin" (1=USA, 2=Europe, 3=Japan) via one-hot encoding using `pd.get_dummies()`
   - Separate target (MPG) from features
3. **Model Training**: MLP neural network on standardized features (StandardScaler)
4. **Artifact Storage**: Persist model and scaler as pickle files (`mlp.pickle`, `features_scaler.pickle`) for test evaluation

## Critical Developer Workflows

### Setup & Environment
```bash
# macOS/Linux
python3 -m venv .venv && source .venv/bin/activate && source setup.sh

# Windows
python3 -m venv .venv && .venv\Scripts\activate && setup
```

### Running Tests
```bash
pytest  # Runs tests/test_homework.py
```

**Test Details** (`tests/test_homework.py`):
- Loads `auto_mpg.csv`, applies same preprocessing (dropna + Origin one-hot encoding)
- Loads `mlp.pickle` and `features_scaler.pickle` from **current working directory**
- Transforms dataset with the stored scaler
- Asserts MSE < 7.745 on full dataset

## Project Structure & Key Patterns

- **homework/**: Package directory (currently empty `__init__.py` - implementation goes here)
- **files/input/**: Static data assets (auto_mpg.csv)
- **tests/**: Pytest-based autograding; assumes pickle files exist in working directory

## Implementation Requirements

1. **Must save exactly two pickle files in working directory**:
   - `mlp.pickle`: Trained MLPRegressor instance
   - `features_scaler.pickle`: StandardScaler fitted on training features
   
2. **Preprocessing must match test expectations**:
   - Use `dataset.dropna()` (no imputation)
   - Map Origin: `{1: "USA", 2: "Europe", 3: "Japan"}`
   - Apply one-hot encoding with `pd.get_dummies(columns=["Origin"], prefix="", prefix_sep="")`
   - Pop "MPG" **after** categorical encoding

3. **Standardization required**: Features must be scaled via scaler before MLP training

4. **Model target**: MSE < 7.745 on the complete auto_mpg dataset (all 398 records after dropna)

## Dependencies
- `pandas`: Data loading and preprocessing
- `scikit-learn`: MLPRegressor, StandardScaler, mean_squared_error
- `pytest`: Test execution
- `matplotlib`: (optional, likely for visualization exercises)

## Common Pitfalls
- Forgetting to standardize features before MLP training (affects convergence)
- Pickle files not saved in working directory when tests run
- Using different preprocessing steps than the test expects (esp. Origin encoding order)
- Training/test set split not matching test assumptions (test uses full dataset)
