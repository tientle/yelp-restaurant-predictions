
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from yelp_data import (
    filter_businesses,
    clean_business_attributes,
    remove_nested_attributes,
)


# --------------------------------------------------
# 1. Set up file paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "yelp_academic_dataset_business.json"
MODEL_PATH = PROJECT_DIR / "models" / "restaurant_rating_model.joblib"


# --------------------------------------------------
# 2. Load and clean the Yelp data
# --------------------------------------------------

def load_restaurant_data():
    print("Loading Yelp dataset...")

    data = pd.read_json(DATA_PATH, lines=True)

    # Keep restaurants with at least 500 reviews.
    restaurants = filter_businesses(data, "Restaurants").copy()

    # Expand the dictionary of restaurant attributes.
    attributes = restaurants["attributes"].apply(pd.Series)

    # Reuse cleaning functions from the original project.
    clean_attributes = clean_business_attributes(attributes)

    # Remove attributes containing nested dictionaries.
    nested_attributes = [
        "BusinessParking",
        "Ambience",
        "GoodForMeal",
        "Music",
        "BestNights",
        "DietaryRestrictions",
    ]

    features = remove_nested_attributes(
        clean_attributes,
        nested_attributes,
    )

    # Ensure the price range column exists.
    if "RestaurantsPriceRange2" not in features.columns:
        features["RestaurantsPriceRange2"] = np.nan

    # Convert price range to a numeric value.
    features["RestaurantsPriceRange2"] = pd.to_numeric(
        features["RestaurantsPriceRange2"],
        errors="coerce",
    )

    # Separate restaurant ratings from their attributes.
    ratings = restaurants["stars"]

    print(f"Loaded {len(features):,} restaurants.")

    return features, ratings


# --------------------------------------------------
# 3. Train and evaluate the model
# --------------------------------------------------

def train_model():
    features, ratings = load_restaurant_data()

    # Identify categorical and numeric attributes.
    categorical_columns = [
        column
        for column in features.columns
        if column != "RestaurantsPriceRange2"
    ]

    numeric_columns = ["RestaurantsPriceRange2"]

    # Convert categorical attributes to consistent strings.
    # Yelp attributes may contain a mixture of strings,
    # booleans, and missing values.
    #
    # Example:
    # True  -> "True"
    # False -> "False"
    # NaN   -> "Missing"
    #
    # This prevents OneHotEncoder from encountering
    # incompatible data types.

    for column in categorical_columns:
        features[column] = (
            features[column]
            .astype("string")
            .fillna("Missing")
            .astype(str)
        )

    # Split the data into training and testing sets.
    # Preprocessing is fitted only on the training data.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        ratings,
        test_size=0.2,
        random_state=42,
    )

    # Encode categorical attributes into numeric features.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Replace missing price values with the training median.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    # Apply different preprocessing to categorical
    # and numeric restaurant attributes.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
        ]
    )

    # Combine preprocessing and the Random Forest model.
    # This allows the Streamlit app to make predictions
    # directly from restaurant attributes.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Random Forest model...")

    # Train the model using the training set.
    model.fit(x_train, y_train)

    # Predict Yelp ratings for the test set.
    predictions = model.predict(x_test)

    # Calculate average absolute prediction error.
    mae = mean_absolute_error(y_test, predictions)

    print(f"Mean Absolute Error: {mae:.3f} stars")

    # --------------------------------------------------
    # 4. Create a baseline restaurant profile
    # --------------------------------------------------

    # The interactive app will let users change selected
    # restaurant attributes, such as outdoor seating,
    # delivery, alcohol offerings, and price range.
    #
    # For attributes users do not select, we will use
    # representative values from the training dataset.

    baseline = {}

    for column in features.columns:
        available_values = x_train[column].dropna()

        if available_values.empty:
            baseline[column] = (
                2 if column == "RestaurantsPriceRange2"
                else "Missing"
            )
        else:
            baseline[column] = available_values.mode().iloc[0]

    # --------------------------------------------------
    # 5. Save the model for the Streamlit app
    # --------------------------------------------------

    # Create the models folder if it does not exist.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model and information needed
    # to generate predictions from user inputs.
    model_bundle = {
        "model": model,
        "feature_columns": features.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "baseline": baseline,
        "mae": mae,
        "restaurant_count": len(features),
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print("Training complete!")


# --------------------------------------------------
# 6. Run the training script
# --------------------------------------------------

if __name__ == "__main__":
    train_model()
