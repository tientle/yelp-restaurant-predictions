
from pathlib import Path

import joblib
import pandas as pd

from yelp_data import (
    filter_businesses,
    clean_business_attributes,
    remove_nested_attributes,
)


# --------------------------------------------------
# 1. Set file paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "yelp_academic_dataset_business.json"
MODEL_PATH = PROJECT_DIR / "models" / "restaurant_rating_model.joblib"

OUTPUT_DIR = PROJECT_DIR / "data" / "dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and filter Yelp restaurants
# --------------------------------------------------

print("Loading Yelp dataset...")

# Yelp's JSON file contains one business per line.
businesses = pd.read_json(DATA_PATH, lines=True)

# Reuse the filtering function from the original project.
restaurants = filter_businesses(
    businesses,
    category="Restaurants",
    min_reviews=500,
).copy()

print(f"Found {len(restaurants):,} restaurants.")


# --------------------------------------------------
# 3. Clean restaurant attributes
# --------------------------------------------------

# Expand Yelp's nested attributes dictionary into columns.
attributes = restaurants["attributes"].apply(pd.Series)

# Apply the same cleaning used in the original analysis.
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

simple_attributes = remove_nested_attributes(
    clean_attributes,
    nested_attributes,
)

# Keep restaurant details and original Yelp ratings.
restaurant_details = restaurants[
    ["name", "stars", "review_count", "city", "state"]
]

# Join by the original restaurant index.
dashboard_data = restaurant_details.join(simple_attributes)

# Export the restaurant-level dataset.
restaurant_output = OUTPUT_DIR / "restaurants.csv"

dashboard_data.to_csv(
    restaurant_output,
    index=False,
)

print(f"Saved restaurant data to {restaurant_output}")


# --------------------------------------------------
# 4. Export model feature importance
# --------------------------------------------------

print("Loading trained model...")

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]

# Access the fitted preprocessing pipeline.
preprocessor = model.named_steps["preprocessor"]

# Access the trained Random Forest.
# The pipeline's final step contains the regression model.
random_forest = model.steps[-1][1]

# Retrieve feature names after one-hot encoding.
encoded_feature_names = preprocessor.get_feature_names_out()

# Retrieve the importance score for each encoded feature.
importance_scores = random_forest.feature_importances_

feature_importance = pd.DataFrame({
    "feature": encoded_feature_names,
    "importance": importance_scores,
})

# --------------------------------------------------
# 5. Group importance by original restaurant attribute
# --------------------------------------------------

# The model's preprocessing pipeline expands categorical
# attributes into multiple encoded features.
#
# For example:
# categorical__OutdoorSeating_True
# categorical__OutdoorSeating_False
#
# We want to combine these into one OutdoorSeating score.

categorical_columns = model_bundle["categorical_columns"]
numeric_columns = model_bundle["numeric_columns"]

# Retrieve the number of encoded categories per attribute.
categorical_pipeline = preprocessor.named_transformers_["categorical"]
encoder = categorical_pipeline.named_steps["encoder"]

original_attributes = []

for column, categories in zip(
    categorical_columns,
    encoder.categories_,
):
    for category in categories:
        original_attributes.append(column)

# Numeric features are passed through separately.
for column in numeric_columns:
    original_attributes.append(column)

# Confirm that each encoded feature has a matching attribute.
if len(original_attributes) != len(feature_importance):
    raise ValueError(
        "Encoded feature count does not match the number "
        "of original attributes. Check the preprocessing pipeline."
    )

feature_importance["attribute"] = original_attributes

# Sum importance scores across encoded categories.
attribute_importance = (
    feature_importance.groupby("attribute", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)

# Save the grouped scores for our dashboard.
importance_output = OUTPUT_DIR / "feature_importance.csv"

attribute_importance.to_csv(
    importance_output,
    index=False,
)

print(f"Saved feature importance to {importance_output}")

print("\nTop 10 most important attributes:")
print(attribute_importance.head(10).to_string(index=False))

print("\nDashboard data preparation complete!")
