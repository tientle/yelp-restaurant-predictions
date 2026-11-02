# Yelp Restaurant Predictions

**Predicting restaurant ratings using machine learning and Yelp business attributes**

[![Python tests](https://github.com/tientle/yelp-restaurant-predictions/actions/workflows/test.yml/badge.svg)](https://github.com/tientle/yelp-restaurant-predictions/actions/workflows/test.yml)

## Project Overview

What characteristics are associated with higher restaurant ratings, and can we use these characteristics to predict how customers evaluate a restaurant?

Using the Yelp Open Dataset, I built a Random Forest regression model to predict restaurant star ratings based on business attributes such as outdoor seating, delivery options, alcohol offerings, and other amenities.

Beyond prediction, I explored which attributes contributed most to model performance and how these findings might help restaurants better understand customer preferences and business performance.

The project also includes automated testing, continuous integration, and Docker containerization to support reproducibility.

## Key Results

- **1,263 restaurants** analyzed, each with at least 500 Yelp reviews.
- **0.322 stars Mean Absolute Error (MAE)** using Random Forest regression.
- **200 decision trees** trained using an 80/20 train-test split.
- **10 restaurant attributes** identified as the most important predictors based on grouped feature importance.

## Business Question

Restaurant ratings influence how customers discover and evaluate businesses. Understanding the characteristics associated with ratings can provide useful insights into customer preferences and restaurant positioning.

This project investigates:

1. Can restaurant star ratings be predicted using business attributes?
2. Which restaurant characteristics contribute most to predictive performance?
3. What patterns might help restaurant owners better understand how their businesses are perceived?

**Important distinction:** Predictive relationships do not necessarily imply causation. An attribute being useful for predicting ratings does not mean adding that attribute will increase a restaurant's rating.

## Dataset

**Source:** [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

The Yelp business dataset contains business information, including categories, ratings, review counts, and available amenities.

I filtered the dataset to include:

- Businesses categorized as restaurants.
- Restaurants with at least 500 reviews.

The resulting dataset contained **1,263 restaurants**.

The model uses restaurant-level business attributes rather than individual review text.

## Methodology

### 1. Data Cleaning and Preparation

Restaurant attributes were stored in dictionary-like structures that required preprocessing before modeling.

I:

- Expanded restaurant attributes into individual columns.
- Standardized inconsistent string representations.
- Examined attribute coverage and identified attributes with at least 75% coverage.
- Excluded nested dictionary attributes from the modeling analysis.
- Converted restaurant price range into a numeric variable.
- Preserved missing values where appropriate rather than assuming that missing information indicated the absence of an amenity.

### 2. Predictive Modeling

I trained a **Random Forest regression model** to predict Yelp star ratings.

| Component         | Approach                   |
| ----------------- | -------------------------- |
| Target variable   | Yelp star rating (`stars`) |
| Predictors        | Restaurant attributes      |
| Model             | Random Forest Regressor    |
| Trees             | 200                        |
| Training split    | 80%                        |
| Testing split     | 20%                        |
| Evaluation metric | Mean Absolute Error (MAE)  |

Categorical attributes were transformed using one-hot encoding, while restaurant price range was treated as a numeric feature.

### 3. Model Evaluation

The model achieved a **Mean Absolute Error of 0.322 stars**.

This means its predictions differed from actual Yelp ratings by approximately 0.32 stars on average on the test set.

### 4. Feature Importance

To understand which restaurant characteristics were most useful for prediction, I examined Random Forest feature importance.

Because one-hot encoding creates multiple features from a single original attribute, I aggregated the encoded feature importance scores back to their original restaurant attributes.

![Feature Importance](images/encoding.png)

![Grouped Feature Importance](images/encoding_grouped_sum.png)

The ten highest-ranked attributes were:

1. Alcohol
2. Outdoor Seating
3. Happy Hour
4. Restaurant Delivery
5. TV Availability
6. Bike Parking
7. Dogs Allowed
8. Smoking
9. Wheelchair Accessibility
10. Restaurant Table Service

![Grouped Attribute Importance](images/grouped_encoding.png)

## Findings and Interpretation

Although several restaurant attributes were important for prediction, differences in average ratings across attribute categories were generally small.

![Average Yelp Rating by Alcohol Offering](images/alcohol_ratings.png)

![Average Yelp Rating by Outdoor Seating Offering](images/outdoor_ratings.png)

Feature importance indicates how useful an attribute is to the model, but it does not establish whether the attribute increases or decreases ratings.

These results highlight the distinction between identifying predictive patterns and making causal business recommendations.

### Business Applications

The analysis demonstrates how machine learning can help identify patterns in customer-facing business characteristics.

Potential applications include:

- Understanding which restaurant attributes are most informative when analyzing ratings.
- Exploring differences in customer perceptions across restaurant characteristics.
- Supporting further investigation into restaurant positioning and customer experience.

These are potential applications rather than demonstrated improvements in restaurant performance.

## Reproducing the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/tientle/yelp-restaurant-predictions.git
cd yelp-restaurant-predictions
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and place `yelp_academic_dataset_business.json` in the `data/` directory.

```text
data/
└── yelp_academic_dataset_business.json
```

The dataset is not included in this repository due to its size.

### 5. Run the analysis

```bash
python yelp_data.py
```

The script performs data filtering, cleaning, feature preprocessing, model training, evaluation, feature importance analysis, and visualization generation.

## Testing and Reproducibility

To improve reliability, I refactored the original analysis into reusable functions and developed automated tests using `pytest`.

The seven tests cover restaurant filtering, attribute cleaning, preprocessing, feature importance calculations, and the end-to-end modeling workflow.

Run the tests using:

```bash
python -m pytest testing/test_yelp_data.py -v
```

![Passing Tests](images/passed-tests.png)

### Continuous Integration

GitHub Actions automatically validates the project on pushes, pull requests, and scheduled runs.

The workflow includes:

- Python 3.12 and 3.13 compatibility testing.
- Black formatting checks.
- flake8 code-quality checks.
- Seven automated pytest tests.
- Docker image builds and containerized testing.

![GitHub Actions Workflow](images/testing.png)

### Docker

Build and run the containerized test environment:

```bash
docker build -t yelp-analysis .
docker run --rm yelp-analysis
```

![Docker Tests](images/docker-tests.png)

## Additional Technical Explorations

### Pandas vs. Polars

I compared the performance of Pandas and Polars on filtering and aggregation operations.

| Operation                       |     Pandas |     Polars |
| ------------------------------- | ---------: | ---------: |
| Filtering                       | 0.020480 s | 0.019030 s |
| Grouping and summary statistics | 0.016093 s | 0.016181 s |

The performance differences were small for the tested operations. Results may vary with dataset size, hardware, and execution environment.

### Rust Ownership

I also explored Rust's ownership system through a Jupyter notebook demonstrating how assigning a vector to another variable transfers ownership.

![Rust Ownership Error](images/rust_error.png)

![Rust Ownership Success](images/rust.png)

## Tools and Technologies

**Data Science:** Python, pandas, Polars, scikit-learn, NumPy

**Machine Learning:** Random Forest Regression, One-Hot Encoding, Feature Importance, Model Evaluation

**Engineering:** pytest, GitHub Actions, Docker, Black, flake8

**Additional Exploration:** Rust
