from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. Page configuration and file paths
# --------------------------------------------------

st.set_page_config(
    page_title="Build Your Restaurant",
    page_icon="🍽️",
    layout="centered",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "restaurant_rating_model.joblib"
DASHBOARD_DIR = PROJECT_DIR / "data" / "dashboard"

ORANGE = "#E94725"
CREAM = "#F2EBD8"
INK = "#28241F"
ORANGE_PALETTE = ["#F6B89B", "#F18B66", "#E94725", "#B93119"]

# CSS affects presentation only; the model and input logic are unchanged.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #F2EBD8 !important;
    color: #28241F !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: #F2EBD8 !important; }
[data-testid="stSidebar"] { background: #E9DFC8 !important; }
.block-container { max-width: 980px; padding-top: 3.1rem; padding-bottom: 5rem; }
h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: #28241F; }
h1 { font-family: 'DM Serif Display', Georgia, serif !important;
     color: #E94725 !important; font-size: clamp(2.7rem, 6vw, 4.6rem) !important;
     letter-spacing: -0.035em; line-height: 1.06; margin-bottom: .6rem; }
h2, h3 { font-family: 'DM Serif Display', Georgia, serif !important;
         letter-spacing: -.02em; color: #28241F !important; }
[data-testid="stCaptionContainer"] { color: #675E52 !important; }
hr { border-color: #D7C7AC !important; margin: 1.6rem 0 !important; }
/* Tab navigation */
# .stTabs [data-baseweb="tab-list"] { gap: .5rem; border-bottom: 1px solid #D6BBA0;
#     padding-bottom: .5rem; }
# .stTabs [data-baseweb="tab"] { background: transparent; color: #28241F;
#     border: 1px solid #D5B99C; border-radius: 999px; padding: .55rem 1.2rem; }
# .stTabs [aria-selected="true"] { background: #E94725 !important;
#     color: #FFFFFF !important; border-color: #E94725 !important; }
# .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }



/* Remove Streamlit's default tab underline */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: none !important;
    align-items: center;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Style all tabs */
.stTabs button[data-baseweb="tab"] {
    background: transparent !important;
    color: #29241E !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    padding: 12px 24px !important;
    min-height: 48px;
    font-weight: 500;
}

/* Selected tab */
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: #ED4524 !important;
    color: #FFFFFF !important;
    border-color: #ED4524 !important;
}

/* Make sure the text inside the selected tab is white */
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {
    color: #FFFFFF !important;
}

/* Unselected tab */
.stTabs button[data-baseweb="tab"][aria-selected="false"] p {
    color: #29241E !important;
}




/* Form controls */
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div { background: #FFF9EC !important;
    color: #28241F !important; border: 1px solid #C9B89F !important;
    border-radius: 12px !important; }
[data-testid="stTextInput"] input:focus { border-color: #E94725 !important; }
[data-testid="stToggle"] [data-checked="true"],
[data-testid="stCheckbox"] [data-checked="true"] { background-color: #E94725 !important; }
[data-testid="stRadio"] label { color: #28241F !important; }
/* Buttons */
.stButton > button[kind="primary"] { background: #E94725 !important;
    border: 1px solid #E94725 !important; color: white !important;
    border-radius: 999px !important; font-weight: 700 !important;
    padding: .85rem 1.4rem !important; min-height: 3.3rem;
    box-shadow: 3px 4px 0 #28241F; transition: transform .15s, box-shadow .15s; }
.stButton > button[kind="primary"]:hover { background: #D83E1E !important;
    transform: translate(1px, 2px); box-shadow: 2px 2px 0 #28241F; }
/* Cards, metrics, explanations */
[data-testid="stMetric"] { background: #FBE0CC; border: 1px solid #D9A88A;
    border-radius: 16px; padding: 1rem 1.25rem; }
[data-testid="stMetricValue"] { color: #E94725 !important;
    font-family: 'DM Serif Display', Georgia, serif; }
[data-testid="stExpander"] { border: 1px solid #D4BBA0 !important;
    border-radius: 14px !important; background: #F7F0E0; }
[data-testid="stAlert"] { background: #FBE0CC !important;
    border-radius: 12px; color: #28241F !important; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
/* Section panels created with st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #D8BCA1 !important;
    border-radius: 18px !important; background: #F7F0E0 !important; }
@media (max-width: 640px) { .block-container { padding-top: 1.8rem; }
    h1 { font-size: 2.8rem !important; } }
</style>
""", unsafe_allow_html=True)


def style_chart(fig, height=None):
    """Apply the same cream-and-orange styling to all Plotly charts."""
    fig.update_layout(
        paper_bgcolor=CREAM,
        plot_bgcolor=CREAM,
        font=dict(family="DM Sans, Arial, sans-serif", color=INK, size=13),
        title_font=dict(family="DM Sans, Arial, sans-serif", color=INK),
        margin=dict(l=12, r=18, t=24, b=42),
        showlegend=False,
        **({"height": height} if height else {}),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#CDBBA1")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="#CDBBA1")
    return fig


# --------------------------------------------------
# 2. Load the saved model and dashboard files
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dashboard_data():
    restaurants = pd.read_csv(DASHBOARD_DIR / "restaurants.csv")
    importance = pd.read_csv(DASHBOARD_DIR / "feature_importance.csv")
    return restaurants, importance


model_bundle = load_model()
model = model_bundle["model"]
baseline = model_bundle["baseline"]
feature_columns = model_bundle["feature_columns"]
categorical_columns = model_bundle["categorical_columns"]
mae = model_bundle["mae"]
restaurant_count = model_bundle["restaurant_count"]


# --------------------------------------------------
# 3. Helper functions for predictions and charts
# --------------------------------------------------

def build_restaurant_features(selected_attributes):
    """Combine the user's selections with baseline training values."""
    restaurant = baseline.copy()

    for attribute, value in selected_attributes.items():
        if attribute in feature_columns:
            restaurant[attribute] = value

    # The encoder was fitted on strings, including 'True' and 'False'.
    for column in categorical_columns:
        value = restaurant.get(column, "Missing")
        restaurant[column] = "Missing" if pd.isna(value) else str(value)

    return pd.DataFrame([restaurant], columns=feature_columns)


def calculate_feature_sensitivity(restaurant_features):
    """Compare a feature switched off vs. on, holding all else fixed."""
    comparison_attributes = {
        "Outdoor seating": "OutdoorSeating",
        "Delivery": "RestaurantsDelivery",
        "Happy hour": "HappyHour",
        "Reservations": "RestaurantsReservations",
        "Table service": "RestaurantsTableService",
        "Good for groups": "RestaurantsGoodForGroups",
        "Takeout": "RestaurantsTakeOut",
        "Good for kids": "GoodForKids",
        "Dogs allowed": "DogsAllowed",
        "Bike parking": "BikeParking",
        "Has a TV": "HasTV",
        "Wheelchair accessible": "WheelchairAccessible",
    }

    results = []
    for label, attribute in comparison_attributes.items():
        if attribute not in feature_columns:
            continue

        without_feature = restaurant_features.copy()
        with_feature = restaurant_features.copy()
        without_feature[attribute] = "False"
        with_feature[attribute] = "True"

        rating_without = float(model.predict(without_feature)[0])
        rating_with = float(model.predict(with_feature)[0])
        results.append({
            "Characteristic": label,
            "Without": rating_without,
            "With": rating_with,
            "Difference": rating_with - rating_without,
        })

    return pd.DataFrame(results)


def show_explore_dashboard():
    """Display feature importance and actual Yelp rating comparisons."""
    restaurants, importance = load_dashboard_data()

    st.subheader("Explore the Data")
    st.write(
        "Explore patterns across highly reviewed Yelp restaurants and see "
        "which characteristics the model uses most."
    )
    st.metric("Restaurants analyzed", f"{len(restaurants):,}")

    # Feature importance: scores are already aggregated by original attribute.
    st.subheader("Feature importance")

    feature_labels = {
        "OutdoorSeating": "Outdoor seating",
        "HappyHour": "Happy hour",
        "RestaurantsDelivery": "Delivery",
        "HasTV": "Has TV",
        "BikeParking": "Bike parking",
        "DogsAllowed": "Dogs allowed",
        "WheelchairAccessible": "Wheelchair accessible",
        "RestaurantsTableService": "Table service",
    }

    top_features = importance.nlargest(10, "importance").copy()
    top_features["attribute"] = top_features["attribute"].replace(feature_labels)
    top_features = top_features.sort_values("importance")

    importance_fig = px.bar(
        top_features,
        x="importance",
        y="attribute",
        orientation="h",
        labels={"importance": "Feature importance", "attribute": ""},
        color_discrete_sequence=[ORANGE],
    )
    importance_fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(style_chart(importance_fig, 450), use_container_width=True)
    st.caption(
        "Feature importance shows how much the Random Forest relies on "
        "each attribute. It does not show whether an attribute raises or "
        "lowers ratings, and it is not a causal measure."
    )

    st.divider()
    st.subheader("Average Yelp ratings by characteristic")

    attribute_options = {
        "Outdoor seating": "OutdoorSeating",
        "Happy hour": "HappyHour",
        "Delivery": "RestaurantsDelivery",
        "Reservations": "RestaurantsReservations",
        "Table service": "RestaurantsTableService",
        "Dogs allowed": "DogsAllowed",
        "Alcohol offerings": "Alcohol",
        "Wi-Fi": "WiFi",
        "Noise level": "NoiseLevel",
    }

    selected_attribute = st.selectbox(
        "Choose a restaurant characteristic",
        options=list(attribute_options),
    )
    attribute_column = attribute_options[selected_attribute]

    # Exclude missing values; keep only actual observed ratings.
    comparison_data = restaurants[["stars", attribute_column]].dropna().copy()
    rating_summary = (
        comparison_data.groupby(attribute_column)["stars"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            attribute_column: "Category",
            "mean": "Average rating",
            "count": "Restaurants",
        })
    )

    # pd.read_csv may parse True/False as booleans; string conversion
    # handles both booleans and text categories.
    category_labels = {
        "True": "Yes",
        "False": "No",
        "full_bar": "Full bar",
        "beer_and_wine": "Beer & wine",
        "none": "No alcohol",
        "free": "Free",
        "paid": "Paid",
        "no": "No Wi-Fi",
        "average": "Average",
        "quiet": "Quiet",
        "loud": "Loud",
        "very_loud": "Very loud",
    }
    rating_summary["Category"] = (
        rating_summary["Category"].astype(str).replace(category_labels)
    )
    rating_summary = rating_summary.sort_values("Average rating", ascending=False)

    ratings_fig = px.bar(
        rating_summary,
        x="Category",
        y="Average rating",
        color="Category",
        text="Average rating",
        hover_data={"Restaurants": True, "Average rating": ":.2f"},
        color_discrete_sequence=ORANGE_PALETTE,
    )
    ratings_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )
    ratings_fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 5.2],
        height=400,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(style_chart(ratings_fig, 400), use_container_width=True)
    st.caption(
        "These are observed average Yelp ratings, not model predictions. "
        "Restaurants with missing values for the selected attribute are "
        "excluded. Differences are associations, not causal effects."
    )
    st.dataframe(
        rating_summary.round({"Average rating": 2}),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# 4. Page heading and tabs
# --------------------------------------------------

st.title("Build Your Restaurant")
st.write(
    "Customize your restaurant and see its predicted Yelp rating."
)
st.caption(
    f"Trained on {restaurant_count:,} Yelp restaurants "
    "with at least 500 reviews each."
)

builder_tab, explore_tab = st.tabs([
    "Build Your Restaurant",
    "Explore the Data",
])


# --------------------------------------------------
# 5. Restaurant builder tab
# --------------------------------------------------

with builder_tab:
    with st.container(border=True):
        st.subheader("Your restaurant")
        restaurant_name = st.text_input(
            "What would you name your restaurant?",
            value="Tomato Club",
        )

        price_options = {
            "Budget": 1,
            "Moderate": 2,
            "Upscale": 3,
            "Luxury": 4,
        }
        selected_price = st.radio(
            "Price range",
            options=list(price_options),
            index=1,
            horizontal=True,
        )
        price_range = price_options[selected_price]


    with st.container(border=True):
        st.subheader("Dining experience")
        col1, col2 = st.columns(2)
        with col1:
            outdoor_seating = st.toggle("Outdoor seating", value=True)
            reservations = st.toggle("Takes reservations", value=True)
            table_service = st.toggle("Table service", value=True)
            good_for_groups = st.toggle("Good for groups", value=True)
        with col2:
            delivery = st.toggle("Delivery available", value=False)
            takeout = st.toggle("Takeout available", value=True)
            good_for_kids = st.toggle("Good for kids", value=True)


    with st.container(border=True):
        st.subheader("Food & drinks")
        alcohol = st.selectbox(
            "Alcohol offerings",
            options=["None", "Beer & Wine", "Full Bar"],
            index=1,
        )
        happy_hour = st.toggle("Happy hour", value=True)


    with st.container(border=True):
        st.subheader("Amenities")
        col1, col2 = st.columns(2)
        with col1:
            wifi = st.selectbox("Wi-Fi", ["Free", "Paid", "None"], index=0)
            dogs_allowed = st.toggle("Dogs allowed", value=False)
            bike_parking = st.toggle("Bike parking", value=True)
        with col2:
            noise_level = st.selectbox(
                "Noise level",
                ["Quiet", "Average", "Loud", "Very Loud"],
                index=1,
            )
            has_tv = st.toggle("Has a TV", value=False)
            wheelchair_accessible = st.toggle(
                "Wheelchair accessible", value=True
            )


    # Map controls to categories observed by the fitted encoder.
    alcohol_values = {
        "None": "none",
        "Beer & Wine": "beer_and_wine",
        "Full Bar": "full_bar",
    }
    wifi_values = {"Free": "free", "Paid": "paid", "None": "no"}
    noise_values = {
        "Quiet": "quiet",
        "Average": "average",
        "Loud": "loud",
        "Very Loud": "very_loud",
    }

    selected_attributes = {
        "RestaurantsPriceRange2": price_range,
        "OutdoorSeating": outdoor_seating,
        "RestaurantsReservations": reservations,
        "RestaurantsTableService": table_service,
        "RestaurantsGoodForGroups": good_for_groups,
        "RestaurantsDelivery": delivery,
        "RestaurantsTakeOut": takeout,
        "GoodForKids": good_for_kids,
        "HappyHour": happy_hour,
        "DogsAllowed": dogs_allowed,
        "BikeParking": bike_parking,
        "HasTV": has_tv,
        "WheelchairAccessible": wheelchair_accessible,
        "Alcohol": alcohol_values[alcohol],
        "WiFi": wifi_values[wifi],
        "NoiseLevel": noise_values[noise_level],
    }

    st.divider()
    if st.button("Predict My Rating", type="primary", use_container_width=True):
        restaurant_features = build_restaurant_features(selected_attributes)
        predicted_rating = float(model.predict(restaurant_features)[0])

        st.subheader(restaurant_name or "Your Restaurant")
        st.metric("Predicted Yelp Rating", f"{predicted_rating:.2f} / 5.00")
        st.write("⭐" * max(0, min(5, round(predicted_rating))))
        st.caption(f"Held-out test mean absolute error: {mae:.3f} stars.")
        st.info(
            "This is an illustrative prediction based on historical Yelp "
            "data. Adding an amenity does not necessarily cause a "
            "restaurant's actual rating to change."
        )

        st.divider()
        st.subheader("What changes your predicted rating?")
        st.write(
            "Switch one characteristic on or off while keeping all "
            "other restaurant characteristics the same."
        )
        sensitivity_results = calculate_feature_sensitivity(restaurant_features)
        sensitivity_results = sensitivity_results.sort_values("Difference")

        sensitivity_fig = px.bar(
            sensitivity_results,
            x="Difference",
            y="Characteristic",
            orientation="h",
            color="Difference",
            color_continuous_scale=["#F6B89B", "#E94725", "#B93119"],
            hover_data={
                "Without": ":.2f",
                "With": ":.2f",
                "Difference": ":+.3f",
            },
            labels={
                "Difference": "Difference in predicted Yelp stars",
                "Characteristic": "",
            },
            title="Predicted rating: With vs. Without",
        )
        sensitivity_fig.update_layout(
            coloraxis_showscale=False,
            height=520,
            margin=dict(l=10, r=10, t=60, b=30),
        )
        sensitivity_fig.add_vline(x=0, line_width=1, line_color="gray")
        st.plotly_chart(style_chart(sensitivity_fig, 520), use_container_width=True)
        st.caption(
            "Positive bars mean there is a higher predicted rating with the "
            "characteristic. Negative bars mean a lower predicted rating. "
            "Each comparison changes one restaurant feature while keeping everything else the same. The differences reflect the model's predictions, not guaranteed changes in real Yelp ratings."
        )

    st.divider()
    with st.expander("How does this work?"):
        st.write(
            "A Random Forest regression model learns patterns between "
            "restaurant characteristics and Yelp star ratings."
        )
        st.write(
            "You can customize 16 characteristics. For the other model "
            "inputs, we use representative values from the training data."
        )
        st.write(
            f"The model was trained using {restaurant_count:,} restaurants "
            "with at least 500 reviews and achieved a held-out test "
            f"mean absolute error of {mae:.3f} stars."
        )


# --------------------------------------------------
# 6. Explore the Data tab
# --------------------------------------------------

with explore_tab:
    if not (DASHBOARD_DIR / "restaurants.csv").exists() or not (
        DASHBOARD_DIR / "feature_importance.csv"
    ).exists():
        st.warning(
            "Dashboard CSV files are missing. Run "
            "`python prepare_dashboard_data.py` first."
        )
    else:
        show_explore_dashboard()
