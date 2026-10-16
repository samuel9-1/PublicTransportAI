
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="TransitAI | Fare Analytics",
    page_icon="🚌",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f3f6fb;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e0e7f0;
    }
    .hero {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(115deg, #102a43, #176b87);
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 {color: white; margin-bottom: 8px;}
    .hero p {color: #e5f4ff; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)


# -------------------- DATA LOADING --------------------

DATA_FILE = Path(__file__).with_name(
    "Public_Transport_Demand_Dataset_150k.csv"
)

st.sidebar.title("🚌 TransitAI")
st.sidebar.caption("Public Transport Analytics")
st.sidebar.divider()

if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Place "
        "Public_Transport_Demand_Dataset_150k.csv "
        "in the same folder as app.py."
    )
    st.stop()


@st.cache_data
def load_data(path, modified_time):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Remove duplicate records and entirely empty columns.
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")

    return df


data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)


# -------------------- COLUMN VALIDATION --------------------

target = "Fare_INR"

if target not in data.columns:
    st.error(
        f"Expected target column '{target}' was not found. "
        f"Available columns: {', '.join(data.columns)}"
    )
    st.stop()

data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target])
data = data[data[target] >= 0].copy()

if len(data) < 20:
    st.error("At least 20 valid fare records are needed for this demo.")
    st.stop()


# -------------------- FEATURE ENGINEERING --------------------

def extract_hour(value):
    """Extract hour from strings such as 08:30, 8:30 AM or 08:30:00."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if not pd.isna(parsed):
            return parsed.hour
    except (ValueError, TypeError):
        pass

    match = re.search(r"(\d{1,2}):\d{2}", text)
    if match:
        hour = int(match.group(1))
        if 0 <= hour <= 23:
            return hour

    return np.nan


def prepare_features(df):
    result = df.copy()

    # Convert departure/arrival time into numeric hours.
    for col, new_col in [
        ("Departure_Time", "Departure_Hour"),
        ("Arrival_Time", "Arrival_Hour"),
    ]:
        if col in result.columns:
            result[new_col] = result[col].map(extract_hour)

    preferred = [
        "Source",
        "Destination",
        "Operator",
        "Bus_Type",
        "Distance_km",
        "Duration_hours",
        "Total_Seats",
        "Departure_Hour",
        "Arrival_Hour",
    ]

    features = [c for c in preferred if c in result.columns]

    if not features:
        # Fallback: use available columns except the target.
        features = [
            c for c in result.columns
            if c != target and c not in [
                "Departure_Time", "Arrival_Time"
            ]
        ]

    X = result[features].copy()

    # Remove identifier-like or unsupported raw fields if present.
    X = X.replace([np.inf, -np.inf], np.nan)

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for col in categorical:
        X[col] = X[col].astype("string").fillna("Unknown")

    # Convert numeric-looking columns safely.
    for col in X.columns:
        if col not in categorical:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Exclude features that contain no usable values.
    X = X.dropna(axis=1, how="all")

    return X


X = prepare_features(data)
y = data.loc[X.index, target]

categorical_features = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

numeric_features = [
    c for c in X.columns if c not in categorical_features
]

if X.empty or len(X.columns) == 0:
    st.error("No usable prediction features were found in the dataset.")
    st.stop()


# -------------------- MODEL TRAINING --------------------

@st.cache_resource
def train_model(X, y):
    # Random split is appropriate for this non-time-series demo.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                max_categories=100,
            )),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
    }

    return model, X_test, y_test, predictions, metrics


with st.spinner("Training Random Forest model on your dataset..."):
    model, X_test, y_test, predictions, metrics = train_model(X, y)


# -------------------- SIDEBAR NAVIGATION --------------------

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Predict Fare",
        "Analytics",
        "Model Evaluation",
        "Dataset Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Model: Random Forest Regressor")
st.sidebar.caption(f"Valid records: {len(data):,}")


# -------------------- SHARED HEADER --------------------

st.markdown("""
<div class="hero">
    <h1>TransitAI</h1>
    <p>Intelligent public transport fare analytics powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# -------------------- OVERVIEW --------------------

if page == "Overview":
    st.subheader("Project Dashboard")
    st.write(
        "Explore bus fare patterns, estimate fares for trip details, "
        "and evaluate the model using held-out data."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Records", f"{len(data):,}")
    c2.metric("Average Fare", f"₹{data[target].mean():,.2f}")
    c3.metric(
        "Average Distance",
        f"{data['Distance_km'].mean():,.1f} km"
        if "Distance_km" in data.columns
        else "Not available",
    )
    c4.metric("Test R²", f"{metrics['r2']:.3f}")

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            data,
            x=target,
            nbins=35,
            title="Distribution of Bus Fares",
            labels={target: "Fare (INR)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "Bus_Type" in data.columns:
            summary = (
                data.groupby("Bus_Type", dropna=False)[target]
                .mean()
                .reset_index()
                .sort_values(target, ascending=False)
            )
            fig = px.bar(
                summary,
                x="Bus_Type",
                y=target,
                title="Average Fare by Bus Type",
                labels={
                    "Bus_Type": "Bus Type",
                    target: "Average Fare (INR)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bus_Type column is unavailable.")

    st.info(
        "R² is a regression evaluation metric, not classification accuracy. "
        "A higher R² is generally better; MAE and RMSE should be lower."
    )


# -------------------- FARE PREDICTION --------------------

elif page == "Predict Fare":
    st.subheader("Estimate a Bus Fare")
    st.write("Enter trip information and request a model prediction.")

    with st.form("fare_form"):
        values = {}

        col1, col2 = st.columns(2)

        for index, feature in enumerate(X.columns):
            container = col1 if index % 2 == 0 else col2

            if feature in categorical_features:
                choices = sorted(
                    X[feature].dropna().astype(str).unique().tolist()
                )
                choices = [v for v in choices if v != "Unknown"]

                if choices:
                    default_index = 0
                    if feature == "Bus_Type":
                        luxury = [
                            v for v in choices
                            if "ac" in v.lower()
                            or "volvo" in v.lower()
                        ]
                        if luxury:
                            default_index = choices.index(luxury[0])

                    with container:
                        values[feature] = st.selectbox(
                            feature.replace("_", " "),
                            choices,
                            index=default_index,
                        )
                else:
                    with container:
                        values[feature] = st.text_input(
                            feature.replace("_", " "),
                            value="Unknown",
                        )
            else:
                series = pd.to_numeric(X[feature], errors="coerce")
                median = series.median()
                median = 0.0 if pd.isna(median) else float(median)

                if feature == "Distance_km":
                    minimum = 1.0
                elif feature in ["Departure_Hour", "Arrival_Hour"]:
                    minimum = 0.0
                else:
                    minimum = 0.0

                with container:
                    values[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=minimum,
                        value=max(minimum, median),
                        step=1.0,
                    )

        submitted = st.form_submit_button(
            "Predict Fare",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([values], columns=X.columns)

        for feature in categorical_features:
            input_df[feature] = input_df[feature].astype(str)

        fare = max(0.0, float(model.predict(input_df)[0]))

        st.success(f"Estimated fare: ₹{fare:,.2f}")
        st.caption(
            "This is a model estimate, not an official operator quote. "
            "Actual fares can vary."
        )


# -------------------- ANALYTICS --------------------

elif page == "Analytics":
    st.subheader("Transport Analytics")

    if "Distance_km" in data.columns:
        chart_data = data.dropna(subset=["Distance_km", target])
        fig = px.scatter(
            chart_data.sample(min(5000, len(chart_data)), random_state=42),
            x="Distance_km",
            y=target,
            color="Bus_Type" if "Bus_Type" in chart_data.columns else None,
            title="Distance vs Fare",
            labels={
                "Distance_km": "Distance (km)",
                target: "Fare (INR)",
            },
            opacity=0.65,
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Source" in data.columns and "Destination" in data.columns:
        routes = (
            data.assign(
                Route=data["Source"].astype(str)
                + " → "
                + data["Destination"].astype(str)
            )
            .groupby("Route")
            .agg(
                Trips=(target, "size"),
                Average_Fare=(target, "mean"),
            )
            .reset_index()
            .sort_values("Trips", ascending=False)
            .head(15)
        )

        fig = px.bar(
            routes.sort_values("Trips"),
            x="Trips",
            y="Route",
            orientation="h",
            title="Most Frequent Routes in the Dataset",
            labels={"Trips": "Number of Records"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(routes, use_container_width=True)

    if "Duration_hours" in data.columns:
        fig = px.scatter(
            data.dropna(subset=["Duration_hours", target]).sample(
                min(5000, len(data.dropna(subset=["Duration_hours", target]))),
                random_state=42,
            ),
            x="Duration_hours",
            y=target,
            title="Journey Duration vs Fare",
            labels={
                "Duration_hours": "Duration (hours)",
                target: "Fare (INR)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)


# -------------------- MODEL EVALUATION --------------------

elif page == "Model Evaluation":
    st.subheader("Model Performance")
    st.write(
        "These metrics are calculated on the 20% test split, "
        "which was not used to fit the model."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['r2']:.4f}")
    c2.metric("MAE", f"₹{metrics['mae']:,.2f}")
    c3.metric("RMSE", f"₹{metrics['rmse']:,.2f}")

    comparison = pd.DataFrame({
        "Actual Fare": y_test.to_numpy(),
        "Predicted Fare": predictions,
    })

    fig = px.scatter(
        comparison.sample(
            min(3000, len(comparison)), random_state=42
        ),
        x="Actual Fare",
        y="Predicted Fare",
        title="Actual vs Predicted Fare",
        opacity=0.65,
    )

    low = min(
        comparison["Actual Fare"].min(),
        comparison["Predicted Fare"].min(),
    )
    high = max(
        comparison["Actual Fare"].max(),
        comparison["Predicted Fare"].max(),
    )

    fig.add_shape(
        type="line",
        x0=low, y0=low,
        x1=high, y1=high,
        line=dict(dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("Sample predictions")
    st.dataframe(comparison.head(30), use_container_width=True)

    st.download_button(
        "Download test predictions (CSV)",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="fare_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "R² can be negative when the model performs worse than predicting "
        "the test-set mean. MAE is the average absolute error; RMSE penalizes "
        "larger errors more heavily."
    )


# -------------------- DATASET EXPLORER --------------------

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")

    st.write("Available columns:", ", ".join(data.columns))

    filtered = data.copy()

    if "Bus_Type" in filtered.columns:
        types = sorted(
            filtered["Bus_Type"].dropna().astype(str).unique().tolist()
        )
        selected_types = st.multiselect(
            "Filter by bus type",
            types,
            default=types,
        )
        filtered = filtered[
            filtered["Bus_Type"].astype(str).isin(selected_types)
        ]

    if "Source" in filtered.columns:
        sources = sorted(
            filtered["Source"].dropna().astype(str).unique().tolist()
        )
        selected_sources = st.multiselect(
            "Filter by source",
            sources,
        )
        if selected_sources:
            filtered = filtered[
                filtered["Source"].astype(str).isin(selected_sources)
            ]

    st.caption(f"Showing {len(filtered):,} records")
    st.dataframe(filtered.head(1000), use_container_width=True)

    st.download_button(
        "Download filtered dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_transport_data.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "TransitAI | Machine Learning Demonstration | "
    "Predictions depend on the quality and coverage of the supplied dataset."
)
