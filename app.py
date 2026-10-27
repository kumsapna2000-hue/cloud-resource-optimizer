
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Cloud Resource Optimizer",
    page_icon="☁️",
    layout="wide"
)

st.title("AI-Based Cloud Resource Monitoring & Predictive Optimization")
st.write("Machine learning for cloud resource usage prediction.")

# Generate sample dataset
np.random.seed(42)
n = 200

data = pd.DataFrame({
    "hour": np.arange(n),
    "cpu_usage": np.random.uniform(10, 90, n),
    "memory_usage": np.random.uniform(20, 85, n),
    "request_count": np.random.randint(100, 2000, n)
})

data["future_cpu"] = (
    0.45 * data["cpu_usage"]
    + 0.20 * data["memory_usage"]
    + 0.015 * (data["request_count"] / 10)
    + np.random.normal(0, 5, n)
).clip(0, 100)

# Display resource data
st.subheader("1. Cloud Resource Data")
st.dataframe(data.head(10), use_container_width=True)

# Train machine-learning models
X = data[["cpu_usage", "memory_usage", "request_count"]]
y = data["future_cpu"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    trained_models[name] = model

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    })

results_df = pd.DataFrame(results).set_index("Model")

# Model comparison
st.subheader("2. Model Performance Comparison")
st.dataframe(results_df.round(3))
st.bar_chart(results_df[["MAE", "RMSE"]])

# Interactive prediction
st.subheader("3. Predict Future CPU Usage")

cpu = st.slider("Current CPU Usage (%)", 0, 100, 75)
memory = st.slider("Memory Usage (%)", 0, 100, 70)
requests = st.number_input(
    "Request Count", min_value=0, max_value=100000, value=1500
)

chosen_model = st.selectbox(
    "Choose Prediction Model", list(trained_models.keys())
)

if st.button("Predict CPU Usage"):
    sample = pd.DataFrame({
        "cpu_usage": [cpu],
        "memory_usage": [memory],
        "request_count": [requests]
    })

    prediction = trained_models[chosen_model].predict(sample)[0]
    st.metric("Predicted Future CPU Usage", f"{prediction:.2f}%")

    if prediction >= 80:
        st.warning("High predicted CPU usage. Review resource capacity.")
    else:
        st.success("No high CPU warning from this model.")

# Resource visualization
st.subheader("4. CPU Usage Overview")
st.line_chart(data.set_index("hour")[["cpu_usage", "future_cpu"]])

# Download results
st.download_button(
    "Download Resource Dataset",
    data.to_csv(index=False),
    file_name="cloud_resource_data.csv",
    mime="text/csv"
)

st.caption(
    "Educational prototype using synthetic data. "
    "It does not monitor live AWS resources."
)
