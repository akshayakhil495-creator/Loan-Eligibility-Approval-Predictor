import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Eligibility Predictor",
                   page_icon="🏦",
                   layout="wide")

st.title("🏦 Loan Eligibility Approval Predictor")

st.markdown("---")

# Load Dataset

df = pd.read_csv("loan_data.csv")

st.subheader("Loan Dataset")

st.dataframe(df)

# Encoding

encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Marital_Status",
    "Education",
    "Employment_Status",
    "Existing_Loan",
    "Property_Area",
    "Loan_Status"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features

X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

st.success(f"Model Accuracy : {accuracy*100:.2f}%")

st.markdown("---")

st.subheader("Loan Status Distribution")

fig, ax = plt.subplots()

df["Loan_Status"].value_counts().plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

st.markdown("---")

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance)

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.barh(
    importance["Feature"],
    importance["Importance"]
)

st.pyplot(fig2)

st.markdown("---")

st.header("Predict Loan Eligibility")

gender = st.selectbox("Gender",["Male","Female"])

age = st.number_input(
    "Age",
    18,
    70,
    30
)

marital = st.selectbox(
    "Marital Status",
    ["Single","Married"]
)

education = st.selectbox(
    "Education",
    ["Graduate","Non-Graduate"]
)

employment = st.selectbox(
    "Employment Status",
    ["Employed","Self-Employed"]
)

income = st.number_input(
    "Annual Income",
    100000,
    5000000,
    500000
)

loan = st.number_input(
    "Loan Amount",
    50000,
    1000000,
    200000
)

credit = st.slider(
    "Credit Score",
    300,
    900,
    700
)

existing = st.selectbox(
    "Existing Loan",
    ["No","Yes"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban","Semiurban","Rural"]
)

if st.button("Predict"):

    input_df = pd.DataFrame({

        "Gender":[gender],
        "Age":[age],
        "Marital_Status":[marital],
        "Education":[education],
        "Employment_Status":[employment],
        "Annual_Income":[income],
        "Loan_Amount":[loan],
        "Credit_Score":[credit],
        "Existing_Loan":[existing],
        "Property_Area":[property_area]

    })

    # Encode

    for col in input_df.columns:

        if col in encoders:

            input_df[col] = encoders[col].transform(input_df[col])

    result = model.predict(input_df)

    if result[0] == 0:

        st.error("❌ Loan Rejected")

    else:

        st.success("✅ Loan Approved")