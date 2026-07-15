import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load model
model = joblib.load("house_price_model.pkl")
st.sidebar.title("About")

st.sidebar.info("""
🏠 House Price Prediction

Machine Learning Model:
• Random Forest Regressor

Built using:
• Python
• Scikit-learn
• Streamlit
""")
st.title("🏠 House Price Prediction")
col1, col2 = st.columns(2)

with col1:
    CRIM = st.number_input("CRIM", value=0.1)
    ZN = st.number_input("ZN", value=18.0)
    INDUS = st.number_input("INDUS", value=2.3)
    CHAS = st.number_input("CHAS", value=0)
    NOX = st.number_input("NOX", value=0.5)
    RM = st.number_input("RM", value=6.5)

with col2:
    AGE = st.number_input("AGE", value=65.0)
    DIS = st.number_input("DIS", value=4.0)
    RAD = st.number_input("RAD", value=1)
    TAX = st.number_input("TAX", value=296)
    PTRATIO = st.number_input("PTRATIO", value=15.3)
    B = st.number_input("B", value=396.9)
    LSTAT = st.number_input("LSTAT", value=8.0)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "CRIM": [CRIM],
        "ZN": [ZN],
        "INDUS": [INDUS],
        "CHAS": [CHAS],
        "NOX": [NOX],
        "RM": [RM],
        "AGE": [AGE],
        "DIS": [DIS],
        "RAD": [RAD],
        "TAX": [TAX],
        "PTRATIO": [PTRATIO],
        "B": [B],
        "LSTAT": [LSTAT]
    })

    prediction = model.predict(input_data)

    st.metric(
        label="🏡 Predicted House Price",
        value=f"${prediction[0]:,.2f}"
    )
st.markdown("---")
st.caption("Developed by Ahmed | Machine Learning Portfolio Project")
