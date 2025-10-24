# pip install --upgrade scikit-learn scikeras

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model, scaler, columns
@st.cache_resource
def create_model():
    return joblib.load('NY_model_pipeline.joblib')
    

loaded_pipeline = create_model()

# User Interface
st.title("Передбачення ціни на будинок в Нью-Йорку")
house_type = st.selectbox(
    "Type",
    [
    "Condo for sale",
    "House for sale",
    "Co-op for sale",
    "Multi-family home for sale"
    ]
)
beds = st.slider(
    "Спальні",
    min_value=1,
    max_value=36,
    value=2,
    step=1
)
bath = st.slider(
    "Ванни",
    min_value=1,
    max_value=32,
    value=2,
    step=1
)
propertysqft = st.slider(
    "Квадратна площа житла",
    min_value=200,
    max_value=21000,
    value=1091,
    step=10
)
LATITUDE = st.number_input("Широта", value=40.71, min_value=39.0, max_value=42.0)
LONGITUDE = st.number_input("Довгота", value=-73.94, min_value=-76.0, max_value=-72.0)
avg_price_street = st.slider(
    "Середня ціна будинків на вулиці (модель має високу залежність від цього параметру)",
    min_value=190000.00,
    max_value=30000000.00,
    value=800410.00,
    step=1000.00,
    format='$%d'

)

new_data = pd.DataFrame([
    {
        "TYPE": house_type,
        "BEDS": beds,
        "BATH": bath,
        "PROPERTYSQFT": propertysqft,
        "LATITUDE": LATITUDE,
        "LONGITUDE": LONGITUDE,
        "avg_price_street": avg_price_street
    }
])

if "prediction" not in st.session_state:
    st.session_state.prediction = None


#  Prediction
if st.button("Визначити ціну"):
    input_data = new_data
    pred = loaded_pipeline.predict(input_data)
    st.session_state.prediction = pred[0]


if st.session_state.prediction is not None:
    st.markdown(f"**Орієнтовна ціна будинку:** ${st.session_state.prediction:,.2f}")


st.caption("Варто зауважити, що модель може поводити себе в деяких випадках нелогічно, однак це проблема її навчання та якості. Вебзастосунок працює коректно та без збоїв.")

