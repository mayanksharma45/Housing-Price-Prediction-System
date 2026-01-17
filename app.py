import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(
    page_title="Housing Price Prediction",
    layout="centered"
)

st.title("🏠 Housing Price Prediction Indicator")
st.markdown("Enter the house details below:")

# ---------------- INPUT FIELDS ----------------

No_of_Bedrooms = st.number_input(
    "No of Bedrooms",
    min_value=0,
    max_value=35,
    step=1
)

No_of_Bathrooms = st.number_input(
    "No of Bathrooms",
    min_value=0.0,
    max_value=8.0,
    step=0.25
)

Flat_Area = st.number_input(
    "Flat Area (in sqft)",
    min_value=0.0
)

Lot_Area = st.number_input(
    "Lot Area (in sqft)",
    min_value=0.0
)

No_of_Floors = st.number_input(
    "No of Floors",
    min_value=1.0,
    max_value=4.0,
    step=0.5
)

Waterfront_View = st.selectbox(
    "Waterfront View",
    options=["NO", "YES"]  
)

Condition_of_the_House = st.selectbox(
    "Condition of the House",
    options=["Fair", "Excellent", "Good", "Bad", "Okay"]
)

Overall_Grade = st.number_input(
    "Overall Grade (out of 10)",
    min_value=1,
    max_value=10,
    step=1
)

Area_of_the_House_from_Basement = st.number_input(
    "Area of the House from Basement (in sqft)",
    min_value=0.0
)

Basement_Area = st.number_input(
    "Basement Area (in sqft)",
    min_value=0
)

Age_of_House = st.number_input(
    "Age of House (years)",
    min_value=1,
    max_value=120,
    step=1
)

Zipcode = st.number_input(
    "Zipcode",
    min_value=0.0
)

Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

Living_Area_after_Renovation = st.number_input(
    "Living Area after Renovation (in sqft)",
    min_value=0.0
)

Lot_Area_after_Renovation = st.number_input(
    "Lot Area after Renovation (in sqft)",
    min_value=0
)

Ever_Renovated = st.selectbox(
    "Ever Renovated",
    options=["NO", "YES"]
)

Years_Since_Renovation = st.number_input(
    "Years Since Renovation",
    min_value=0.0
)

# ---------------- PREDICTION ----------------

if st.button("Predict your Sale Price"):
    data = CustomData(
        No_of_Bedrooms=No_of_Bedrooms,
        No_of_Bathrooms=No_of_Bathrooms,
        Flat_Area=Flat_Area,
        Lot_Area=Lot_Area,
        No_of_Floors=No_of_Floors,
        Waterfront_View=Waterfront_View,
        Condition_of_the_House=Condition_of_the_House,
        Overall_Grade=Overall_Grade,
        Area_of_the_House_from_Basement=Area_of_the_House_from_Basement,
        Basement_Area=Basement_Area,
        Age_of_House=Age_of_House,
        Zipcode=Zipcode,
        Latitude=Latitude,
        Longitude=Longitude,
        Living_Area_after_Renovation=Living_Area_after_Renovation,
        Lot_Area_after_Renovation=Lot_Area_after_Renovation,
        Ever_Renovated=Ever_Renovated,
        Years_Since_Renovation=Years_Since_Renovation,
    )

    df = data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)

    st.success(f"💰 THE PREDICTED SALE PRICE IS {prediction[0]:,.2f} INR")