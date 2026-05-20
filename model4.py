import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model, encoders, and scaler ---
model = joblib.load("travelyatri_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# --- Streamlit UI ---
st.title("Travel Yatri - Smart Trip Budget Predictor")
st.write("Plan your next trip with AI! Get a smart budget estimate based on your preferences.")

# --- User Inputs ---

destination = st.text_input("📍 Enter your destination (e.g. Manali, Paris)")
country = st.text_input("🌎 Enter your country (e.g. India, France)")
type = st.selectbox("🏕️ Choose your type of destination", 
                     ["Coastal", "Heritage", "Island", "Cultural", "Desert", 
                      "Beach", "Hill Station", "Mountain", "City", "Spiritual"])

season = st.selectbox("☀️ Choose your season of travel", 
                      ["Autumn", "Summer", "Monsoon", "Winter", "Spring"])

besttime = st.selectbox("🗓️ Best time to visit", 
                        ["April-June", "July-Sept", "October-March", "November-Febuary"])

activities = st.multiselect("🎯 Select activities you enjoy", 
                            ["Trekking", "Food Exploration", "Shopping", "Cultural Tours", 
                             "Adventure Sports", "Sightseeing", "Beach Activities", 
                             "Historical Site Visits"])

days = st.slider("🕒 Duration of trip (days)", 1, 15, 5)

# --- Prediction Function  ---
def predict_budget(user_input):
    try:
        destination_enc = encoders['Destination'].transform([user_input[0]])[0] if user_input[0] in encoders['Destination'].classes_ else 0
        country_enc     = encoders['Country'].transform([user_input[1]])[0] if user_input[1] in encoders['Country'].classes_ else 0
        type_enc        = encoders['Type'].transform([user_input[2]])[0] if user_input[2] in encoders['Type'].classes_ else 0
        season_enc      = encoders['Season'].transform([user_input[3]])[0] if user_input[3] in encoders['Season'].classes_ else 0
        duration_enc    = int(user_input[4])
        activity_enc    = encoders['Activities'].transform([user_input[5]])[0] if user_input[5] in encoders['Activities'].classes_ else 0
        besttime_enc    = encoders['Besttime'].transform([user_input[6]])[0] if user_input[6] in encoders['Besttime'].classes_ else 0

        # Combine all into array
        encoded_array = np.array([[destination_enc, country_enc, type_enc, season_enc,
                                   duration_enc, activity_enc, besttime_enc]])

        # Scale and predict
        scaled_array = scaler.transform(encoded_array)
        pred_encoded = model.predict(scaled_array)[0]
        pred_rounded = int(round(pred_encoded))

        # Decode to readable label
        if 0 <= pred_rounded < len(encoders['Budget'].classes_):
            return encoders['Budget'].inverse_transform([pred_rounded])[0]
        else:
            return "Unknown (out of range)"
    except Exception as e:
        return f"Error: {e}"

# --- Predict Button ---
if st.button("💰 Predict My Trip Budget"):
    try:
        activities_str = ", ".join(activities) if activities else "Sightseeing"

        # Order of input as trained
        user_input = [destination.title(), country.title(), type.title(), 
                      season.title(), str(days), activities_str.title(), besttime.title()]

        # Predict
        budget_label = predict_budget(user_input)
        st.success(f"🧾 Estimated Budget for your trip: **{budget_label}** per person!")

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {str(e)}")
