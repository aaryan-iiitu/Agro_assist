import streamlit as st
import joblib
import numpy as np

crop_dict={
    'rice': 1,
    'maize': 2,
    'chickpea': 3,
    'kidneybeans': 4,
    'pigeonpeas': 5,
    'mothbeans': 6,
    'mungbean': 7,
    'blackgram': 8,
    'lentil': 9,
    'pomegranate': 10,
    'banana': 11,
    'mango': 12,
    'grapes': 13,
    'watermelon': 14,
    'muskmelon': 15,
    'apple': 16,
    'orange': 17,
    'papaya': 18,
    'coconut': 19,
    'cotton': 20,
    'jute': 21,
    'coffee':22
}
reverse_crop_dict = {
    1: 'rice',
    2: 'maize',
    3: 'chickpea',
    4: 'kidneybeans',
    5: 'pigeonpeas',
    6: 'mothbeans',
    7: 'mungbean',
    8: 'blackgram',
    9: 'lentil',
    10: 'pomegranate',
    11: 'banana',
    12: 'mango',
    13: 'grapes',
    14: 'watermelon',
    15: 'muskmelon',
    16: 'apple',
    17: 'orange',
    18: 'papaya',
    19: 'coconut',
    20: 'cotton',
    21: 'jute',
    22: 'coffee'
}
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = scaler.transform(features)
    prediction = model.predict(mx_features).reshape(1, -1)
    crop_label = prediction[0][0]  # Access the scalar value
    crop_name = reverse_crop_dict[crop_label]  # Map numeric label to crop name
    return crop_name
# Load the saved model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("minmaxscaler.pkl")
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'crop_recommendation_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

st.set_page_config(
    page_title='Agro-assist'
)
# App title and description
st.title("🌾 Crop Recommendation System")
st.sidebar.success('Select a page above ')

st.write("""
This app recommends the most suitable crop based on environmental factors and soil properties.  
Please provide the following inputs to get started:
""")

# Input fields for environmental factors
temperature = st.number_input("🌡️ Temperature (°C):", min_value=0.0, step=0.1, format="%.1f")
humidity = st.number_input("💧 Humidity (%):", min_value=0.0, step=0.1, format="%.1f")
ph = st.number_input("🧪 Soil pH:", min_value=0.0, max_value=14.0, step=0.1, format="%.1f")
rainfall = st.number_input("🌧️ Rainfall (mm):", min_value=0.0, step=0.1, format="%.1f")

# Input fields for NPK levels
nitrogen = st.number_input("🌱 Nitrogen (N):", min_value=0.0, step=0.1, format="%.1f")
phosphorus = st.number_input("🧪 Phosphorus (P):", min_value=0.0, step=0.1, format="%.1f")
potassium = st.number_input("⚡ Potassium (K):", min_value=0.0, step=0.1, format="%.1f")

# Prediction button
if st.button("🌱 Recommend Crop"):
    try:
        # Prepare input data
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        scaled_data = scaler.transform(input_data)

        # Predict the crop
        prediction = model.predict(scaled_data)
        crop = recommendation(nitrogen,phosphorus,potassium, temperature, humidity, ph, rainfall)

        # Display the result
        st.success(f"Recommended Crop: **{crop}** 🌾")
    except Exception as e:
        st.error(f"An error occurred: {e}")
