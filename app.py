import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Page config
st.set_page_config(page_title="Bike Demand Predictor", layout="centered")

# Title
st.title("🚴 Bike Sharing Demand Prediction")
st.markdown("### Predict bike demand based on weather & conditions")

st.markdown("---")

# Load data
data = pd.read_csv("day.csv")
data = data.drop(['instant','dteday','casual','registered'], axis=1)

X = data[['season','yr','mnth','holiday','weekday',
          'workingday','weathersit','temp','hum','windspeed']]
y = data['cnt']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# ================= UI =================

st.subheader("📊 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Season", [1,2,3,4])
    mnth = st.slider("Month", 1, 12)
    weekday = st.slider("Weekday", 0, 6)
    workingday = st.selectbox("Working Day", [0,1])

with col2:
    temp = st.slider("Temperature (0–1)", 0.0, 1.0)
    hum = st.slider("Humidity (0–1)", 0.0, 1.0)
    windspeed = st.slider("Windspeed (0–1)", 0.0, 1.0)
    weathersit = st.selectbox("Weather Condition", [1,2,3,4])

yr = st.selectbox("Year (0 = 2011, 1 = 2012)", [0,1])
holiday = st.selectbox("Holiday", [0,1])

st.markdown("---")

# ================= Prediction =================

if st.button("🚀 Predict Demand"):

    input_data = [[season, yr, mnth, holiday, weekday,
                   workingday, weathersit, temp, hum, windspeed]]

    prediction = int(model.predict(input_data)[0])

    st.success(f"🎯 Predicted Bike Demand: {prediction}")

    # Progress bar (visual feel)
    st.progress(min(prediction / 8000, 1.0))

    st.markdown("### 📌 Interpretation")
    
    if prediction < 2000:
        st.warning("Low demand expected 🚶")
    elif prediction < 5000:
        st.info("Moderate demand 🚴")
    else:
        st.success("High demand 🚴‍♂️🔥")

# ================= Feature Importance =================

st.markdown("---")
st.subheader("📈 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_xlabel("Importance")

st.pyplot(fig)

st.markdown("💡 Temperature and weather conditions have strong impact on bike demand.")