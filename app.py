import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Bike Demand Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("day.csv")

data = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(data):
    X = data[['season','yr','mnth','holiday','weekday',
              'workingday','weathersit','temp','hum','windspeed']]
    y = data['cnt']

    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model(data)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Analytics", "Insights"])

# ---------------- HEADER ----------------
st.title("🚴 Bike Sharing Demand Prediction")
st.markdown("Interactive ML dashboard using Streamlit")
st.markdown("---")

# ===================== PREDICTION =====================
if page == "Prediction":

    st.subheader("📊 Enter Conditions")

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox("Season", [1,2,3,4])
        yr = st.selectbox("Year (0=2011, 1=2012)", [0,1])
        mnth = st.slider("Month", 1, 12)
        weekday = st.slider("Weekday", 0, 6)

    with col2:
        temp = st.slider("Temperature (0–1)", 0.0, 1.0)
        hum = st.slider("Humidity (0–1)", 0.0, 1.0)
        windspeed = st.slider("Windspeed (0–1)", 0.0, 1.0)
        weathersit = st.selectbox("Weather Condition", [1,2,3,4])

    holiday = st.selectbox("Holiday", [0,1])
    workingday = st.selectbox("Working Day", [0,1])

    st.markdown("---")

    if st.button("🚀 Predict Demand"):

        input_data = [[
            season, yr, mnth, holiday, weekday,
            workingday, weathersit, temp, hum, windspeed
        ]]

        prediction = int(model.predict(input_data)[0])

        st.success(f"🎯 Predicted Bike Demand: {prediction}")

        # KPI Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Demand", prediction)
        c2.metric("Temperature", temp)
        c3.metric("Humidity", hum)

        # Interpretation
        if prediction < 2000:
            st.warning("📉 Low demand expected")
        elif prediction < 5000:
            st.info("📊 Moderate demand expected")
        else:
            st.success("📈 High demand expected")

# ===================== ANALYTICS =====================
elif page == "Analytics":

    st.subheader("📈 Data Analytics")

    # KPI cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Demand", int(data['cnt'].sum()))
    c2.metric("Average Demand", int(data['cnt'].mean()))
    c3.metric("Max Demand", int(data['cnt'].max()))

    st.markdown("---")

    # Monthly trend
    monthly = data.groupby('mnth')['cnt'].mean()

    fig, ax = plt.subplots()
    ax.plot(monthly.index, monthly.values, marker='o')
    ax.set_title("Average Monthly Demand")
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand")

    st.pyplot(fig)

    # Weather impact
    weather = data.groupby('weathersit')['cnt'].mean()

    fig2, ax2 = plt.subplots()
    ax2.bar(weather.index, weather.values)
    ax2.set_title("Demand by Weather Condition")

    st.pyplot(fig2)

# ===================== INSIGHTS =====================
elif page == "Insights":

    st.subheader("🧠 Key Insights")

    st.markdown("""
### 📊 Observations:

- 📈 Higher temperature leads to increased bike demand  
- 🌧 Poor weather conditions reduce usage  
- 📅 Working days show stable demand patterns  
- 🔄 Seasonal changes significantly impact demand  

### 🤖 Model Insight:
Random Forest performs best as it captures complex relationships in the data.

### ✅ Conclusion:
Machine learning can effectively predict bike demand and support better resource planning.
""")

    st.markdown("---")
    st.info("This project demonstrates real-world application of machine learning.")