import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# ---------------- Page setup ----------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 100%);}
.stApp, .stApp label, .stApp p, .stApp span, .stApp li {color: #1f2937;}
#MainMenu, footer {visibility: hidden;}
.hero {
    background: linear-gradient(120deg, #4f46e5, #9333ea, #ec4899);
    padding: 34px 40px; border-radius: 22px; margin-bottom: 26px;
    box-shadow: 0 12px 30px rgba(79,70,229,0.25);
}
.hero h1 {color: #ffffff !important; margin: 0; font-size: 2.4rem;}
.hero p {color: #ffffff !important; margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.92;}
.card {
    background: #ffffff; padding: 18px 20px; border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.07); text-align: center;
    border-top: 5px solid #6366f1;
}
.card .big {font-size: 1.7rem; font-weight: 700; color: #4f46e5;}
.card .small {font-size: 0.85rem; color: #6b7280; letter-spacing: 0.5px;}
.result {
    background: linear-gradient(120deg, #10b981, #06b6d4);
    padding: 30px; border-radius: 20px; text-align: center; margin-top: 18px;
    box-shadow: 0 12px 30px rgba(16,185,129,0.3);
}
.result h2 {color: #ffffff !important; margin: 0; font-size: 2.8rem;}
.result p {color: #ffffff !important; margin: 4px 0 0 0;}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(120deg, #4f46e5, #9333ea); color: #ffffff;
    border: none; border-radius: 12px; padding: 0.65rem 2rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚗 Car Price Predictor</h1>
  <p>Upload your dataset, train a Linear Regression model, and estimate a car's price instantly.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Same encodings as the notebook ----------------
NUM_COLS = ["normalized_losses", "bore", "stroke", "horsepower", "peak_rpm", "price"]

MAPS = {
    "Fuel_type": {"gas": 0, "diesel": 1},
    "make": {"std": 0, "turbo": 1},
    "num_of_doors": {"two": 2, "four": 4},
    "aspiration": {"convertible": 0, "hatchback": 1, "sedan": 2, "wagon": 3, "hardtop": 4},
    "wheel_base": {"rwd": 0, "fwd": 1, "4wd": 2},
    "engine_location": {"front": 0, "rear": 1},
    "engine_type": {"dohc": 0, "ohcv": 1, "ohc": 2, "l": 3, "rotor": 4, "ohcf": 5, "dohcv": 6},
    "num_of_cylinders": {"four": 4, "six": 6, "five": 5, "three": 3,
                         "twelve": 12, "two": 2, "eight": 8},
}

# Friendly names (your CSV header is shifted, so these describe what the values really are)
LABELS = {
    "Fuel_type": "Fuel Type", "make": "Aspiration", "num_of_doors": "Number of Doors",
    "aspiration": "Body Style", "wheel_base": "Drive Wheels", "engine_location": "Engine Location",
    "engine_type": "Engine Type", "num_of_cylinders": "Number of Cylinders",
    "body_style": "Body Style (encoded)", "fuel_system": "Fuel System",
    "normalized_losses": "Normalized Losses", "bore": "Bore", "stroke": "Stroke",
    "horsepower": "Horsepower", "peak_rpm": "Peak RPM",
}


def nice(col):
    return LABELS.get(col, col.replace("_", " ").title())


# ---------------- Prepare data + train ----------------
@st.cache_data(show_spinner=False)
def prepare_and_train(raw):
    car = raw.copy()
    car.replace("?", np.nan, inplace=True)

    for c in NUM_COLS:
        if c in car.columns:
            car[c] = pd.to_numeric(car[c], errors="coerce")

    for c in car.columns:                      # fill missing values with the mode
        car[c] = car[c].fillna(car[c].mode()[0])

    options = {}                               # label -> code, used for dropdowns in the form
    for c, mp in MAPS.items():
        if c in car.columns:
            car[c] = pd.to_numeric(car[c].replace(mp), errors="coerce")
            options[c] = mp

    for c in car.select_dtypes(include="object").columns[:-1]:   # body_style, fuel_system, etc.
        le = LabelEncoder()
        car[c] = le.fit_transform(car[c].astype(str))
        options[c] = {name: int(i) for i, name in enumerate(le.classes_)}

    car = car.apply(pd.to_numeric, errors="coerce").fillna(0)

    train, test = train_test_split(car, test_size=0.2, random_state=42)
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    model = LinearRegression().fit(X_train, y_train)

    pred_test = model.predict(X_test)
    N, K = X_train.shape
    r2_train = model.score(X_train, y_train)
    metrics = {
        "r2_train": r2_train,
        "adj_r2": 1 - (1 - r2_train) * (N - 1) / (N - K - 1),
        "r2_test": r2_score(y_test, pred_test),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mape": float(mean_absolute_percentage_error(y_test, pred_test) * 100),
    }
    compare = pd.DataFrame({"Actual": y_test.values, "Predicted": pred_test})
    stats = X_train.agg(["min", "max", "median"]).T
    return model, metrics, compare, options, stats, list(X_train.columns), car


# ---------------- Upload ----------------
uploaded = st.file_uploader("📂 Upload car_price.csv", type=["csv"])

if uploaded is None:
    st.info("Please upload your car_price.csv file to begin.")
    st.stop()

raw = pd.read_csv(uploaded)
with st.spinner("Training the model..."):
    model, m, compare, options, stats, features, clean = prepare_and_train(raw)

tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Model Performance", "🗂️ Data"])

# ---------------- Tab 1: Predict ----------------
with tab1:
    st.subheader("Enter car details")
    with st.form("predict_form"):
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(features):
            with cols[i % 3]:
                if f in options:
                    choice = st.selectbox(nice(f), list(options[f].keys()), key=f)
                    values[f] = options[f][choice]
                else:
                    lo, hi, med = (float(stats.loc[f, "min"]), float(stats.loc[f, "max"]),
                                   float(stats.loc[f, "median"]))
                    is_int = float(lo).is_integer() and float(hi).is_integer() and float(med).is_integer()
                    values[f] = st.number_input(
                        nice(f), min_value=lo, max_value=hi, value=med,
                        step=1.0 if is_int else 0.01, key=f)
        go = st.form_submit_button("✨ Predict Price")

    if go:
        row = pd.DataFrame([values])[features]
        price = max(float(model.predict(row)[0]), 0)
        st.markdown(f"""
        <div class="result">
          <p>Estimated Price</p>
          <h2>$ {price:,.0f}</h2>
          <p>Typical error about ±{m['mape']:.1f}% on unseen cars</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

# ---------------- Tab 2: Performance ----------------
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    for col, title, val in [
        (c1, "R² (TRAIN)", f"{m['r2_train']:.3f}"),
        (c2, "R² (TEST)", f"{m['r2_test']:.3f}"),
        (c3, "RMSE (TEST)", f"{m['rmse']:,.0f}"),
        (c4, "MAPE (TEST)", f"{m['mape']:.1f}%"),
    ]:
        col.markdown(f'<div class="card"><div class="big">{val}</div>'
                     f'<div class="small">{title}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.caption(f"Adjusted R² (train): {m['adj_r2']:.3f}")
    st.subheader("Actual vs Predicted (test data)")
    st.scatter_chart(compare, x="Actual", y="Predicted", color="#6366f1")

# ---------------- Tab 3: Data ----------------
with tab3:
    st.subheader("Raw data preview")
    st.dataframe(raw.head(20), use_container_width=True)
    st.caption(f"{raw.shape[0]} rows × {raw.shape[1]} columns")