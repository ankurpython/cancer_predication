import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered",
    page_icon="🔬"
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.big-title {
    font-size:2.5rem;
    font-weight:bold;
    background: -webkit-linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: bold;
    border-radius:10px;
    height:50px;
    width:100%;
}
.stButton>button:hover {
    box-shadow: 0 10px 20px rgba(102,126,234,0.3);
}
input {
    height:40px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>🔬 Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.write("Enter the 5 key features below to get a prediction.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, step=0.001, value=17.99)
        area_mean = st.number_input("Area Mean", min_value=0.0, step=0.001, value=1001.0)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, step=0.001, value=0.1471)
    with col2:
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, step=0.001, value=122.8)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, step=0.001, value=0.3001)



@st.cache_data
def train_model():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/refs/heads/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    features = ["radius_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave points_mean"]
    X = df[features]
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


model = train_model()


if st.button("Predict"):
    input_data = [[radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean]]
    new_df = pd.DataFrame(input_data, columns=["radius_mean", "perimeter_mean", "area_mean", "concavity_mean",
                                               "concave points_mean"])

    prediction = model.predict(new_df)[0]
    probability = model.predict_proba(new_df)[0]

    if prediction == 1:
        st.markdown(f"<div style='background-color:#ffcccc;padding:15px;border-radius:10px;'>"
                    f"<h3>❌ The Patient is diagnosed with Cancer</h3>"
                    f"<p>Confidence: {probability[1] * 100:.2f}%</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#ccffcc;padding:15px;border-radius:10px;'>"
                    f"<h3>✅ The Patient is not diagnosed with Cancer</h3>"
                    f"<p>Confidence: {probability[0] * 100:.2f}%</p></div>", unsafe_allow_html=True)


st.subheader("Entered Patient Data:")
st.table(new_df)
