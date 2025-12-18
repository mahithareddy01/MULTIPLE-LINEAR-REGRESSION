import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error

#Page Configuration
st.set_page_config("Multiple Linear Regression",layout="centered")

#load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
st.markdown("""
<div class="card">
            <h1>MULTIPLE LINEAR REGRESSION</h1>
            <p>Predict <b> Tip amount</b> based on <b> Total Bill</b> using Linear Regression...</p>
</div>
            """, unsafe_allow_html=True)

#Load Dataset
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df = load_data()

# Display Dataset Preview
st.markdown('<div class="card2"><b>DATASET PREVIEW:</b></div>', unsafe_allow_html=True)
st.dataframe(df[["total_bill", "tip", "size"]].head())
st.markdown("---")

#Prepare Data
X = df[["total_bill", "size"]]
Y = df["tip"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
#Evaluate Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#Visualization
st.markdown('<div class="card2"><B>PREDICTIONS vs ACTUAL:</B></div>', unsafe_allow_html=True)
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Actual Tip Amount')
plt.ylabel('Predicted Tip Amount')
plt.title('Multiple Linear Regression: Actual vs Predicted Tip Amount')
st.pyplot(plt)
st.markdown("---")

#performace metrics

st.markdown('<div class="card2"><b>PERFORMANCE METRICS:</b></div>', unsafe_allow_html=True)
st.markdown("""
<div class="metric-row">
    <div class="metric-box">
        <span>MSE</span>
        <strong>{:.2f}</strong>
    </div>
    <div class="metric-box">
        <span>RMSE</span>
        <strong>{:.2f}</strong>
    </div>
    <div class="metric-box">
        <span>R² Score</span>
        <strong>{:.2f}</strong>
    </div>
</div>
""".format(mse, rmse, r2), unsafe_allow_html=True)
st.markdown("---")


#intercept and slope
st.markdown('<div class="card2"><b>MODEL COEFFICIENTS:</b></div>', unsafe_allow_html=True)
st.write(f"Intercept: {model.intercept_:.2f}")  
for i, col in enumerate(X.columns):
    st.write(f"Coefficient for {col}: {model.coef_[i]:.2f}")
st.markdown("---")

#prediction
st.markdown('<div class="card2"><b>PREDICT TIP AMOUNT:</b></div>', unsafe_allow_html=True)
total_bill_input = st.slider("Total Bill Amount", float(df["total_bill"].min()), float(df["total_bill"].max()), float(df["total_bill"].mean()))
size = st.slider("Size of group", int(df["size"].min()), int(df["size"].max()), int(df["size"].mean()))
input_scaled = scaler.transform([[total_bill_input, size]])
predicted_tip = model.predict(input_scaled)
st.markdown(f"<div class='prediction-box'>Predicted Tip Amount: <strong>${predicted_tip[0]:.2f}</strong></div>", unsafe_allow_html=True)

st.markdown("---")