import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
 
# Load saved model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("your_dataset.csv", sep=";")  # FIXED: semicolon separator
    return df

df = load_data()

# Column names used in the model
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Title
st.title("🫀 Heart Disease Prediction App")
st.markdown("Predict cardiovascular disease risk based on patient input.")

# Sidebar input form
st.sidebar.header("📝 Enter Patient Info")

user_input = {}
for col in features:
    if col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
        user_input[col] = st.sidebar.selectbox(f"{col.capitalize()}", [0, 1])
    else:
        user_input[col] = st.sidebar.number_input(f"{col.capitalize()}", step=1)

input_df = pd.DataFrame([user_input])

# Scale user input
scaler = StandardScaler()
scaler.fit(df[features])
scaled_input = scaler.transform(input_df)

# Predict
if st.sidebar.button("🔍 Predict"):
    pred = model.predict(scaled_input)
    st.subheader("Prediction Result")
    st.success("✅ At Risk of Heart Disease" if pred[0] == 1 else "🟢 No Heart Disease Detected")

# Show dataset column names for debug (optional)
# st.write("✅ Dataset Columns:", df.columns.tolist())

# Visualization section
st.header("📊 Visual Data Insights")

# 1. Target Distribution
st.subheader("Distribution of Cardiovascular Disease")
fig1, ax1 = plt.subplots()
sns.countplot(x='cardio', data=df, ax=ax1)
ax1.set_title("Target: Cardiovascular Disease")
st.pyplot(fig1)

# 2. Age vs Cardio
st.subheader("Age vs Cardiovascular Risk")
fig2, ax2 = plt.subplots()
sns.boxplot(x='cardio', y='age', data=df, ax=ax2)
ax2.set_title("Age Distribution by Heart Disease Risk")
st.pyplot(fig2)

# 3. Cholesterol vs Cardio
st.subheader("Cholesterol Levels vs Disease")
fig3, ax3 = plt.subplots()
sns.countplot(x='cholesterol', hue='cardio', data=df, ax=ax3)
ax3.set_title("Cholesterol vs Cardiovascular Disease")
st.pyplot(fig3)

# 4. Correlation Matrix
st.subheader("Feature Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
ax4.set_title("Correlation Matrix")
st.pyplot(fig4)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")
