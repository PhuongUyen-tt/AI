import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# --- File Paths ---
# Files are in the same directory as app.py in the GitHub repository
model_path = 'gradient_boosting_model.pkl'
features_path = 'gradient_boosting_features.pkl'
label_encoder_target_path = 'product_category_label_encoder.pkl'
data_path = 'data.csv'  # Assumes data.csv is in the repository's root directory

# --- Alternative: Load data.csv from a GitHub raw URL ---
# If data.csv is hosted on GitHub, uncomment and update the URL below
# data_url = 'https://raw.githubusercontent.com/your-username/your-repo/main/data.csv'
# try:
#     original_df = pd.read_csv(data_url)
# except Exception as e:
#     st.error(f"Error loading data from GitHub URL: {e}")
#     st.stop()

# --- Load Data for Dropdowns ---
try:
    original_df = pd.read_csv(data_path)
    unique_store_locations = sorted(original_df['store_location'].dropna().unique().tolist())
    unique_genders = sorted(original_df['Gender'].dropna().unique().tolist())
    unique_seasons = sorted(original_df['Season'].dropna().unique().tolist())
    unique_sizes = sorted(original_df['Size'].dropna().unique().tolist())
except FileNotFoundError:
    st.error(f"Data file 'data.csv' not found in the repository. Ensure it is uploaded.")
    st.stop()
except KeyError as e:
    st.error(f"Missing column in data file: {e}. Expected columns: store_location, Gender, Season, Size.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Load Trained Model and Artifacts ---
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Ensure it is uploaded to the repository.")
    st.stop()
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

if not os.path.exists(features_path):
    st.error(f"Feature columns file '{features_path}' not found. Ensure it is uploaded to the repository.")
    st.stop()
try:
    expected_features = joblib.load(features_path)
except Exception as e:
    st.error(f"Error loading feature columns: {e}")
    st.stop()

if not os.path.exists(label_encoder_target_path):
    st.warning(f"Target LabelEncoder file '{label_encoder_target_path}' not found. Predictions will be numerical.")
    label_encoder_y = None
else:
    try:
        label_encoder_y = joblib.load(label_encoder_target_path)
    except Exception as e:
        st.error(f"Error loading Target LabelEncoder: {e}")
        label_encoder_y = None

# --- Streamlit App Layout ---
st.title("Product Category Prediction")
st.header("Predict Suggested Product Category")

# --- Input Widgets ---
st.subheader("Enter Customer and Transaction Details")

# Categorical inputs with help text
store_location = st.selectbox("Store Location", unique_store_locations, help="Select the store location")
gender = st.selectbox("Gender", unique_genders, help="Select the customer's gender")
season = st.selectbox("Season", unique_seasons, help="Select the season")
size = st.selectbox("Size", unique_sizes, help="Select the product size")

# Numerical inputs with constraints
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1, help="Customer's age")
total_bill = st.number_input("Total Bill", min_value=0.0, value=100.0, step=0.1, help="Total bill amount")
transaction_qty = st.number_input("Transaction Quantity", min_value=1, value=1, step=1, help="Number of items purchased")

# --- Prediction Logic ---
if st.button("Predict Product Category"):
    # Create input dictionary
    input_data = {
        'store_location': store_location,
        'Gender': gender,
        'Season': season,
        'Size': size,
        'Age': age,
        'Total_Bill': total_bill,
        'transaction_qty': transaction_qty
    }
    input_df = pd.DataFrame([input_data])

    # Apply LabelEncoding to categorical features
    categorical_features = ['store_location', 'Gender', 'Season', 'Size']
    for col in categorical_features:
        if col in input_df.columns:
            le = LabelEncoder()
            # Fit LabelEncoder on unique values from original data
            le.fit(original_df[col].dropna().unique())
            input_df[col] = le.transform(input_df[col])

    # Ensure input DataFrame matches expected features
    input_df_processed = input_df.reindex(columns=expected_features, fill_value=0)

    # Make prediction
    try:
        predicted_label = model.predict(input_df_processed)[0]
        if label_encoder_y:
            predicted_category = label_encoder_y.inverse_transform([predicted_label])[0]
            st.success(f"Predicted Product Category: **{predicted_category}**")
        else:
            st.success(f"Predicted Product Category (Numerical Label): **{int(predicted_label)}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Ensure input values are valid and all model files are loaded correctly.")

# --- Instructions ---
st.markdown("---")
st.markdown("**How to Run This App:**")
st.code("1. Save this code as `app.py`.")
st.code("2. Ensure 'data.csv', 'gradient_boosting_model.pkl', 'gradient_boosting_features.pkl', and 'product_category_label_encoder.pkl' are in the same repository.")
st.code("3. Include 'requirements.txt' with dependencies: streamlit, pandas, scikit-learn, joblib.")
st.code("4. Run locally: `streamlit run app.py`")
st.code("5. For Streamlit Cloud, deploy via your GitHub repository with all files.")
st.markdown("Ensure all files are uploaded to your GitHub repository for Streamlit Cloud deployment.")
