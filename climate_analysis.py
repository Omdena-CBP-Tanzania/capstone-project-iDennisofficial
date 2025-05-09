import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Set Streamlit config
st.set_page_config(page_title='Climate Change in Tanzania', layout='wide')

# Load data (from user-uploaded file or default CSV)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('data/tanzania_climate_data.csv')

# Preprocessing
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df['Season'] = df['Month'].apply(lambda m: 'Summer' if m in [12, 1, 2]
                                 else 'Autumn' if m in [3, 4, 5]
                                 else 'Winter' if m in [6, 7, 8]
                                 else 'Spring')

# Handle missing values only for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Sidebar — user inputs
st.sidebar.title('Model Input Parameters')
year = st.sidebar.slider('Year', int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].mean()))
month = st.sidebar.slider('Month', 1, 12, 6)
total_rainfall = st.sidebar.slider('Total Rainfall (mm)', float(df['Total_Rainfall_mm'].min()), 
                                   float(df['Total_Rainfall_mm'].max()), float(df['Total_Rainfall_mm'].mean()))
max_temp = st.sidebar.slider('Max Temperature (°C)', float(df['Max_Temperature_C'].min()),
                             float(df['Max_Temperature_C'].max()), float(df['Max_Temperature_C'].mean()))
min_temp = st.sidebar.slider('Min Temperature (°C)', float(df['Min_Temperature_C'].min()),
                             float(df['Min_Temperature_C'].max()), float(df['Min_Temperature_C'].mean()))

# Model selection
model_choice = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Random Forest'])

if model_choice == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of Trees', 10, 200, 100)
    max_depth = st.sidebar.slider('Max Depth', 1, 20, 5)

if st.sidebar.button('Train Model'):
    if model_choice == 'Linear Regression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Split the data
    X = df[['Year', 'Month', 'Total_Rainfall_mm', 'Max_Temperature_C', 'Min_Temperature_C']]
    y = df['Average_Temperature_C']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    st.success(f'{model_choice} trained! RMSE: {rmse:.2f}, MAE: {mae:.2f}')

    # Save model after training
    joblib.dump(model, f'{model_choice.lower().replace(" ", "_")}_model.pkl')

# Title and overview
st.title('Climate Change Analysis in Tanzania')

st.markdown("""
This interactive dashboard analyzes **historical climate data** in Tanzania and predicts future patterns using machine learning.
""")

# --- 1️Exploratory Data Analysis (EDA) ---
st.header(' Exploratory Data Analysis')

# Descriptive stats
st.subheader('Statistical Summary')
st.dataframe(df.drop(columns=['Year', 'Month', 'Date']).describe())

# Line plots in two columns
st.subheader('Temperature & Rainfall Trends Over Time')
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df, x='Date', y='Average_Temperature_C', color='orange', ax=ax1)
    ax1.set_title('Average Temperature Over Time')
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=df, x='Date', y='Total_Rainfall_mm', color='blue', ax=ax2)
    ax2.set_title('Total Rainfall Over Time')
    st.pyplot(fig2)

# Boxplot and heatmap in two columns
col3, col4 = st.columns(2)
with col3:
    st.subheader('Seasonal Temperature Variation')
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Season', y='Average_Temperature_C', palette='Set2', ax=ax3)
    ax3.set_title('Seasonal Temperature Distribution')
    st.pyplot(fig3)
with col4:
    st.subheader('Correlation Heatmap')
    corr = df[['Average_Temperature_C', 'Total_Rainfall_mm', 'Max_Temperature_C', 'Min_Temperature_C']].corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title('Variable Correlations')
    st.pyplot(fig4)

# Seasonal decomposition
st.subheader('Seasonal Decomposition: Average Temperature')
df.set_index('Date', inplace=True)
result = seasonal_decompose(df['Average_Temperature_C'], model='additive', period=12)
fig5 = result.plot()
st.pyplot(fig5.figure)
df.reset_index(inplace=True)

# --- Machine Learning Modeling ---
st.header('Machine Learning Modeling')

# Features and target
X = df[['Year', 'Month', 'Total_Rainfall_mm', 'Max_Temperature_C', 'Min_Temperature_C']]
y = df['Average_Temperature_C']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# Evaluate
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

lr_rmse, lr_mae = evaluate(y_test, lr_preds)
rf_rmse, rf_mae = evaluate(y_test, rf_preds)

# Display metrics side by side
st.subheader('Model Performance Metrics')
col5, col6 = st.columns(2)
with col5:
    st.write(f"**Linear Regression** → RMSE: {lr_rmse:.2f}, MAE: {lr_mae:.2f}")
with col6:
    st.write(f"**Random Forest** → RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")

# --- Predict Future Climate ---
st.header('Predict Future Climate Conditions')

input_df = pd.DataFrame([[year, month, total_rainfall, max_temp, min_temp]],
                        columns=['Year', 'Month', 'Total_Rainfall_mm', 'Max_Temperature_C', 'Min_Temperature_C'])

lr_pred = lr.predict(input_df)[0]
rf_pred = rf.predict(input_df)[0]

col7, col8 = st.columns(2)
with col7:
    st.write(f"🔸 **Linear Regression Prediction:** {lr_pred:.2f} °C average temperature")
with col8:
    st.write(f"🔸 **Random Forest Prediction:** {rf_pred:.2f} °C average temperature")

# Footer
st.markdown('---')
