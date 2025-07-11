import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="🌱 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# 1. Load Dataset with Error Handling
# ------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("crop_data.csv")
        return df
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        st.warning("⚠️ crop_data.csv not found. Using sample data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        sample_data = {
            'Fertilizer': np.random.uniform(50, 300, n_samples),
            'temp': np.random.uniform(15, 35, n_samples),
            'N': np.random.uniform(20, 150, n_samples),
            'P': np.random.uniform(10, 80, n_samples),
            'K': np.random.uniform(15, 100, n_samples)
        }
        # Create realistic yield based on inputs
        sample_data['yeild'] = (
            sample_data['Fertilizer'] * 0.02 +
            sample_data['temp'] * 0.3 +
            sample_data['N'] * 0.05 +
            sample_data['P'] * 0.08 +
            sample_data['K'] * 0.04 +
            np.random.normal(0, 2, n_samples)
        )
        return pd.DataFrame(sample_data)

# ------------------------
# 2. Model Training with Enhanced Metrics
# ------------------------
model_path = "yield_model.pkl"

@st.cache_resource
def train_model():
    df_clean = df.dropna()
    X = df_clean[["Fertilizer", "temp", "N", "P", "K"]]
    y = df_clean["yeild"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save model
    joblib.dump(model, model_path)
    
    return model, mse, r2, feature_importance, X_test, y_test, y_pred

# Load data
df = load_data()

# Prepare data for training/evaluation
df_clean = df.dropna()
X = df_clean[["Fertilizer", "temp", "N", "P", "K"]]
y = df_clean["yeild"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train or load model
if os.path.exists(model_path):
    model = joblib.load(model_path)
    # Calculate metrics for display
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    model, mse, r2, feature_importance, X_test, y_test, y_pred = train_model()

# ------------------------
# 3. Sidebar Configuration
# ------------------------
st.sidebar.header("🎛️ Control Panel")

# Model performance metrics in sidebar
st.sidebar.subheader("📊 Model Performance")
st.sidebar.metric("R² Score", f"{r2:.3f}")
st.sidebar.metric("MSE", f"{mse:.2f}")
st.sidebar.metric("RMSE", f"{np.sqrt(mse):.2f}")

# Dataset info
st.sidebar.subheader("📋 Dataset Info")
st.sidebar.write(f"Total Records: {len(df)}")
st.sidebar.write(f"Features: {len(df.columns)-1}")
st.sidebar.write(f"Missing Values: {df.isnull().sum().sum()}")

# ------------------------
# 4. Main App Layout
# ------------------------
st.markdown('<h1 class="main-header">🌱 Advanced Crop Yield Predictor</h1>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Data Analysis", "🔍 Model Insights", "📈 Batch Prediction"])

# ------------------------
# Tab 1: Prediction Interface
# ------------------------
with tab1:
    st.header("Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Parameters")
        
        # Input fields with realistic ranges and descriptions
        fertilizer = st.slider(
            "Fertilizer Used (kg/ha)",
            min_value=0.0, max_value=500.0, value=150.0, step=5.0,
            help="Amount of fertilizer applied per hectare"
        )
        
        temp = st.slider(
            "Temperature (°C)",
            min_value=0.0, max_value=50.0, value=25.0, step=0.5,
            help="Average temperature during growing season"
        )
        
        N = st.slider(
            "Nitrogen (N)",
            min_value=0.0, max_value=200.0, value=80.0, step=5.0,
            help="Nitrogen content in soil"
        )
        
        P = st.slider(
            "Phosphorus (P)",
            min_value=0.0, max_value=200.0, value=40.0, step=5.0,
            help="Phosphorus content in soil"
        )
        
        K = st.slider(
            "Potassium (K)",
            min_value=0.0, max_value=200.0, value=60.0, step=5.0,
            help="Potassium content in soil"
        )
        
        # Preset buttons
        st.subheader("🌟 Quick Presets")
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        
        with col_preset1:
            if st.button("🌱 Low Input", use_container_width=True):
                fertilizer = 50.0
                temp = 20.0
                N = 30.0
                P = 15.0
                K = 25.0
                st.rerun()
        
        with col_preset2:
            if st.button("🌾 Medium Input", use_container_width=True):
                fertilizer = 150.0
                temp = 25.0
                N = 80.0
                P = 40.0
                K = 60.0
                st.rerun()
        
        with col_preset3:
            if st.button("🌽 High Input", use_container_width=True):
                fertilizer = 300.0
                temp = 30.0
                N = 150.0
                P = 75.0
                K = 100.0
                st.rerun()
        
        # Main prediction button
        st.subheader("🔮 Make Prediction")
        predict_button = st.button("🌾 Predict Crop Yield", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Only show results when predict button is clicked
        if predict_button:
            # Make prediction
            input_data = np.array([[fertilizer, temp, N, P, K]])
            prediction = model.predict(input_data)[0]
            
            # Calculate prediction confidence using model's built-in uncertainty
            # For Random Forest, we can use the standard deviation across trees
            tree_predictions = []
            for estimator in model.estimators_:
                tree_predictions.append(estimator.predict(input_data)[0])
            
            confidence_interval = np.std(tree_predictions)
            
            # Alternative: Use a simpler confidence measure based on feature ranges
            if confidence_interval == 0:  # Fallback if all trees give same prediction
                confidence_interval = prediction * 0.05  # 5% of prediction as uncertainty
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🌾 Predicted Crop Yield</h3>
                <h2>{prediction:.2f} ± {confidence_interval:.2f} tons/hectare</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance indicators
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                performance = "High" if prediction > df['yeild'].quantile(0.75) else "Medium" if prediction > df['yeild'].quantile(0.25) else "Low"
                st.metric("Performance", performance)
            
            with col_metric2:
                percentile = (prediction > df['yeild']).mean() * 100
                st.metric("Percentile", f"{percentile:.1f}%")
            
            with col_metric3:
                roi = (prediction * 1000 - fertilizer * 2) / (fertilizer * 2) * 100  # Simplified ROI
                st.metric("Est. ROI", f"{roi:.1f}%")
            
            # Recommendation system
            st.subheader("💡 Recommendations")
            
            recommendations = []
            if fertilizer < 100:
                recommendations.append("Consider increasing fertilizer application")
            if temp < 20:
                recommendations.append("Temperature might be too low for optimal growth")
            if N < 50:
                recommendations.append("Nitrogen levels could be increased")
            if P < 30:
                recommendations.append("Consider phosphorus supplementation")
            if K < 40:
                recommendations.append("Potassium levels might need improvement")
            
            if recommendations:
                for rec in recommendations:
                    st.info(f"💡 {rec}")
            else:
                st.success("✅ Your inputs look well-balanced for good yield!")
            
            # Show input summary
            st.subheader("📋 Input Summary")
            input_summary = pd.DataFrame({
                'Parameter': ['Fertilizer (kg/ha)', 'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
                'Value': [fertilizer, temp, N, P, K]
            })
            st.table(input_summary)
            
        else:
            # Show placeholder when no prediction is made
            st.info("👆 Adjust the input parameters and click 'Predict Crop Yield' to see results!")
            
            # Show input ranges from dataset
            st.subheader("📊 Dataset Ranges (for reference)")
            ranges_df = pd.DataFrame({
                'Parameter': ['Fertilizer', 'Temperature', 'Nitrogen', 'Phosphorus', 'Potassium'],
                'Min': [df['Fertilizer'].min(), df['temp'].min(), df['N'].min(), df['P'].min(), df['K'].min()],
                'Max': [df['Fertilizer'].max(), df['temp'].max(), df['N'].max(), df['P'].max(), df['K'].max()],
                'Average': [df['Fertilizer'].mean(), df['temp'].mean(), df['N'].mean(), df['P'].mean(), df['K'].mean()]
            })
            ranges_df = ranges_df.round(2)
            st.table(ranges_df)

# ------------------------
# Tab 2: Data Analysis
# ------------------------
with tab2:
    st.header("📊 Data Analysis Dashboard")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dataset Overview")
        st.write(df.describe())
    
    with col2:
        st.subheader("🔍 Data Distribution")
        fig = px.histogram(df, x="yeild", nbins=30, 
                          title="Distribution of Crop Yields")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlation Matrix")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, 
                    labels=dict(color="Correlation"),
                    title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.subheader("📈 Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("Select X-axis feature", df.columns[:-1])
        feature_y = st.selectbox("Select Y-axis feature", df.columns[:-1], index=1)
    
    with col2:
        color_by = st.selectbox("Color by", ["yeild"] + list(df.columns[:-1]))
    
    fig = px.scatter(df, x=feature_x, y=feature_y, color=color_by,
                     title=f"{feature_x} vs {feature_y}")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab 3: Model Insights
# ------------------------
with tab3:
    st.header("🔍 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Feature Importance")
        fig = px.bar(feature_importance, x='importance', y='feature',
                     orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Model Performance")
        
        # Actual vs Predicted scatter plot
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual Yield', 'y': 'Predicted Yield'},
                        title="Actual vs Predicted Yield")
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual analysis
    st.subheader("📉 Residual Analysis")
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(residuals, nbins=30, title="Distribution of Residuals")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(x=y_pred, y=residuals, 
                        labels={'x': 'Predicted Yield', 'y': 'Residuals'},
                        title="Residuals vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab 4: Batch Prediction
# ------------------------
with tab4:
    st.header("📈 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with columns: `Fertilizer`, `temp`, `N`, `P`, `K` 
    or use the sample data generator below.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.subheader("📁 Uploaded Data")
        st.write(batch_data.head())
        
        # Make predictions
        if st.button("🔮 Make Batch Predictions"):
            predictions = model.predict(batch_data[["Fertilizer", "temp", "N", "P", "K"]])
            batch_data['Predicted_Yield'] = predictions
            
            st.subheader("📊 Batch Prediction Results")
            st.write(batch_data)
            
            # Download results
            csv = batch_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.subheader("🎲 Generate Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.slider("Number of samples", 10, 100, 20)
            
        with col2:
            if st.button("🎯 Generate & Predict"):
                # Generate sample data
                sample_data = pd.DataFrame({
                    'Fertilizer': np.random.uniform(50, 300, n_samples),
                    'temp': np.random.uniform(15, 35, n_samples),
                    'N': np.random.uniform(20, 150, n_samples),
                    'P': np.random.uniform(10, 80, n_samples),
                    'K': np.random.uniform(15, 100, n_samples)
                })
                
                # Make predictions
                predictions = model.predict(sample_data)
                sample_data['Predicted_Yield'] = predictions
                
                st.subheader("📊 Sample Predictions")
                st.write(sample_data)
                
                # Visualization
                fig = px.scatter(sample_data, x='Fertilizer', y='Predicted_Yield',
                               size='temp', color='N',
                               title="Sample Predictions Visualization")
                st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌱 Advanced Crop Yield Predictor | Built with Streamlit & Machine Learning</p>
    <p>For best results, ensure your input values are within realistic farming ranges</p>
</div>
""", unsafe_allow_html=True)