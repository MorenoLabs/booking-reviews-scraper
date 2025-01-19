import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Placeholder to store all hotel risk scores
risk_scores = []

def analyze_single_hotel(df: pd.DataFrame, hotel_name: str):
    """Comprehensive analysis of a single hotel"""
    global risk_scores  # Access the global list to store results

    # Filter for hotel data
    hotel_data = df[df['hotel_name'] == hotel_name].copy()
    hotel_data = hotel_data.sort_values('review_post_date')
    
    st.title(f"Comprehensive Risk Analysis: {hotel_name}")
    
    # 1. Basic Hotel Statistics
    st.header("1. Basic Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(hotel_data))
    with col2:
        st.metric("Average Rating", f"{hotel_data['rating'].mean():.2f}")
    with col3:
        st.metric("Rating Std Dev", f"{hotel_data['rating'].std():.2f}")
    with col4:
        st.metric("% Below Q1", f"{(hotel_data['rating'] < hotel_data['rating'].quantile(0.25)).mean()*100:.1f}%")
    
    # 2. Performance Deviation Components
    st.header("2. Performance Deviation Components")
    z_score = (hotel_data['rating'].mean() - df['rating'].mean()) / df['rating'].std()
    perf_deviation = max(0, -z_score * 0.25)
    
    # 3. Guest Type Adaptability
    stay_type_stats = hotel_data.groupby('stay_type').agg({
        'rating': ['count', 'std']
    }).round(2)
    stay_type_variation = stay_type_stats[('rating', 'std')].mean() * 0.15

    # 4. Time-Based Components
    monthly_stats = hotel_data.groupby('Month')['rating'].agg(['std']).round(2)
    time_variation = monthly_stats['std'].mean() * 0.15

    # 5. Stability Component
    stability = (hotel_data['rating'].std() / df['rating'].std()) * 0.10

    # 6. Room Type Component
    room_stats = hotel_data.groupby('room_view')['rating'].agg(['std']).round(2)
    room_variation = room_stats['std'].mean() * 0.10

    # 7. Trend Components
    hotel_data['30d_avg'] = hotel_data['rating'].rolling(30).mean()
    recent_trend = (hotel_data['30d_avg'].iloc[-1] - hotel_data['30d_avg'].iloc[-30]) * 0.10 if len(hotel_data) > 30 else 0

    # Total Risk Score
    total_risk = (perf_deviation + stay_type_variation + time_variation +
                  stability + room_variation + max(0, -recent_trend))
    
    st.header("7. Overall Risk Assessment")
    st.metric("Total Risk Score", f"{total_risk:.3f}")
    
    # Store the result
    risk_scores.append({'hotel_name': hotel_name, 'risk_score': total_risk})

    # Return the total risk score
    return total_risk

# Main Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("Hotel Risk Scoring System")
    
    global risk_scores  # Access the global list
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload review dataset", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['review_post_date'] = pd.to_datetime(df['review_post_date'])
        df = df[df['review_post_date'] < '2025-01-01']  # Filter dates
        
        # Get list of hotels
        hotel_list = df['hotel_name'].unique()
        
        if st.button("Analyze All Hotels"):
            st.write("Calculating risk scores for all hotels...")
            
            for hotel in hotel_list:
                analyze_single_hotel(df, hotel)
            
            # Create a DataFrame of risk scores
            risk_scores_df = pd.DataFrame(risk_scores)
            
            st.success("Analysis complete! Below is the risk score dataset.")
            st.dataframe(risk_scores_df)
            
            # Option to download the dataset
            csv = risk_scores_df.to_csv(index=False)
            st.download_button(label="Download Risk Scores as CSV", 
                               data=csv,
                               file_name='hotel_risk_scores.csv',
                               mime='text/csv')

if __name__ == "__main__":
    main()
