import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_single_hotel(df: pd.DataFrame, hotel_name: str):
    """Comprehensive analysis of a single hotel"""
    
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
    
    # 2. Performance Deviation Analysis
    st.header("2. Performance Deviation Components")
    
    # Calculate deviations for stay types
    stay_type_deviation = hotel_data.groupby('stay_type')['rating'].mean() - hotel_data['rating'].mean()
    
    # Display deviation summaries
    st.write("Deviation from Stay Type Averages:")
    st.dataframe(stay_type_deviation)
    
    # Rolling metrics
    hotel_data['7d_avg'] = hotel_data['rating'].rolling(7).mean()
    hotel_data['30d_avg'] = hotel_data['rating'].rolling(30).mean()
    hotel_data['90d_avg'] = hotel_data['rating'].rolling(90).mean()
    
    # Plot rolling averages
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hotel_data['review_post_date'], y=hotel_data['rating'],
                             mode='markers', name='Individual Ratings',
                             marker=dict(size=5, opacity=0.5)))
    fig.add_trace(go.Scatter(x=hotel_data['review_post_date'], y=hotel_data['7d_avg'],
                             name='7-day MA', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=hotel_data['review_post_date'], y=hotel_data['30d_avg'],
                             name='30-day MA', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=hotel_data['review_post_date'], y=hotel_data['90d_avg'],
                             name='90-day MA', line=dict(width=2)))
    st.plotly_chart(fig)
    
    # 3. Guest Type and Origin Analysis
    st.header("3. Guest Type and Origin Adaptability")
    
    # Stay type performance
    stay_type_stats = hotel_data.groupby('stay_type').agg({
        'rating': ['count', 'mean', 'std']
    }).round(2)
    
    # User country performance
    origin_stats = hotel_data.groupby('user_country').agg({
        'rating': ['count', 'mean', 'std']
    }).round(2)
    
    # Display guest type and origin stats
    st.write("Stay Type Performance:")
    st.dataframe(stay_type_stats)
    st.write("Guest Origin Performance:")
    st.dataframe(origin_stats)
    
    # 4. Time-Based Patterns
    st.header("4. Time-Based Components")
    
    # Monthly patterns
    monthly_stats = hotel_data.groupby('Month')['rating'].agg(['mean', 'std', 'count']).round(2)
    
    # Create monthly trend plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['count'],
                         name='Review Count'), secondary_y=True)
    fig.add_trace(go.Scatter(x=monthly_stats.index, y=monthly_stats['mean'],
                             name='Average Rating'), secondary_y=False)
    st.plotly_chart(fig)
    
    # 5. Room Type Performance
    st.header("5. Room Type Performance")
    
    # Room-specific stats
    room_stats = hotel_data.groupby('room_view').agg({
        'rating': ['count', 'mean', 'std']
    }).round(2)
    
    # Room-specific patterns
    room_stay_pivot = pd.pivot_table(hotel_data, 
                                     values='rating',
                                     index='room_view',
                                     columns='stay_type',
                                     aggfunc='mean')
    
    st.write("Room-Specific Performance:")
    st.dataframe(room_stats)
    
    fig = px.imshow(room_stay_pivot,
                    labels=dict(x="Stay Type", y="Room Type", color="Average Rating"),
                    aspect="auto",
                    color_continuous_scale=["red", "yellow", "green"],  # Red to yellow to green
                    color_continuous_midpoint=room_stay_pivot.values.mean())  # Center the color scale
    st.plotly_chart(fig)
    
    # 6. Trend Analysis
    st.header("6. Trend Components")
    
    # Calculate trend acceleration/deceleration
    trend_acceleration = hotel_data['30d_avg'].diff(1).mean()
    st.metric("Trend Acceleration (30-day)", f"{trend_acceleration:.2f}")
    
    # 7. Overall Risk Assessment
    st.header("7. Overall Risk Assessment")
    
    # Performance Deviation (25%)
    z_score = (hotel_data['rating'].mean() - df['rating'].mean()) / df['rating'].std()
    perf_deviation = max(0, -z_score * 0.25)
    
    # Guest Type Adaptability (15%)
    stay_type_variation = stay_type_stats[('rating', 'std')].mean() * 0.15
    origin_variation = origin_stats[('rating', 'std')].mean() * 0.10
    
    # Time Components (15%)
    time_variation = monthly_stats['std'].mean() * 0.15
    
    # Stability Component (10%)
    stability = (hotel_data['rating'].std() / df['rating'].std()) * 0.10
    
    # Room Type Component (10%)
    room_variation = room_stats[('rating', 'std')].mean() * 0.10
    
    # Trend Component (10%)
    recent_trend = trend_acceleration * 0.10
    
    total_risk = (perf_deviation + stay_type_variation + origin_variation +
                  time_variation + stability + room_variation + 
                  max(0, -recent_trend))
    
    # Display risk components
    risk_components = pd.DataFrame({
        'Component': ['Performance Deviation', 'Guest Type Adaptability', 
                     'Time Variation', 'Stability', 
                     'Room Type Variation', 'Trend Acceleration'],
        'Risk Score': [perf_deviation, stay_type_variation + origin_variation, 
                      time_variation, stability, room_variation, max(0, -recent_trend)]
    })
    
    fig = px.bar(risk_components, x='Component', y='Risk Score',
                 title=f"Risk Component Breakdown (Total Risk Score: {total_risk:.3f})")
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide")
    
    uploaded_file = st.file_uploader("Upload review data", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['review_post_date'] = pd.to_datetime(df['review_post_date'])
        df = df[df['review_post_date'] < '2025-01-01']  # Remove January 2025
        
        # Hotel selection
        hotel_name = st.selectbox("Select Hotel", df['hotel_name'].unique())
        
        if st.button("Analyze Hotel"):
            analyze_single_hotel(df, hotel_name)

if __name__ == "__main__":
    main()
