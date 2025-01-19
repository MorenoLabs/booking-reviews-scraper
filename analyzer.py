import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from hotel_risk_analyzer import HotelRiskAnalyzer  # Assuming your updated class is in hotel_risk_analyzer.py

# Load data and initialize analyzer
DATA_PATH = '../booking-reviews-scraper/data/combined_reviews_all.csv'
#extract Month
analyzer = HotelRiskAnalyzer(DATA_PATH)

# Streamlit App Title
st.title("Hotel Risk Analyzer")

# Sidebar for Hotel and Stay Type Selection
hotels = analyzer.df['Hotel Name'].str.lower().unique()
stay_types = analyzer.df['stay_type'].str.lower().unique()

selected_hotel = st.sidebar.selectbox("Select Hotel", hotels)
selected_stay_type = st.sidebar.selectbox("Select Stay Type", stay_types)

# Generate Risk Report for the Selected Hotel and Stay Type
report = analyzer.generate_risk_report(selected_hotel, selected_stay_type)

if report:
    # Display Metrics as Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Final Risk Score", value=f"{report['risk_score']:.2f}")
    col2.metric(label="Average Rating", value=f"{report['metrics']['avg_rating']:.2f}")
    col3.metric(label="Review Count", value=f"{report['metrics']['review_count']}")
    col4.metric(label="Low Rating %", value=f"{report['metrics']['low_rating_pct']:.1f}%")

    col5, col6, col7 = st.columns(3)
    col5.metric(label="Low Rating Impact", value=f"{report['components']['base'].low_rating:.2f}")
    col6.metric(label="Performance Gap", value=f"{report['components']['base'].performance:.2f}")
    col7.metric(label="Volatility", value=f"{report['components']['base'].volatility:.2f}")

    # Plot 1: Volume and Rating Trends
    st.subheader("Monthly Trend: Ratings and Review Volume")
    hotel_data = analyzer.df[
        (analyzer.df['Hotel Name'].str.startswith(selected_hotel)) &
        (analyzer.df['stay_type'] == selected_stay_type)
    ]
    hotel_data['month'] = pd.to_datetime(hotel_data['review_post_date']).dt.to_period('M')

    monthly_data = hotel_data.groupby('month').agg({
        'rating': 'mean',
        'review_post_date': 'count'
    }).rename(columns={'review_post_date': 'volume'}).reset_index()

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Rating (Left Axis)
    ax1.set_xlabel('month')
    ax1.set_ylabel('Average Rating', color='blue')
    ax1.plot(monthly_data['month'].astype(str), monthly_data['rating'], marker='o', label='Average Rating', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 10)
    #add 8.6 target
    ax1.axhline(y=8.6, color='r', linestyle='--', label='Target Rating')

    # Volume (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Review Volume', color='orange')
    ax2.bar(monthly_data['month'].astype(str), monthly_data['volume'], alpha=0.5, label='Review Volume', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title(f"Monthly Trend for {selected_hotel} - {selected_stay_type}")
    st.pyplot(fig1)

    # Plot 2: Monthly Risk Scores for the Last 6 Months
    st.subheader("Monthly Risk Scores for the Last 6 Months")
    risk_scores = analyzer.calculate_monthly_risk_scores(selected_hotel, selected_stay_type)
    if not risk_scores.empty:
        # Filter to the last 6 months
        risk_scores['month'] = pd.to_datetime(risk_scores['month'].astype(str))
        recent_risk_scores = risk_scores.sort_values('month').tail(12)

        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(recent_risk_scores['month'], recent_risk_scores['Risk Score'], marker='o', label='Risk Score', color='red')
        ax.set_xlabel('month')
        ax.set_ylabel('Risk Score')
        ax.set_title(f"Monthly Risk Scores for {selected_hotel} - {selected_stay_type}")
        #set 0 to 100

        ax.legend()
        st.pyplot(fig2)
    else:
        st.warning("Insufficient data to calculate monthly risk scores.")
else:
    st.warning("No sufficient data available for the selected combination of hotel and stay type.")
