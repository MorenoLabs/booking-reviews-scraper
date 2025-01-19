import streamlit as st
import pandas as pd
import numpy as np

# Function: Analyze a single hotel
def analyze_single_hotel(df: pd.DataFrame, hotel_name: str):
    """Comprehensive analysis of a single hotel"""
    # Filter for hotel data
    hotel_data = df[df['name'] == hotel_name].copy()
    hotel_data = hotel_data.sort_values('review_post_date')

    # 1. Basic Hotel Statistics
    total_reviews = len(hotel_data)
    avg_rating = hotel_data['rating'].mean()
    std_rating = hotel_data['rating'].std()
    percent_below_q1 = (hotel_data['rating'] < hotel_data['rating'].quantile(0.25)).mean() * 100

    # 2. Performance Deviation Components
    z_score = (hotel_data['rating'].mean() - df['rating'].mean()) / df['rating'].std()
    perf_deviation = max(0, -z_score * 0.25)

    # 3. Guest Type Adaptability
    stay_type_stats = hotel_data.groupby('stay_type').agg({
        'rating': ['count', 'std']
    }).round(2)
    stay_type_variation = stay_type_stats[('rating', 'std')].mean() * 0.15 if not stay_type_stats.empty else 0

    # 4. Time-Based Components
    hotel_data['Month'] = hotel_data['review_post_date'].dt.month
    monthly_stats = hotel_data.groupby('Month')['rating'].agg(['std']).round(2)
    time_variation = monthly_stats['std'].mean() * 0.15 if not monthly_stats.empty else 0

    # 5. Stability Component
    stability = (hotel_data['rating'].std() / df['rating'].std()) * 0.10 if df['rating'].std() > 0 else 0

    # 6. Room Type Component
    room_stats = hotel_data.groupby('room_view')['rating'].agg(['std']).round(2)
    room_variation = room_stats['std'].mean() * 0.10 if not room_stats.empty else 0

    # 7. Trend Components
    hotel_data['30d_avg'] = hotel_data['rating'].rolling(30).mean()
    recent_trend = (
        (hotel_data['30d_avg'].iloc[-1] - hotel_data['30d_avg'].iloc[-30]) * 0.10
        if len(hotel_data) > 30 else 0
    )

    # Total Risk Score
    # total_risk = (
    #     perf_deviation + stay_type_variation + time_variation +
    #     stability + room_variation + max(0, -recent_trend)
    # )
    
    # Total Risk Score (Weighted)
    # add weights to each component
    total_risk = (
        perf_deviation + stay_type_variation + time_variation +
        stability + room_variation + max(0, -recent_trend)
    ) * 0.25

    # Store the results
    results = {
        'name': hotel_name,
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'std_rating': std_rating,
        'percent_below_q1': percent_below_q1,
        'perf_deviation': perf_deviation,
        'stay_type_variation': stay_type_variation,
        'time_variation': time_variation,
        'stability': stability,
        'room_variation': room_variation,
        'recent_trend': recent_trend,
        'total_risk': total_risk
    }

    return results

# Function: Analyze all hotels
def analyze_all_hotels(df: pd.DataFrame):
    """Analyze all hotels in the dataset"""
    hotel_list = df['name'].unique()
    risk_scores = []

    for hotel in hotel_list:
        results = analyze_single_hotel(df, hotel)
        risk_scores.append(results)

    return pd.DataFrame(risk_scores)

# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("Hotel Risk Analysis")

    uploaded_file = st.file_uploader("Upload your dataset", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['review_post_date'] = pd.to_datetime(df['review_post_date'])

        hotel_list = df['name'].unique()

        st.sidebar.header("Select Hotel")
        selected_hotel = st.sidebar.selectbox("Choose a hotel", options=hotel_list)

        if selected_hotel:
            st.header(f"Analysis for {selected_hotel}")
            hotel_results = analyze_single_hotel(df, selected_hotel)

            st.subheader("Basic Performance Metrics")
            st.metric("Total Reviews", hotel_results['total_reviews'])
            st.metric("Average Rating", f"{hotel_results['avg_rating']:.2f}")
            st.metric("Standard Deviation", f"{hotel_results['std_rating']:.2f}")
            st.metric("% Below Q1", f"{hotel_results['percent_below_q1']:.2f}%")

            st.subheader("Risk Components")
            st.write({
                "Performance Deviation": f"{hotel_results['perf_deviation']:.3f}",
                "Stay Type Variation": f"{hotel_results['stay_type_variation']:.3f}",
                "Time Variation": f"{hotel_results['time_variation']:.3f}",
                "Stability": f"{hotel_results['stability']:.3f}",
                "Room Variation": f"{hotel_results['room_variation']:.3f}",
                "Recent Trend": f"{hotel_results['recent_trend']:.3f}",
                "Total Risk Score": f"{hotel_results['total_risk']:.3f}"
            })

        if st.sidebar.button("Analyze All Hotels"):
            st.write("Analyzing all hotels...")
            risk_scores_df = analyze_all_hotels(df)

            st.success("Analysis complete!")
            st.dataframe(risk_scores_df)

            # Option to download results
            csv = risk_scores_df.to_csv(index=False)
            st.download_button(label="Download Results as CSV", data=csv, file_name="hotel_risk_scores.csv", mime="text/csv")

if __name__ == "__main__":
    main()
