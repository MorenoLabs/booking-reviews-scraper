import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def calculate_risk_benchmark(df: pd.DataFrame, analyzed_hotel: str):
    """
    Calculate benchmarks and categorize risk scores.
    :param df: DataFrame containing 'hotel_name' and 'risk_score' columns.
    :param analyzed_hotel: The name of the hotel being analyzed.
    """
    # 1. Calculate statistics
    mean_risk = df['risk_score'].mean()
    std_dev_risk = df['risk_score'].std()
    low_threshold = mean_risk - std_dev_risk
    high_threshold = mean_risk + std_dev_risk
    
    # 2. Categorize risk scores
    def categorize_risk(score):
        if score <= low_threshold:
            return 'Low Risk'
        elif low_threshold < score <= high_threshold:
            return 'Moderate Risk'
        else:
            return 'High Risk'
    
    df['risk_category'] = df['risk_score'].apply(categorize_risk)
    
    # 3. Get analyzed hotel's risk category
    analyzed_score = df.loc[df['hotel_name'] == analyzed_hotel, 'risk_score'].values[0]
    analyzed_category = df.loc[df['hotel_name'] == analyzed_hotel, 'risk_category'].values[0]
    
    # 4. Display statistics and thresholds
    st.write(f"**Risk Score Analysis for {analyzed_hotel}:**")
    st.write(f"Total Risk Score: {analyzed_score:.3f}")
    st.write(f"Risk Category: {analyzed_category}")
    st.write("**Benchmark Statistics:**")
    st.metric("Mean Risk Score", f"{mean_risk:.3f}")
    st.metric("Standard Deviation", f"{std_dev_risk:.3f}")
    st.metric("Low Risk Threshold", f"≤ {low_threshold:.3f}")
    st.metric("High Risk Threshold", f"> {high_threshold:.3f}")
    
    # 5. Visualize risk score distribution
    fig = px.histogram(df, x='risk_score', nbins=20, color='risk_category',
                       title="Risk Score Distribution",
                       labels={'risk_score': 'Risk Score'},
                       marginal="rug",  # Add a rug plot
                       color_discrete_map={'Low Risk': 'green', 'Moderate Risk': 'orange', 'High Risk': 'red'})
    
    # Highlight the analyzed hotel
    fig.add_vline(x=analyzed_score, line_width=3, line_dash="dash", line_color="blue",
                  annotation_text=f"{analyzed_hotel} ({analyzed_score:.3f})", annotation_position="top left")
    
    st.plotly_chart(fig)

# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("Hotel Risk Score Benchmarking")
    
    # Upload the dataset
    uploaded_file = st.file_uploader("Upload dataset with risk scores", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Ensure required columns exist
        if 'hotel_name' not in df.columns or 'risk_score' not in df.columns:
            st.error("Dataset must contain 'hotel_name' and 'risk_score' columns.")
            return
        
        # Hotel selection
        hotel_name = st.selectbox("Select Hotel to Analyze", df['hotel_name'].unique())
        
        if st.button("Analyze Hotel"):
            calculate_risk_benchmark(df, hotel_name)

if __name__ == "__main__":
    main()
