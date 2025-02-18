import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data():
    """Load risk scores and raw ratings data"""
    risk_df = pd.read_csv("../booking-reviews-scraper/notebooks/hotel_risk_scores.csv")
    raw_df = pd.read_csv("../booking-reviews-scraper/data/combined_reviews_6m2.csv")
    raw_df['date'] = pd.to_datetime(raw_df['review_post_date'])
    raw_df['combined_name'] = raw_df['name'] + " - " + raw_df['hotel_name']
    raw_risk_df = pd.read_csv("../booking-reviews-scraper/notebooks/risk_bookings_v1.csv")
    
    return risk_df, raw_df, raw_risk_df

def create_badge(text, color):
        return f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: bold;
            display: inline-block;
        ">
            {text}
        </span>
        """

def get_disliked_texts_by_hotel(raw_df, hotel_name):
    """Get disliked texts for a specific hotel"""
    # Filter for selected hotel
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['review_text_disliked'])
    #only rows with review_text_disliked longer than 20 characters
    hotel_ratings = hotel_ratings[hotel_ratings['review_text_disliked'].str.len() > 20]
    
    # Get top disliked texts
    disliked_texts = hotel_ratings['review_text_disliked'].value_counts().reset_index()
    
    #make dataframe a list
    disliked_texts = disliked_texts.values.tolist()
    
    # only rows
    
    return disliked_texts

def create_heatmap_guest_origin_room_type(raw_df, hotel_name):
    """Create heatmap for average rating by guest origin and room type"""
    # Get hotel-specific data
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['user_country', 'room_view'])
    
    # Get list of countries that actually have reviews
    countries_with_reviews = hotel_ratings['user_country'].unique()
    
    # Filter to only include those countries
    hotel_ratings = hotel_ratings[hotel_ratings['user_country'].isin(countries_with_reviews)]
    
    # Calculate average ratings and counts
    avg_ratings = hotel_ratings.groupby(['user_country', 'room_view'])['rating'].agg(['mean', 'count']).reset_index()
    
    # Create pivot tables for both mean and count
    pivot_means = avg_ratings.pivot(
        index='room_view',
        columns='user_country',
        values='mean'
    )
    
    pivot_counts = avg_ratings.pivot(
        index='room_view',
        columns='user_country',
        values='count'
    )
    
    # Create custom text for hover annotations
    hover_text = [[f"Avg: {mean:.1f}<br>Count: {count}" if pd.notnull(mean) else ""
                  for mean, count in zip(pivot_means.iloc[i], pivot_counts.iloc[i])]
                 for i in range(len(pivot_means))]
    
    fig = px.imshow(
        pivot_means,
        color_continuous_scale='YlGn',
        aspect='auto',
        title='Average Rating by Guest Origin and Room Type',
        text_auto=True
    )
    
    # Update hover template to show both average and count
    fig.update_traces(
        text=hover_text,
        hovertemplate="Country: %{x}<br>Room: %{y}<br>%{text}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="Guest Origin Country",
        yaxis_title="Room Type",
        xaxis={'tickangle': 45},
        font=dict(size=14),
    )
    
    return fig
def show_average_rating_by_stay_type(raw_df, hotel_name):
    col1, col2 = st.columns(2)
    """Create heatmap for average rating by stay_type and room type"""
    # Get hotel-specific data
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['stay_type', 'room_view'])
    
    # Get list of countries that actually have reviews
    stay_type_with_reviews = hotel_ratings['stay_type'].unique()
    
    # Filter to only include those countries
    hotel_ratings = hotel_ratings[hotel_ratings['stay_type'].isin(stay_type_with_reviews)]
    
    # Calculate average ratings and counts
    avg_ratings = round(hotel_ratings.groupby(['stay_type'])['rating'].agg(['mean', 'count']).reset_index(),1)
    avg_ratings['weighted_avg'] = avg_ratings['mean'] * avg_ratings['count']
    avg_ratings['weighted_avg'] = avg_ratings['weighted_avg'] / avg_ratings['count'].sum()
    #sort by weighted_avg
    avg_ratings = avg_ratings.sort_values(by='weighted_avg', ascending=False)
    # copy avg_ratings and only show average per stay_type
    col1.write(avg_ratings)
    #add pie chart for stay_type
    fig = px.pie(avg_ratings, values='count', names='stay_type', title='Stay Type Distribution')
    col2.plotly_chart(fig)
    return avg_ratings

def show_average_rating_by_room_type(raw_df, hotel_name):
    """Create heatmap for average rating by stay_type and room type"""
    col1, col2 = st.columns(2)
    # Get hotel-specific data
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['room_view'])
    
    # Get list of countries that actually have reviews
    room_type_with_reviews = hotel_ratings['room_view'].unique()
    
    # Filter to only include those countries
    hotel_ratings = hotel_ratings[hotel_ratings['room_view'].isin(room_type_with_reviews)]

    
    # Calculate average ratings and counts
    avg_ratings = round(hotel_ratings.groupby(['room_view'])['rating'].agg(['mean', 'count']).reset_index(),1)
    avg_ratings['weighted_avg'] = avg_ratings['mean'] * avg_ratings['count']
    avg_ratings['weighted_avg'] = avg_ratings['weighted_avg'] / avg_ratings['count'].sum()
    #sort by weighted_avg
    avg_ratings = avg_ratings.sort_values(by='weighted_avg', ascending=False)
    # copy avg_ratings and only show average per stay_type
    col1.write(avg_ratings)
    #assign lowest mean rating to global variable

    lowest_mean_rating = avg_ratings['mean'].min()
    #get the corresponding room type
    global lowest_mean_room_type
    lowest_mean_room_type = avg_ratings[avg_ratings['mean'] == lowest_mean_rating]['room_view'].values[0]
    st.sidebar.write(lowest_mean_room_type)
    # add pie chart for room_type
    fig = px.pie(avg_ratings, values='count', names='room_view', title='Room Type Distribution')
    col2.plotly_chart(fig)
    return avg_ratings

def create_heatmap_stay_type_room_type(raw_df, hotel_name):
    """Create heatmap for average rating by stay_type and room type"""
    # Get hotel-specific data
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['stay_type', 'room_view'])
    
    # Get list of countries that actually have reviews
    countries_with_reviews = hotel_ratings['stay_type'].unique()
    
    # Filter to only include those countries
    hotel_ratings = hotel_ratings[hotel_ratings['stay_type'].isin(countries_with_reviews)]
    
    # Calculate average ratings and counts
    avg_ratings = round(hotel_ratings.groupby(['stay_type', 'room_view'])['rating'].agg(['mean', 'count']).reset_index(),1)
    # copy avg_ratings and only show average per stay_type
  
    
    # Create pivot tables for both mean and count
    pivot_means = avg_ratings.pivot(
        index='room_view',
        columns='stay_type',
        values='mean'
    )
    
    pivot_counts = avg_ratings.pivot(
        index='room_view',
        columns='stay_type',
        values='count'
    )
    
    # Create custom text for hover annotations
    hover_text = [[f"Avg: {mean:.1f}<br>Count: {count}" if pd.notnull(mean) else ""
                  for mean, count in zip(pivot_means.iloc[i], pivot_counts.iloc[i])]
                 for i in range(len(pivot_means))]
    
    fig = px.imshow(
        pivot_means,
        color_continuous_scale='YlGn',
        aspect='auto',
        title='Average Rating by stay_type and Room Type',
        text_auto=True
    )
    
    # Update hover template to show both average and count
    fig.update_traces(
        text=hover_text,
        hovertemplate="stay_type: %{x}<br>Room: %{y}<br>%{text}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="stay_type",
        yaxis_title="Room Type",
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_treemap_stay_type_room_type(raw_df, hotel_name):
    """Create treemap for total ratings count by stay_type and room type, with color by average rating"""
    # Get hotel-specific data
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['stay_type', 'room_view'])
    
    # Aggregate the data to calculate counts and average ratings
    aggregated_data = hotel_ratings.groupby(['stay_type', 'room_view']).agg(
        count=('rating', 'size'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # Create treemap
    fig = px.treemap(
        aggregated_data,
        path=['stay_type', 'room_view'],  # Hierarchical structure
        values='count',  # Size of boxes based on total ratings count
        color='avg_rating',  # Color based on average rating
        color_continuous_scale='YlGn',
        title='Total Ratings Count and Average Rating by Stay Type and Room Type'
    )
    
    fig.update_traces(
        hovertemplate=(
            "Stay Type: %{label}<br>"
            "Room View: %{parent}<br>"
            "Total Ratings: %{value}<br>"
            "Average Rating: %{color:.2f}<extra></extra>"
        )
    )
    
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)  # Adjust margins for better display
    )
    
    return fig

def create_treemap_guest_origin_room_type(raw_df, hotel_name):
    '''Create treemap for total ratings count by guest origin and room type, with color by average rating'''
    # Get hotel-specific data
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Filter out null values
    hotel_ratings = hotel_ratings.dropna(subset=['user_country', 'room_view'])
    
    # Aggregate the data to calculate counts and average ratings
    aggregated_data = hotel_ratings.groupby(['user_country', 'room_view']).agg(
        count=('rating', 'size'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # Create treemap
    fig = px.treemap(
        aggregated_data,
        path=['user_country', 'room_view'],  # Hierarchical structure
        values='count',  # Size of boxes based on total ratings count
        color='avg_rating',  # Color based on average rating
        color_continuous_scale='YlGn',
        title='Total Ratings Count and Average Rating by Guest Origin and Room Type'
    )
    
    fig.update_traces(
        hovertemplate=(
            "user_country: %{label}<br>"
            "Room View: %{parent}<br>"
            "Total Ratings: %{value}<br>"
            "Average Rating: %{color:.2f}<extra></extra>"
        )
    )
    
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)  # Adjust margins for better display
    )
    
    return fig






def get_delta_color(label):
    """Return appropriate delta color based on label"""
    label = str(label).lower()
    if 'high' in label:
        return "inverse"  # Will show as red
    elif 'low' in label:
        return "normal"   # Will show as green
    elif 'medium' in label or 'mid' in label:
        return "off"      # Will show as gray/neutral
    return "off"

def create_risk_distribution(df, hotel_data):
    """Create colored risk distribution histogram"""
    
    # Calculate percentile boundaries
    low_bound = df['normalized_risk_score'].quantile(0.25)
    high_bound = df['normalized_risk_score'].quantile(0.75)
    
    fig = go.Figure()
    
    # Add colored bars for each risk zone
    # Low Risk (Green)
    fig.add_trace(go.Histogram(
        x=df[df['normalized_risk_score'] <= low_bound]['normalized_risk_score'],
        name='Low Risk',
        nbinsx=10,
        marker_color='rgba(75, 192, 192, 0.7)',
        opacity=0.7
    ))
    
    # Medium Risk (Yellow)
    fig.add_trace(go.Histogram(
        x=df[(df['normalized_risk_score'] > low_bound) & 
            (df['normalized_risk_score'] <= high_bound)]['normalized_risk_score'],
        name='Medium Risk',
        nbinsx=10,
        marker_color='rgba(255, 206, 86, 0.7)',
        opacity=0.7
    ))
    
    # High Risk (Red)
    fig.add_trace(go.Histogram(
        x=df[df['normalized_risk_score'] > high_bound]['normalized_risk_score'],
        name='High Risk',
        nbinsx=10,
        marker_color='rgba(255, 99, 132, 0.7)',
        opacity=0.7
    ))
    
    # Add vertical line for current hotel
    fig.add_vline(
        x=hotel_data['normalized_risk_score'],
        line_dash="dash",
        line_color="black",
        annotation_text="Current Hotel",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Normalized Risk Score",
        yaxis_title="Number of Hotels",
        barmode='overlay',
        showlegend=True,
        legend_title="Risk Zones"
    )
    
    return fig

def create_radar_chart(hotel_data):
    """Create radar chart showing all risk score components"""
    # Define components with their weights and whether higher is worse
    components = [
        ('mean_rating', 'Mean Rating', -0.2, False),  # False = higher is better
        ('low_score_proportion', 'Low Score %', 0.15, True),
        ('variance_rating', 'Rating Variance', 0.1, True),
        ('confidence_interval', 'Confidence Int.', 0.1, True),
        ('recent_trend', 'Recent Trend', -0.1, False),
        ('stay_type_variation', 'Stay Type Var.', 0.1, True),
        ('room_type_variation', 'Room Type Var.', 0.1, True),
        ('time_variation', 'Time Var.', 0.1, True),
        ('performance_deviation', 'Perf. Dev.', -0.05, False),
        ('guest_origin_variation', 'Origin Var.', 0.1, True),
        ('stability_index', 'Stability', -0.1, False)
    ]
    
    values = []
    labels = []
    for comp, label, weight, higher_is_worse in components:
        val = float(hotel_data[comp])
        
        # Normalize value to 0-1 scale if needed
        if comp == 'mean_rating':
            val = val / 10
        
        # Invert values where higher is better to make radar consistent
        # (larger area always means more risk)
        if not higher_is_worse:
            val = 1 - val
            
        values.append(val)
        labels.append(label)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Risk Levels'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                title='Risk Level',
                tickmode='array',
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0.33, 0.66, 1.0]
            )),
        showlegend=False,
        title="Risk Components Overview (Larger = Higher Risk)"
    )
    
    # Add reference circles for risk levels
    for level, color in [(0.33, "green"), (0.66, "orange")]:
        fig.add_trace(go.Scatterpolar(
            r=[level] * len(labels),
            theta=labels,
            mode='lines',
            line=dict(color=color, dash='dot'),
            showlegend=False
        ))
    
    return fig

def create_ratings_analysis(raw_df, hotel_name):
    """Create ratings analysis with control boundaries"""
    
    # Filter for selected hotel
    hotel_ratings = raw_df[raw_df['combined_name'] == hotel_name].copy()
    
    # Group by date for daily aggregation
    daily_ratings = hotel_ratings.groupby(hotel_ratings['date'].dt.date).agg({
        'rating': 'mean',
        'combined_name': 'count'  # This gives us daily review count
    }).reset_index()
    
    weekly_ratings = hotel_ratings.groupby(hotel_ratings['date'].dt.to_period('W')).agg({
        'rating': 'mean',
        'combined_name': 'count'  # This gives us weekly review count
    }).reset_index()
    weekly_ratings['date'] = weekly_ratings['date'].dt.start_time  # Convert period to timestamp
    
    # Calculate control limits
    mean = weekly_ratings['rating'].mean()
    std = weekly_ratings['rating'].std()
    ucl = min(10, mean + 2*std)
    lcl = max(0, mean - 2*std)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add ratings line
    fig.add_trace(
        go.Scatter(
            x=weekly_ratings['date'],
            y=weekly_ratings['rating'],
            name="Weekly avg. Rating",
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    #add 30 day ema
    daily_ratings['30_day_ema'] = daily_ratings['rating'].ewm(span=30).mean()
    fig.add_trace(
        go.Scatter(
            x=daily_ratings['date'],
            y=daily_ratings['30_day_ema'],
            name="30 Day EMA",
            line=dict(color='red')
        ),
        secondary_y=False
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=daily_ratings['date'],
            y=daily_ratings['combined_name'],
            name="Review Count",
            marker_color='lightgray',
            opacity=0.5
        ),
        secondary_y=True
    )
    
    # Add mean and control limits
    fig.add_hline(
        y=mean, 
        line_dash="solid", 
        line_color="green",
        annotation=dict(
            text=f"Mean: {mean:.2f}",
            xanchor="left",
            x=1.02
        )
    )
    fig.add_hline(
        y=ucl, 
        line_dash="dash", 
        line_color="red",
        annotation=dict(
            text=f"UCL: {ucl:.2f}",
            xanchor="left",
            x=1.02
        )
    )
    fig.add_hline(
        y=lcl, 
        line_dash="dash", 
        line_color="red",
        annotation=dict(
            text=f"LCL: {lcl:.2f}",
            xanchor="left",
            x=1.02
        )
    )
    
    fig.update_layout(
        title="Daily Ratings Analysis with Control Limits",
        xaxis_title="Date",
        yaxis_title="Rating",
        yaxis2_title="Review Count",
        height=500
    )
    
    # Update axis ranges
    fig.update_yaxes(range=[0, 10], secondary_y=False)
    fig.update_yaxes(range=[0, max(daily_ratings['combined_name'])*1.2], secondary_y=True)
    
    return fig

def app():
    st.set_page_config(page_title="Hotel Risk Analysis", layout="wide")
    
    # Load data
    risk_df, raw_df, raw_risk_df = load_data()
    
    # Sidebar
    st.sidebar.title("Hotel Selection")
    #sort hotels by normalized risk score
    risk_df = risk_df.sort_values(by='normalized_risk_score', ascending=False)
    selected_hotel = st.sidebar.selectbox(
        "Select Hotel",
        options=risk_df['combined_name'].unique()
    )
    
    # Filter data for selected hotel
    hotel_data = risk_df[risk_df['combined_name'] == selected_hotel].iloc[0]
 
    
    # Main layout
    st.title(f"Risk Analysis: {selected_hotel}")
    
    # 1. Key Metrics Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Normalized Risk Score",
            f"{hotel_data['normalized_risk_score']:.1f}",
            hotel_data['normalized_risk_score_label'],
            delta_color=get_delta_color(hotel_data['normalized_risk_score_label'])
        )
    
    with col2:
        percentile = (risk_df['normalized_risk_score'] <= hotel_data['normalized_risk_score']).mean() * 100
        st.metric(
            "Percentile Position", 
            f"{percentile:.1f}%",
            delta_color="inverse"  # Higher percentile is worse for risk
        )
        
    with col3:
        rank = (risk_df['normalized_risk_score'] <= hotel_data['normalized_risk_score']).sum()
        st.metric(
            "Rank",
            f"{rank} / {len(risk_df)}",
            delta_color="inverse"  # Higher rank is worse for risk
        )
    
    with col1:
        #ranking by review count
        review_count_rank = (risk_df['review_count'] <= hotel_data['review_count']).sum()
        st.metric(
            "Review Count",
            hotel_data['review_count'],
            f"{review_count_rank}"
        )
        
    with col2:
        st.metric(
            "Mean Rating",
            f"{hotel_data['mean_rating']:.2f}",
            hotel_data['mean_rating_label'],
            delta_color=get_delta_color(hotel_data['mean_rating_label'])
        )
        
    with col3:
        #ema 30 mean rating for selected hotel from raw_df
        hotel_ratings = raw_df[raw_df['combined_name'] == selected_hotel].copy()
        hotel_ratings['date'] = pd.to_datetime(hotel_ratings['review_post_date'])
        hotel_ratings = hotel_ratings.sort_values(by='date')
        hotel_ratings['30_day_ema'] = hotel_ratings['rating'].ewm(span=30).mean()
        mean_ema_better = hotel_ratings['rating'].mean() > hotel_ratings['30_day_ema'].iloc[-1]
        st.metric(
            "30 Day EMA",
            f"{hotel_ratings['30_day_ema'].iloc[-1]:.2f}",
            f"{mean_ema_better}"
        )
    
    #create API signal badges based on ranking and review count and mean rating
    #create 3 badges for each column color and text based on ranking and review count and mean rating
    #sample badges
    st.sidebar.title("API Signals")
    if hotel_data['normalized_risk_score_label'] == 'High':
        badges = f"""{create_badge("High Risk", "#FF5733")}"""
        st.sidebar.markdown(badges, unsafe_allow_html=True)
    
    # badges = f"""
    # {create_badge("SUPER CARE", "#4CAF50")} 
    # {create_badge("EXPERIENCE RISK", "#FF5733")} 
    # {create_badge("SOLO TRAVELER", "#2196F3")}
    # """
    
    st.sidebar.warning('This is a warning', icon=":material/thumb_up:")
    st.sidebar.info('Tier 1', icon="ℹ️")
    st.sidebar.success('Low Complexity', icon="✅")
    
    with col1:
        st.metric(
            "Low Score Proportion",
            f"{hotel_data['low_score_proportion']:.2%}",
            hotel_data['low_score_proportion_label'],
            delta_color=get_delta_color(hotel_data['low_score_proportion_label'])
        )
    
    
    # 2. Risk Distribution and Radar Chart
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(
            create_risk_distribution(risk_df, hotel_data),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_radar_chart(hotel_data),
            use_container_width=True
        )
    
    # 3. Ratings Analysis
    st.plotly_chart(
        create_ratings_analysis(raw_df, selected_hotel),
        use_container_width=True
    )
    
    
    # 3.5 Heatmap of Guest Origin and Room Type
    st.plotly_chart(
        create_heatmap_guest_origin_room_type(raw_df, selected_hotel),
        use_container_width=True
    )
    
    # 3.6 Heatmap of Stay Type and Room Type
    st.plotly_chart(
        create_heatmap_stay_type_room_type(raw_df, selected_hotel),
        use_container_width=True
    )
    
    # 3.7 Average Rating by Stay Type
    show_average_rating_by_stay_type(raw_df, selected_hotel)
    
    # 3.8 Average Rating by Room Type
    show_average_rating_by_room_type(raw_df, selected_hotel)
    
    #3.9 Treemap of Stay Type and Room Type
    st.plotly_chart(
        create_treemap_stay_type_room_type(raw_df, selected_hotel),
        use_container_width=True
    )
    
    #3.10 Treemap of Guest Origin and Room Type
    st.plotly_chart(
        create_treemap_guest_origin_room_type(raw_df, selected_hotel),
        use_container_width=True
    )
    

    
    # 4. Detailed Metrics Analysis
    st.subheader("Detailed Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Core Performance Metrics
        with st.expander("Core Performance Metrics", expanded=True):
            metrics_df = pd.DataFrame({
                'Metric': ['Mean Rating', 'Low Score Proportion', 'Rating Variance'],
                'Value': [
                    f"{hotel_data['mean_rating']:.2f}",
                    f"{hotel_data['low_score_proportion']:.2%}",
                    f"{hotel_data['variance_rating']:.2f}"
                ],
                'Evaluation': [
                    hotel_data['mean_rating_label'],
                    hotel_data['low_score_proportion_label'],
                    hotel_data['variance_rating_label']
                ]
            })
            st.table(metrics_df)
        
        # Trend Metrics
        with st.expander("Trend Analysis", expanded=True):
            metrics_df = pd.DataFrame({
                'Metric': ['Recent Trend', 'Performance Deviation'],
                'Value': [
                    f"{hotel_data['recent_trend']:.3f}",
                    f"{hotel_data['performance_deviation']:.2f}"
                ],
                'Evaluation': [
                    hotel_data['recent_trend_label'],
                    hotel_data['performance_deviation_label']
                ]
            })
            st.table(metrics_df)
        
        # Variation Metrics
        with st.expander("Variation Analysis", expanded=True):
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Stay Type Variation',
                    'Room Type Variation',
                    'Time Variation',
                    'Guest Origin Variation'
                ],
                'Value': [
                    f"{hotel_data['stay_type_variation']:.2f}",
                    f"{hotel_data['room_type_variation']:.2f}",
                    f"{hotel_data['time_variation']:.2f}",
                    f"{hotel_data['guest_origin_variation']:.2f}"
                ],
                'Evaluation': [
                    hotel_data['stay_type_variation_label'],
                    hotel_data['room_type_variation_label'],
                    hotel_data['time_variation_label'],
                    hotel_data['guest_origin_variation_label']
                ]
            })
            st.table(metrics_df)
        
        # Statistical Reliability
        with st.expander("Statistical Reliability", expanded=True):
            metrics_df = pd.DataFrame({
                'Metric': ['Confidence Interval', 'Review Count', 'Stability Index'],
                'Value': [
                    f"{hotel_data['confidence_interval']:.2f}",
                    f"{hotel_data['review_count']}",
                    f"{hotel_data['stability_index']:.2f}"
                ],
                'Evaluation': [
                    hotel_data['confidence_interval_label'],
                    'N/A',
                    hotel_data['stability_index_label']
                ]
            })
            st.table(metrics_df)
            
        # Disliked Texts
    with st.expander("Disliked Texts", expanded=True):
        st.title("Negative Reviews and Summary/Suggestions")
        disliked_texts = get_disliked_texts_by_hotel(raw_df, selected_hotel)
        st.table(disliked_texts)
        
        
        from dotenv import load_dotenv
        import os
        import openai 


        load_dotenv(override=True)
        OPENAPI_API_KEY = os.getenv("OPENAPI_API_KEY")
        client = openai.Client()



        # Create a chat completion
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': f'You are a hotel review analyst and operations manager. Summarize these disliked texts for the hotel {disliked_texts}. First the concise and structured summary (include customer voices in english if needed) then, Give a bulletpoint list of recommendations to improve the hotel operations. Write all in markdown'},
            ]
        )

        # Print the completion
        st.markdown(response.choices[0].message.content)

if __name__ == "__main__":
    app()