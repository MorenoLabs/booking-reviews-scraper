import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import os
import glob

# Set the path to the output directory containing CSVs
output_dir = "../booking-reviews-scraper/output"

# Collect all CSV file paths in the directory and subdirectories
csv_files = glob.glob(os.path.join(output_dir, "**/*.csv"), recursive=True)
# Create a new DataFrame with hotel names extracted from the folder paths
combined_reviews = pd.DataFrame()

for csv_file in csv_files:
    # Extract the hotel name from the folder path
    hotel_name = os.path.basename(os.path.dirname(csv_file))
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Add a column for the hotel name
    df['Hotel Name'] = hotel_name
    
    # Combine with the main DataFrame
    combined_reviews = pd.concat([combined_reviews, df], ignore_index=True)

# Load the DataFrame (replace with actual loading code)
# df = pd.read_csv("path_to_combined_reviews.csv")
# Using a placeholder DataFrame for demonstration


# Convert to DataFrame
df = combined_reviews.copy()


# Convert review_post_date to datetime
df['review_post_date'] = pd.to_datetime(df['review_post_date'])

# Streamlit app
st.title("Hotel Reviews Analytics Dashboard")

# Filters
st.sidebar.header("Filters")

# Filter by hotel name
hotel_names = df["Hotel Name"].unique()
selected_hotel = st.sidebar.selectbox("Select Hotel", options=["All"] + list(hotel_names))

# Filter by rating range
rating_min, rating_max = st.sidebar.slider("Select Rating Range", min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)

# Filter by date range
date_min = df['review_post_date'].min()
date_max = df['review_post_date'].max()
selected_date_range = st.sidebar.date_input("Select Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

# Apply filters
filtered_df = df.copy()
if selected_hotel != "All":
    filtered_df = filtered_df[filtered_df["Hotel Name"] == selected_hotel]
filtered_df = filtered_df[(filtered_df["rating"] >= rating_min) & (filtered_df["rating"] <= rating_max)]
filtered_df = filtered_df[(filtered_df["review_post_date"] >= pd.Timestamp(selected_date_range[0])) & (filtered_df["review_post_date"] <= pd.Timestamp(selected_date_range[1]))]

# Display filtered data
st.subheader("Filtered Reviews")
st.dataframe(filtered_df)

# Analytics: Average Ratings and Review Volume
st.subheader("Average Ratings and Review Volume")
avg_ratings = filtered_df.groupby("Hotel Name")["rating"].mean().reset_index()
review_counts = filtered_df.groupby("Hotel Name")["rating"].count().reset_index()
review_counts.columns = ["Hotel Name", "Review Volume"]

# Merge average ratings and review volume
overall_stats = pd.merge(avg_ratings, review_counts, on="Hotel Name")

# Plot with twin axes
fig = go.Figure()

# Add bar for review volume
fig.add_trace(go.Bar(
    x=overall_stats["Hotel Name"],
    y=overall_stats["Review Volume"],
    name="Review Volume",
    yaxis="y1"
))

# Add line for average rating
fig.add_trace(go.Scatter(
    x=overall_stats["Hotel Name"],
    y=overall_stats["rating"],
    name="Average Rating",
    yaxis="y2",
    mode="lines+markers"
))

# Update layout for twin axes
fig.update_layout(
    title="Average Ratings and Review Volume by Hotel",
    xaxis_title="Hotel Name",
    yaxis=dict(
        title="Review Volume",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue"),
        side="left"
    ),
    yaxis2=dict(
        title="Average Rating",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right",
        range=[0, 10]
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    barmode="group"
)

st.plotly_chart(fig)

