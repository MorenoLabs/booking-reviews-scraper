import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt


@dataclass
class RiskComponents:
    low_rating: float
    performance: float
    volatility: float
    trend: float
    
@dataclass
class TrendComponents:
    rating_momentum: float
    risk_velocity: float
    seasonality: float
    volume_trend: float

class HotelRiskAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.baselines = {}
        self.risk_scores = {}
        self.trend_scores = {}
        
        # Convert review_post_date to datetime and extract Month
        # if 'review_post_date' in self.df.columns:
        #     self.df['review_post_date'] = pd.to_datetime(self.df['review_post_date'], errors='coerce')
        #     self.df['Month'] = self.df['review_post_date'].dt.to_period('M')  # Extract year and month as a period
        
        # Configuration
        self.LOW_RATING_THRESHOLD = 7
        self.TREND_WINDOW = 3  # months
        self.WEIGHTS = {
            'base_risk': {
                'low_rating': 0.4,
                'performance': 0.3,
                'volatility': 0.3
            },
            'trend': {
                'rating_momentum': 0.4,
                'risk_velocity': 0.3,
                'seasonality': 0.1,
                'volume_trend': 0.2
            }
        }
        
        # Initialize baselines on creation
        self.initialize_baselines()
        
    def initialize_baselines(self):
        """Calculate baseline metrics for each stay type"""
        for stay_type in self.df['stay_type'].str.strip().str.lower().unique():
            stay_data = self.df[self.df['stay_type'].str.strip().str.lower() == stay_type]
            
            # Calculate baseline metrics
            avg_rating = stay_data['rating'].mean()
            std_dev = stay_data['rating'].std()
            low_rating_pct = (stay_data['rating'] < self.LOW_RATING_THRESHOLD).mean()
            
            self.baselines[stay_type] = {
                'avg_rating': avg_rating,
                'std_dev': std_dev,
                'low_rating_pct': low_rating_pct,
                'count': len(stay_data)
            }
        print("Initialized baselines with normalized keys:", list(self.baselines.keys()))


    def calculate_base_risk(self, hotel_data: pd.DataFrame, 
                          baseline: Dict) -> RiskComponents:
        """Calculate base risk components"""
        avg_rating = hotel_data['rating'].mean()
        low_ratings = (hotel_data['rating'] < self.LOW_RATING_THRESHOLD).mean()
        
        # Low rating component
        low_rating_ratio = low_ratings / baseline['low_rating_pct']
        low_rating_component = low_rating_ratio * self.WEIGHTS['base_risk']['low_rating']
        
        # Performance component
        z_score = (avg_rating - baseline['avg_rating']) / \
                 (baseline['std_dev'] / np.sqrt(len(hotel_data)))
        performance_component = (abs(z_score) if z_score < -1 else 0) * \
                              self.WEIGHTS['base_risk']['performance']
        
        # Volatility component
        rating_variance = hotel_data['rating'].var()
        volatility_ratio = rating_variance / (baseline['std_dev'] ** 2)
        volatility_component = volatility_ratio * self.WEIGHTS['base_risk']['volatility']
        
        return RiskComponents(
            low_rating=low_rating_component,
            performance=performance_component,
            volatility=volatility_component,
            trend=0  # Will be updated later
        )

    def calculate_monthly_risk(self, monthly_data: pd.DataFrame) -> float:
        """Calculate monthly risk based on average rating."""
        avg_rating = monthly_data['rating'].mean()
        std_dev = monthly_data['rating'].std()

        # Higher risk for lower ratings and higher variability
        risk = (1 - avg_rating / 10) + (std_dev / 10)  # Scale risk between 0 and 2
        return max(0, risk)  # Ensure risk is non-negative
    
    def calculate_trend_components(self, hotel_data: pd.DataFrame) -> TrendComponents:
        """Calculate all trend components"""

        # 1. Rating Momentum
        monthly_ratings = hotel_data.groupby('month')['rating'].mean()
        if len(monthly_ratings) >= self.TREND_WINDOW:
            rating_slope = stats.linregress(
                range(len(monthly_ratings[-self.TREND_WINDOW:])),
                monthly_ratings[-self.TREND_WINDOW:]
            )[0]
            rating_momentum = -rating_slope if rating_slope < 0 else 0
        else:
            rating_momentum = 0

        # 2. Risk Velocity
        monthly_risks = pd.Series({
            month: self.calculate_monthly_risk(data)
            for month, data in hotel_data.groupby('month')
        })
        risk_velocity = monthly_risks.diff().mean() if len(monthly_risks) > 1 else 0

        # 3. Volume Trend
        monthly_volumes = hotel_data.groupby('month').size()
        volume_slope = stats.linregress(range(len(monthly_volumes)), monthly_volumes)[0]
        volume_trend = -volume_slope if volume_slope < 0 else 0

        return TrendComponents(
            rating_momentum=rating_momentum * self.WEIGHTS['trend']['rating_momentum'],
            risk_velocity=risk_velocity * self.WEIGHTS['trend']['risk_velocity'],
            seasonality=0,  # Default to 0 if not calculated
            volume_trend=volume_trend * self.WEIGHTS['trend']['volume_trend']
        )


    def calculate_seasonality(self, hotel_data: pd.DataFrame) -> float:
        """Calculate seasonality impact"""
        # Convert Month to datetime
        hotel_data['date'] = pd.to_datetime(hotel_data['month'])
        monthly_ratings = hotel_data.groupby('month')['rating'].mean()
        
        # Compare current period to same period last year
        # This is a simplified version - could be more sophisticated
        current_month = monthly_ratings.index[-1]
        try:
            seasonal_diff = monthly_ratings[current_month] - \
                          monthly_ratings[current_month - 12]
            return -seasonal_diff if seasonal_diff < 0 else 0
        except:
            return 0

    def calculate_combined_risk_score(self, hotel: str, stay_type: str) -> Dict:
        """Calculate final risk score combining base and trend components"""
        # Clean hotel name first (remove timestamp)
        hotel_base_name = hotel.split('_')[0]
        
        #lowecase column names
        self.df['Hotel Name'] = self.df['Hotel Name'].str.lower()
        self.df['stay_type'] = self.df['stay_type'].str.lower()
        print(f"Filtering data for {hotel_base_name} - {stay_type}")  # Debug print
        
        # Filter data using base name
        hotel_data = self.df[
            (self.df['Hotel Name'].str.startswith(hotel_base_name)) & 
            (self.df['stay_type'] == stay_type)
        ]
        
        print(f"Found {len(hotel_data)} records for {hotel_base_name} - {stay_type}")  # Debug print
        
        if len(hotel_data) < 10:  # Minimum threshold
            return None
            
        # Calculate base risk
        base_components = self.calculate_base_risk(
            hotel_data, 
            self.baselines[stay_type]
        )
        
        # Calculate trend components
        trend_components = self.calculate_trend_components(hotel_data)
        
        # Calculate trend multiplier
        trend_multiplier = 1.0 + sum([
            trend_components.rating_momentum,
            trend_components.risk_velocity,
            trend_components.seasonality,
            trend_components.volume_trend
        ])
        
        # Calculate final risk score
        base_risk = sum([
            base_components.low_rating,
            base_components.performance,
            base_components.volatility
        ])
        
        final_risk = base_risk * trend_multiplier
        
        return {
            'final_risk_score': final_risk,
            'base_components': base_components,
            'trend_components': trend_components,
            'trend_multiplier': trend_multiplier,
            'metrics': {
                'avg_rating': hotel_data['rating'].mean(),
                'review_count': len(hotel_data),
                'low_rating_pct': (hotel_data['rating'] < 7).mean() * 100
            }
        }

    def get_risk_category(self, risk_score: float) -> Tuple[str, str]:
        """Determine risk category and recommended action"""
        if risk_score > 3.5:
            return "Critical", "Immediate intervention required"
        elif risk_score > 2.5:
            return "High Risk", "Urgent attention needed"
        elif risk_score > 1.5:
            return "Moderate Risk", "Active monitoring required"
        else:
            return "Low Risk", "Routine monitoring"
        
    def calculate_monthly_risk_scores(self, hotel_base_name: str, stay_type: str) -> pd.DataFrame:
        """Calculate risk scores for each month to analyze the trend."""
        hotel_data = self.df[
            (self.df['Hotel Name'].str.startswith(hotel_base_name)) &
            (self.df['stay_type'] == stay_type)
        ]
        monthly_risk_scores = []

        for month, data in hotel_data.groupby('month'):
            if len(data) < 10:  # Skip months with insufficient data
                continue

            # Calculate base risk
            base_components = self.calculate_base_risk(data, self.baselines[stay_type])
            base_risk = sum([
                base_components.low_rating,
                base_components.performance,
                base_components.volatility
            ])

            # Trend Components (only relevant metrics)
            trend_components = self.calculate_trend_components(data)
            trend_multiplier = 1.0 + sum([
                trend_components.rating_momentum,
                trend_components.risk_velocity,
                trend_components.seasonality
            ])

            # Final risk score for the month
            final_risk = base_risk * trend_multiplier
            monthly_risk_scores.append({'month': month, 'Risk Score': final_risk})

        return pd.DataFrame(monthly_risk_scores)

    def generate_risk_report(self, hotel: str, stay_type: str) -> Dict:
        """Generate comprehensive risk report"""
        print(f"Generating report for hotel: '{hotel}', stay_type: '{stay_type}'")

        # Normalize inputs
        hotel_base_name = hotel.split('_')[0].strip().lower()
        stay_type = stay_type.strip().lower()

        # Ensure baseline exists
        if stay_type not in self.baselines:
            print(f"Error: Stay type '{stay_type}' not found in baselines.")
            print("Available stay types in baselines:", list(self.baselines.keys()))
            return None

        # Filter data
        filtered_data = self.df[
            (self.df['Hotel Name'].str.startswith(hotel_base_name)) & 
            (self.df['stay_type'].str.strip().str.lower() == stay_type)
        ]

        print(f"Filtered Data Length: {len(filtered_data)}")
        if len(filtered_data) < 10:
            print("Filtered rows are below the threshold of 10.")
            return None
        
        self.plot_volume_and_rating(filtered_data)

        # Calculate combined risk score
        risk_data = self.calculate_combined_risk_score(hotel, stay_type)
        if not risk_data:
            print("No risk data generated.")
            return None

        category, action = self.get_risk_category(risk_data['final_risk_score'])

        trend_direction = "Deteriorating" if risk_data['trend_multiplier'] > 1.1 else \
                        "Stable" if risk_data['trend_multiplier'] <= 1.1 else \
                        "Improving"
                        
        baseline = self.baselines.get(stay_type)
        if baseline:
            baseline_avg_rating = baseline['avg_rating']
        else:
            baseline_avg_rating = None
            
            

        return {
            'hotel': hotel,
            'stay_type': stay_type,
            'risk_category': category,
            'recommended_action': action,
            'trend_direction': trend_direction,
            'risk_score': risk_data['final_risk_score'],
            'baseline_avg_rating': baseline_avg_rating,  # Add baseline average
            'components': {
                'base': risk_data['base_components'],
                'trend': risk_data['trend_components']
            },
            'metrics': risk_data['metrics']
        }

    def plot_volume_and_rating(self, hotel_data: pd.DataFrame):
        """Generate a dual-axis plot for volume and rating trends."""
        # Prepare data
        monthly_data = hotel_data.groupby('month').agg({
            'rating': 'mean',
            'month': 'size'
        }).rename(columns={'month': 'volume'}).reset_index()
        
        # Sort by date
        monthly_data['month'] = pd.to_datetime(monthly_data['month'])
        monthly_data = monthly_data.sort_values('month')

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Rating (Left Axis)
        ax1.set_xlabel('month')
        ax1.set_ylabel('Average Rating', color='blue')
        ax1.plot(monthly_data['month'], monthly_data['rating'], marker='o', label='Average Rating', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Volume (Right Axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Review Volume', color='orange')
        ax2.bar(monthly_data['month'], monthly_data['volume'], alpha=0.5, label='Review Volume', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Title and Legend
        plt.title('monthly Trend: Ratings and Review Volume')
        fig.tight_layout()
        plt.show()
    

        
