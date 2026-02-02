# Enhanced Software Monetization Dashboard

## Overview
This enhanced dashboard consolidates all current features into a single organized section and adds advanced analytics capabilities for product recommendations, churn prediction, customer segmentation, and survival analysis.

## Key Features

### 1. Main Dashboard Section
- **Consolidated View**: All existing dashboard features (product trends, usage trends, deployment percentages, world map, top customers/products) are now organized in a single section
- **Unified Interface**: No more separate buttons for each feature - everything is visible at once

### 2. Advanced Analytics Section

#### Product Recommendations
- **Apriori Algorithm**: Uses association rule mining to recommend products based on customer purchase patterns
- **Collaborative Filtering**: Customer-product matrix analysis for personalized recommendations
- **Confidence & Support Metrics**: Shows recommendation strength and frequency

#### Churn Prediction & Driver Analysis
- **Logistic Regression Model**: Predicts customer churn probability
- **Feature Importance**: Identifies key drivers of churn
- **Risk Assessment**: Categorizes customers as High/Medium/Low risk
- **Actionable Recommendations**: Provides specific strategies to prevent churn

#### Customer Segmentation
- **K-Means Clustering**: Automatically segments customers into 4 groups
- **Premium vs Non-Premium**: Identifies high-value customers
- **Segment Characteristics**: Describes each segment's behavior and value

#### Survival Analysis
- **Renewal Prediction**: Analyzes customer retention over time
- **Risk Assessment**: Identifies customers at risk of not renewing
- **Time-to-Event Analysis**: Shows survival probability curves

## Technical Implementation

### Machine Learning Models
- **Churn Prediction**: Logistic Regression with feature scaling
- **Customer Segmentation**: K-Means clustering with standardized features
- **Product Recommendations**: Apriori algorithm for association rules

### Data Processing
- **Customer Summary**: Comprehensive customer profiles with purchase, activation, and usage metrics
- **Feature Engineering**: Calculated metrics for days since last activity, total values, etc.
- **Data Validation**: Handles missing values and data quality issues

### API Endpoints
- `/api/product-recommendations/<customer_id>` - Get product recommendations
- `/api/churn-prediction/<customer_id>` - Predict churn and get driver analysis
- `/api/customer-segmentation` - Perform customer segmentation
- `/api/survival-analysis` - Run survival analysis

## Usage Instructions

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Dashboard**: `python enhanced_dashboard_complete.py`
3. **Access Dashboard**: Open browser to `http://localhost:5000`
4. **Navigate Sections**: Use the two main navigation buttons:
   - Main Dashboard: All current features in one view
   - Advanced Analytics: New ML-powered features

## Business Value

### For Sales Teams
- **Product Recommendations**: Increase cross-selling and upselling opportunities
- **Customer Segmentation**: Prioritize high-value prospects and customers
- **Churn Prevention**: Proactively address at-risk customers

### For Customer Success
- **Early Warning System**: Identify customers likely to churn
- **Personalized Engagement**: Tailor strategies based on customer segments
- **Renewal Optimization**: Focus efforts on customers with renewal risk

### For Product Management
- **Usage Patterns**: Understand how customers use products
- **Feature Adoption**: Identify popular product combinations
- **Market Insights**: Analyze customer behavior and preferences

## Data Requirements

The dashboard requires the following CSV files in the `dataset/` folder:
- `customers.csv` - Customer information and demographics
- `products.csv` - Product catalog
- `entitlements.csv` - License purchases and contracts
- `activations.csv` - Product usage and activations
- `users.csv` - User activity and login data
- `renewals.csv` - License renewal information

## Future Enhancements

1. **Real-time Updates**: Live data streaming and real-time predictions
2. **Advanced ML Models**: Deep learning for more accurate predictions
3. **Interactive Visualizations**: Dynamic charts and drill-down capabilities
4. **Automated Alerts**: Proactive notifications for high-risk customers
5. **A/B Testing**: Test different intervention strategies


