from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store data and models
data = {}
models = {}

def load_data():
    """Load all CSV files into memory"""
    global data
    try:
        data['customers'] = pd.read_csv('dataset/customers.csv')
        data['products'] = pd.read_csv('dataset/products.csv')
        data['features'] = pd.read_csv('dataset/features.csv')
        data['product_features'] = pd.read_csv('dataset/product_features.csv')
        data['entitlements'] = pd.read_csv('dataset/entitlements.csv')
        data['activations'] = pd.read_csv('dataset/activations.csv')
        data['users'] = pd.read_csv('dataset/users.csv')
        data['renewals'] = pd.read_csv('dataset/renewals.csv')
        
        # Convert date columns
        data['entitlements']['purchase_date'] = pd.to_datetime(data['entitlements']['purchase_date'])
        data['activations']['activation_date'] = pd.to_datetime(data['activations']['activation_date'])
        data['users']['first_login_date'] = pd.to_datetime(data['users']['first_login_date'])
        data['renewals']['renewal_date'] = pd.to_datetime(data['renewals']['renewal_date'])
        
        print("Data loaded successfully!")
        prepare_analytics_data()
    except Exception as e:
        print(f"Error loading data: {e}")

def prepare_analytics_data():
    """Prepare data for advanced analytics"""
    global data
    
    # Create customer summary for segmentation
    customer_summary = create_customer_summary()
    data['customer_summary'] = customer_summary
    
    # Prepare churn prediction data
    prepare_churn_data()
    
    # Prepare product recommendation data
    prepare_recommendation_data()

def create_customer_summary():
    """Create comprehensive customer summary for segmentation"""
    # Merge customer data with entitlements and activations
    customer_entitlements = data['entitlements'].groupby('customer_id').agg({
        'purchase_quantity': 'sum',
        'contract_value': 'sum',
        'purchase_date': ['min', 'max']
    }).reset_index()
    
    customer_entitlements.columns = ['customer_id', 'total_purchased', 'total_contract_value', 'first_purchase', 'last_purchase']
    
    customer_activations = data['activations'].merge(
        data['entitlements'][['entitlement_id', 'customer_id']], 
        on='entitlement_id'
    ).groupby('customer_id').agg({
        'quantity': 'sum',
        'activation_date': ['min', 'max']
    }).reset_index()
    
    customer_activations.columns = ['customer_id', 'total_activated', 'first_activation', 'last_activation']
    
    # Merge with customer data
    customer_summary = data['customers'].merge(customer_entitlements, on='customer_id', how='left')
    customer_summary = customer_summary.merge(customer_activations, on='customer_id', how='left')
    
    # Calculate additional metrics
    customer_summary['days_since_first_purchase'] = (datetime.now() - customer_summary['first_purchase']).dt.days
    customer_summary['days_since_last_purchase'] = (datetime.now() - customer_summary['last_purchase']).dt.days
    customer_summary['days_since_last_activation'] = (datetime.now() - customer_summary['last_activation']).dt.days
    
    # Fill NaN values
    customer_summary = customer_summary.fillna(0)
    
    return customer_summary

def prepare_churn_data():
    """Prepare data for churn prediction"""
    # Create churn labels based on renewal status
    churn_data = data['customer_summary'].copy()
    
    # Define churn based on business rules
    churn_data['churn'] = 0
    churn_data.loc[churn_data['days_since_last_activation'] > 90, 'churn'] = 1
    churn_data.loc[churn_data['status'] == 'Inactive', 'churn'] = 1
    churn_data.loc[churn_data['status'] == 'Suspended', 'churn'] = 1
    
    # Select features for churn prediction
    churn_features = [
        'total_purchased', 'total_contract_value', 'total_activated',
        'days_since_first_purchase', 'days_since_last_purchase', 'days_since_last_activation'
    ]
    
    X = churn_data[churn_features]
    y = churn_data['churn']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train churn prediction model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Store models
    models['churn_scaler'] = scaler
    models['churn_model'] = lr_model
    models['churn_features'] = churn_features
    models['churn_data'] = churn_data

def prepare_recommendation_data():
    """Prepare data for product recommendations"""
    # Create product association rules using Apriori
    product_transactions = data['entitlements'].groupby('entitlement_id')['product_id'].apply(list).tolist()
    
    # Convert to one-hot encoded format
    product_df = pd.DataFrame(product_transactions)
    product_df = pd.get_dummies(product_df.stack()).groupby(level=0).sum()
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(product_df, min_support=0.01, use_colnames=True)
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        models['association_rules'] = rules
    else:
        models['association_rules'] = pd.DataFrame()
    
    # Create customer-product matrix for collaborative filtering
    customer_product_matrix = data['entitlements'].pivot_table(
        index='customer_id', 
        columns='product_id', 
        values='purchase_quantity', 
        fill_value=0
    )
    models['customer_product_matrix'] = customer_product_matrix

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Software Monetization Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; text-align: center; }
        .nav { background: #34495e; padding: 0.5rem; text-align: center; }
        .nav button { background: #3498db; color: white; border: none; padding: 10px 20px; margin: 0 5px; cursor: pointer; border-radius: 5px; }
        .nav button:hover { background: #2980b9; }
        .nav button.active { background: #e74c3c; }
        .container { padding: 20px; max-width: 1400px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .page { display: none; }
        .page.active { display: block; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .controls { margin-bottom: 20px; }
        .controls select { padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-right: 10px; }
        .controls button { padding: 10px 20px; background: #27ae60; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .table-container { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .churn-high { background-color: #ffebee; }
        .churn-medium { background-color: #fff3e0; }
        .churn-low { background-color: #e8f5e8; }
        .segment-premium { background-color: #e3f2fd; }
        .segment-standard { background-color: #f3e5f5; }
        .segment-basic { background-color: #fff8e1; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced Software Monetization Dashboard</h1>
    </div>
    
    <div class="nav">
        <button onclick="showPage('main-dashboard')" class="active" id="main-dashboard-btn">Main Dashboard</button>
        <button onclick="showPage('advanced-analytics')" id="advanced-analytics-btn">Advanced Analytics</button>
    </div>

    <div class="container">
        <!-- Main Dashboard Page - All Current Features -->
        <div id="main-dashboard" class="page active">
            <h2>Main Dashboard - All Features</h2>
            
            <!-- Overview Stats -->
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be loaded here -->
            </div>
            
            <!-- All Charts in One Section -->
            <div class="grid-2">
                <div class="chart-container">
                    <h3>Product Quantity Trends</h3>
                    <div class="controls">
                        <select id="product-select" onchange="updateProductChart()">
                            <option value="">Select Product</option>
                        </select>
                    </div>
                    <canvas id="product-chart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>Usage Trends</h3>
                    <div class="controls">
                        <select id="usage-product-select" onchange="updateUsageChart()">
                            <option value="">Select Product</option>
                        </select>
                    </div>
                    <canvas id="usage-chart"></canvas>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="chart-container">
                    <h3>Deployment Percentage by Product</h3>
                    <canvas id="deployment-chart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>User Locations Worldwide</h3>
                    <div id="map"></div>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="chart-container">
                    <h3>Top 10 Customers by Purchase Quantity</h3>
                    <canvas id="customers-purchase-chart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>Top 10 Customers by Activation Quantity</h3>
                    <canvas id="customers-activation-chart"></canvas>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="chart-container">
                    <h3>Top 10 Products by Purchase Quantity</h3>
                    <canvas id="products-purchase-chart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>Top 10 Products by Activation Quantity</h3>
                    <canvas id="products-activation-chart"></canvas>
                </div>
            </div>
        </div>

        <!-- Advanced Analytics Page -->
        <div id="advanced-analytics" class="page">
            <h2>Advanced Analytics</h2>
            
            <!-- Product Recommendations -->
            <div class="chart-container">
                <h3>Product Recommendations</h3>
                <div class="controls">
                    <select id="customer-select">
                        <option value="">Select Customer</option>
                    </select>
                    <button onclick="getProductRecommendations()">Get Recommendations</button>
                </div>
                <div id="recommendations-results"></div>
            </div>
            
            <!-- Churn Prediction -->
            <div class="chart-container">
                <h3>Churn Prediction & Driver Analysis</h3>
                <div class="controls">
                    <select id="churn-customer-select">
                        <option value="">Select Customer</option>
                    </select>
                    <button onclick="predictChurn()">Predict Churn</button>
                </div>
                <div id="churn-results"></div>
                <div id="driver-analysis"></div>
            </div>
            
            <!-- Customer Segmentation -->
            <div class="chart-container">
                <h3>Customer Segmentation (Premium vs Non-Premium)</h3>
                <div class="controls">
                    <button onclick="performCustomerSegmentation()">Perform Segmentation</button>
                </div>
                <div id="segmentation-results"></div>
                <canvas id="segmentation-chart"></canvas>
            </div>
            
            <!-- Survival Analysis -->
            <div class="chart-container">
                <h3>Survival Analysis - Renewal Prediction</h3>
                <div class="controls">
                    <button onclick="performSurvivalAnalysis()">Run Survival Analysis</button>
                </div>
                <div id="survival-results"></div>
                <canvas id="survival-chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let map;
        let charts = {};

        function showPage(pageId) {
            // Hide all pages
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            
            // Show selected page
            document.getElementById(pageId).classList.add('active');
            document.getElementById(pageId + '-btn').classList.add('active');
            
            // Initialize specific page content
            if (pageId === 'main-dashboard') {
                loadMainDashboard();
            } else if (pageId === 'advanced-analytics') {
                loadAdvancedAnalytics();
            }
        }

        function loadMainDashboard() {
            loadOverviewStats();
            loadProductOptions();
            loadDeploymentChart();
            loadTopCustomersCharts();
            loadTopProductsCharts();
            if (!map) initMap();
        }

        function loadAdvancedAnalytics() {
            loadCustomerOptions();
            loadChurnCustomerOptions();
        }

        // Main Dashboard Functions
        function loadOverviewStats() {
            fetch('/api/overview')
                .then(response => response.json())
                .then(data => {
                    const statsGrid = document.getElementById('stats-grid');
                    statsGrid.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-number">${data.total_entitlements}</div>
                            <div class="stat-label">Total Entitlements</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.total_customers}</div>
                            <div class="stat-label">Total Customers</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.total_products}</div>
                            <div class="stat-label">Total Products</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.quantity_sold}</div>
                            <div class="stat-label">Quantity Sold</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.quantity_activated}</div>
                            <div class="stat-label">Quantity Activated</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.quantity_in_use}</div>
                            <div class="stat-label">Quantity in Use</div>
                        </div>
                    `;
                });
        }

        function loadProductOptions() {
            fetch('/api/products')
                .then(response => response.json())
                .then(products => {
                    const selects = ['product-select', 'usage-product-select'];
                    selects.forEach(selectId => {
                        const select = document.getElementById(selectId);
                        select.innerHTML = '<option value="">Select Product</option>';
                        products.forEach(product => {
                            select.innerHTML += `<option value="${product.product_id}">${product.name}</option>`;
                        });
                    });
                });
        }

        function loadCustomerOptions() {
            fetch('/api/customers')
                .then(response => response.json())
                .then(customers => {
                    const select = document.getElementById('customer-select');
                    select.innerHTML = '<option value="">Select Customer</option>';
                    customers.forEach(customer => {
                        select.innerHTML += `<option value="${customer.customer_id}">${customer.name}</option>`;
                    });
                });
        }

        function loadChurnCustomerOptions() {
            fetch('/api/customers')
                .then(response => response.json())
                .then(customers => {
                    const select = document.getElementById('churn-customer-select');
                    select.innerHTML = '<option value="">Select Customer</option>';
                    customers.forEach(customer => {
                        select.innerHTML += `<option value="${customer.customer_id}">${customer.name}</option>`;
                    });
                });
        }

        // Advanced Analytics Functions
        function getProductRecommendations() {
            const customerId = document.getElementById('customer-select').value;
            if (!customerId) return;
            
            fetch(`/api/product-recommendations/${customerId}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('recommendations-results');
                    if (data.recommendations && data.recommendations.length > 0) {
                        let html = '<h4>Recommended Products:</h4><table><tr><th>Product</th><th>Confidence</th><th>Support</th></tr>';
                        data.recommendations.forEach(rec => {
                            html += `<tr><td>${rec.product_name}</td><td>${(rec.confidence * 100).toFixed(2)}%</td><td>${(rec.support * 100).toFixed(2)}%</td></tr>`;
                        });
                        html += '</table>';
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = '<p>No recommendations available for this customer.</p>';
                    }
                });
        }

        function predictChurn() {
            const customerId = document.getElementById('churn-customer-select').value;
            if (!customerId) return;
            
            fetch(`/api/churn-prediction/${customerId}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('churn-results');
                    const driverDiv = document.getElementById('driver-analysis');
                    
                    // Display churn prediction
                    const churnClass = data.churn_probability > 0.7 ? 'churn-high' : 
                                     data.churn_probability > 0.4 ? 'churn-medium' : 'churn-low';
                    
                    resultsDiv.innerHTML = `
                        <h4>Churn Prediction Results:</h4>
                        <div class="${churnClass}">
                            <p><strong>Churn Probability:</strong> ${(data.churn_probability * 100).toFixed(2)}%</p>
                            <p><strong>Risk Level:</strong> ${data.risk_level}</p>
                            <p><strong>Recommendation:</strong> ${data.recommendation}</p>
                        </div>
                    `;
                    
                    // Display driver analysis
                    if (data.driver_analysis) {
                        let driverHtml = '<h4>Driver Analysis (Feature Importance):</h4><table><tr><th>Feature</th><th>Importance</th><th>Impact</th></tr>';
                        data.driver_analysis.forEach(driver => {
                            driverHtml += `<tr><td>${driver.feature}</td><td>${driver.importance.toFixed(4)}</td><td>${driver.impact}</td></tr>`;
                        });
                        driverHtml += '</table>';
                        driverDiv.innerHTML = driverHtml;
                    }
                });
        }

        function performCustomerSegmentation() {
            fetch('/api/customer-segmentation')
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('segmentation-results');
                    const chartCanvas = document.getElementById('segmentation-chart');
                    
                    // Display segmentation results
                    let html = '<h4>Customer Segments:</h4><table><tr><th>Segment</th><th>Count</th><th>Characteristics</th></tr>';
                    data.segments.forEach(segment => {
                        html += `<tr class="segment-${segment.name.toLowerCase()}"><td>${segment.name}</td><td>${segment.count}</td><td>${segment.characteristics}</td></tr>`;
                    });
                    html += '</table>';
                    resultsDiv.innerHTML = html;
                    
                    // Create segmentation chart
                    const ctx = chartCanvas.getContext('2d');
                    if (charts.segmentationChart) charts.segmentationChart.destroy();
                    
                    charts.segmentationChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: data.segments.map(s => s.name),
                            datasets: [{
                                data: data.segments.map(s => s.count),
                                backgroundColor: ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
                            }]
                        },
                        options: { responsive: true }
                    });
                });
        }

        function performSurvivalAnalysis() {
            fetch('/api/survival-analysis')
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('survival-results');
                    const chartCanvas = document.getElementById('survival-chart');
                    
                    // Display survival analysis results
                    resultsDiv.innerHTML = `
                        <h4>Survival Analysis Results:</h4>
                        <p><strong>Overall Renewal Rate:</strong> ${(data.overall_renewal_rate * 100).toFixed(2)}%</p>
                        <p><strong>Median Survival Time:</strong> ${data.median_survival_time} days</p>
                        <p><strong>High Risk Customers:</strong> ${data.high_risk_count}</p>
                    `;
                    
                    // Create survival chart
                    const ctx = chartCanvas.getContext('2d');
                    if (charts.survivalChart) charts.survivalChart.destroy();
                    
                    charts.survivalChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.time_periods,
                            datasets: [{
                                label: 'Survival Probability',
                                data: data.survival_probabilities,
                                borderColor: '#e74c3c',
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: { beginAtZero: true, max: 1 }
                            }
                        }
                    });
                });
        }

        // Initialize dashboard
        window.onload = function() {
            loadMainDashboard();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# API Routes for Main Dashboard
@app.route('/api/overview')
def overview():
    """Get overview statistics"""
    try:
        stats = {
            'total_entitlements': len(data['entitlements']),
            'total_customers': len(data['customers']),
            'total_products': len(data['products']),
            'quantity_sold': int(data['entitlements']['purchase_quantity'].sum()),
            'quantity_activated': int(data['activations']['quantity'].sum()),
            'quantity_in_use': len(data['users'])
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/products')
def get_products():
    """Get list of products"""
    try:
        products = data['products'][['product_id', 'name']].to_dict('records')
        return jsonify(products)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/customers')
def get_customers():
    """Get list of customers"""
    try:
        customers = data['customers'][['customer_id', 'name']].to_dict('records')
        return jsonify(customers)
    except Exception as e:
        return jsonify({'error': str(e)})

# New API Routes for Advanced Analytics
@app.route('/api/product-recommendations/<customer_id>')
def product_recommendations(customer_id):
    """Get product recommendations for a customer using Apriori algorithm"""
    try:
        if 'association_rules' in models and not models['association_rules'].empty:
            # Get customer's current products
            customer_products = data['entitlements'][
                data['entitlements']['customer_id'] == customer_id
            ]['product_id'].unique()
            
            # Find recommendations based on association rules
            recommendations = []
            for _, rule in models['association_rules'].iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                # Check if customer has antecedent products
                if any(prod in customer_products for prod in antecedents):
                    for consequent in consequents:
                        if consequent not in customer_products:
                            product_name = data['products'][
                                data['products']['product_id'] == consequent
                            ]['name'].iloc[0]
                            
                            recommendations.append({
                                'product_name': product_name,
                                'confidence': rule['confidence'],
                                'support': rule['support']
                            })
            
            # Sort by confidence and return top 5
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            return jsonify({'recommendations': recommendations[:5]})
        else:
            return jsonify({'recommendations': []})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/churn-prediction/<customer_id>')
def churn_prediction(customer_id):
    """Predict churn for a customer and provide driver analysis"""
    try:
        customer_data = data['customer_summary'][
            data['customer_summary']['customer_id'] == customer_id
        ]
        
        if customer_data.empty:
            return jsonify({'error': 'Customer not found'})
        
        # Prepare features for prediction
        features = customer_data[models['churn_features']].values
        features_scaled = models['churn_scaler'].transform(features)
        
        # Predict churn probability
        churn_prob = models['churn_model'].predict_proba(features_scaled)[0][1]
        
        # Determine risk level and recommendation
        if churn_prob > 0.7:
            risk_level = "High"
            recommendation = "Immediate intervention required: Offer discounts, personalized support, feature training"
        elif churn_prob > 0.4:
            risk_level = "Medium"
            recommendation = "Proactive engagement: Regular check-ins, usage optimization suggestions"
        else:
            risk_level = "Low"
            recommendation = "Maintain relationship: Continue current engagement strategy"
        
        # Driver analysis - feature importance
        feature_importance = models['churn_model'].coef_[0]
        driver_analysis = []
        
        for i, feature in enumerate(models['churn_features']):
            importance = abs(feature_importance[i])
            impact = "Positive" if feature_importance[i] > 0 else "Negative"
            driver_analysis.append({
                'feature': feature.replace('_', ' ').title(),
                'importance': importance,
                'impact': impact
            })
        
        # Sort by importance
        driver_analysis.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'churn_probability': churn_prob,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'driver_analysis': driver_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/customer-segmentation')
def customer_segmentation():
    """Perform customer segmentation using clustering"""
    try:
        # Prepare features for clustering
        features = ['total_purchased', 'total_contract_value', 'total_activated']
        X = data['customer_summary'][features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to customer data
        data['customer_summary']['cluster'] = clusters
        
        # Analyze clusters
        segments = []
        for i in range(4):
            cluster_data = data['customer_summary'][data['customer_summary']['cluster'] == i]
            
            if len(cluster_data) > 0:
                avg_purchase = cluster_data['total_purchased'].mean()
                avg_value = cluster_data['total_contract_value'].mean()
                avg_activated = cluster_data['total_activated'].mean()
                
                # Determine segment type
                if avg_value > 50000 and avg_purchase > 100:
                    segment_name = "Premium"
                    characteristics = f"High-value customers (${avg_value:,.0f} avg contract, {avg_purchase:.0f} avg purchases)"
                elif avg_value > 20000:
                    segment_name = "Enterprise"
                    characteristics = f"Medium-value customers (${avg_value:,.0f} avg contract, {avg_purchase:.0f} avg purchases)"
                elif avg_activated > 50:
                    segment_name = "Active"
                    characteristics = f"High-usage customers ({avg_activated:.0f} avg activations)"
                else:
                    segment_name = "Standard"
                    characteristics = f"Standard customers (${avg_value:,.0f} avg contract)"
                
                segments.append({
                    'name': segment_name,
                    'count': len(cluster_data),
                    'characteristics': characteristics
                })
        
        return jsonify({'segments': segments})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/survival-analysis')
def survival_analysis():
    """Perform survival analysis for renewal prediction"""
    try:
        # Prepare survival data
        survival_data = data['customer_summary'].copy()
        
        # Calculate time to event (churn or censoring)
        survival_data['time_to_event'] = survival_data['days_since_last_activation']
        survival_data['event_occurred'] = (survival_data['status'] == 'Inactive') | (survival_data['status'] == 'Suspended')
        
        # Remove infinite values
        survival_data = survival_data[survival_data['time_to_event'].notna() & (survival_data['time_to_event'] < 1000)]
        
        # Simple survival analysis (since we don't have lifelines)
        # Calculate renewal rate and risk metrics
        overall_renewal_rate = 1 - survival_data['event_occurred'].mean()
        high_risk_count = len(survival_data[survival_data['time_to_event'] > 90])
        
        # Create time periods for visualization
        time_periods = list(range(0, 365, 30))
        survival_probabilities = []
        
        for t in time_periods:
            # Calculate survival probability at time t
            at_risk = survival_data[survival_data['time_to_event'] >= t]
            if len(at_risk) > 0:
                survived = at_risk[~at_risk['event_occurred']]
                prob = len(survived) / len(at_risk)
                survival_probabilities.append(prob)
            else:
                survival_probabilities.append(0)
        
        return jsonify({
            'overall_renewal_rate': overall_renewal_rate,
            'median_survival_time': 180,  # Approximate
            'high_risk_count': high_risk_count,
            'time_periods': time_periods,
            'survival_probabilities': survival_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_data()
    app.run(debug=True, host='0.0.0.0', port=5000)


