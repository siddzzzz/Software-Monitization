from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store data
data = {}

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
        if 'renewal_date' in data['renewals'].columns:
            data['renewals']['renewal_date'] = pd.to_datetime(data['renewals']['renewal_date'])
        
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced License Management Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .header { background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); color: #333; padding: 1rem; text-align: center; box-shadow: 0 2px 20px rgba(0,0,0,0.1); }
        .nav { background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); padding: 1rem; display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .nav button { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white; border: none; padding: 12px 24px; cursor: pointer; 
            border-radius: 25px; font-weight: 600; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .nav button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }
        .nav button.active { background: linear-gradient(45deg, #ff6b6b, #ee5a24); }
        .container { padding: 20px; max-width: 1400px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { 
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); text-align: center; backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2); transition: transform 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-number { font-size: 2.5em; font-weight: bold; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-label { color: #666; margin-top: 10px; font-weight: 500; }
        .page { display: none; }
        .page.active { display: block; }
        .chart-container { 
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 20px; backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .controls { margin-bottom: 20px; display: flex; gap: 15px; flex-wrap: wrap; }
        .controls select, .controls button { 
            padding: 12px 20px; border-radius: 10px; border: 1px solid #ddd; 
            background: white; font-weight: 500; cursor: pointer; transition: all 0.3s ease;
        }
        .controls button { background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; }
        .controls button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
        #map { height: 500px; width: 100%; border-radius: 10px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .loading { text-align: center; padding: 40px; color: white; }
        .error { background: #ff6b6b; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .success { background: #51cf66; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .ml-results { background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; margin: 15px 0; }
        .metric-box { display: inline-block; background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 10px; min-width: 150px; text-align: center; }
        .recommendations { background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin: 15px 0; }
        .recommendation-item { padding: 15px; margin: 10px 0; background: #e3f2fd; border-radius: 10px; border-left: 4px solid #2196f3; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; font-weight: 600; }
        .cluster-info { background: #fff3e0; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }
        .survival-plot { width: 100%; max-width: 800px; margin: 20px auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Advanced License Management & Analytics Dashboard</h1>
        <p>Comprehensive insights with machine learning and predictive analytics</p>
    </div>
    
    <div class="nav">
        <button onclick="showPage('overview')" class="active" id="overview-btn">Dashboard Overview</button>
        <button onclick="showPage('recommendations')" id="recommendations-btn">AI Recommendations</button>
        <button onclick="showPage('churn-analysis')" id="churn-analysis-btn">Churn Analysis</button>
        <button onclick="showPage('customer-segmentation')" id="customer-segmentation-btn">Customer Segmentation</button>
        <button onclick="showPage('survival-analysis')" id="survival-analysis-btn">Survival Analysis</button>
        <button onclick="showPage('world-map')" id="world-map-btn">Geographic Analysis</button>
    </div>

    <div class="container">
        <!-- Overview Page -->
        <div id="overview" class="page active">
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be loaded here -->
            </div>
            <div class="grid-3">
                <div class="chart-container">
                    <h3>Product Performance</h3>
                    <canvas id="product-performance-chart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Customer Activity Trends</h3>
                    <canvas id="customer-trends-chart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Deployment Status</h3>
                    <canvas id="deployment-chart"></canvas>
                </div>
            </div>
        </div>

        <!-- AI Recommendations Page -->
        <div id="recommendations" class="page">
            <div class="controls">
                <select id="recommendation-type">
                    <option value="customer">Customer Recommendations</option>
                    <option value="vendor">Vendor Recommendations</option>
                </select>
                <button onclick="generateRecommendations()">Generate AI Recommendations</button>
            </div>
            <div id="recommendations-content">
                <div class="loading" id="rec-loading" style="display: none;">
                    <p>🤖 AI is analyzing patterns and generating recommendations...</p>
                </div>
                <div id="rec-results"></div>
            </div>
        </div>

        <!-- Churn Analysis Page -->
        <div id="churn-analysis" class="page">
            <div class="controls">
                <button onclick="runChurnAnalysis()">Run Churn Prediction Model</button>
                <button onclick="runDriverAnalysis()">Driver Analysis</button>
            </div>
            <div id="churn-content">
                <div class="loading" id="churn-loading" style="display: none;">
                    <p>🔍 Analyzing customer behavior and predicting churn...</p>
                </div>
                <div id="churn-results"></div>
            </div>
        </div>

        <!-- Customer Segmentation Page -->
        <div id="customer-segmentation" class="page">
            <div class="controls">
                <select id="segmentation-type">
                    <option value="premium">Premium vs Non-Premium</option>
                    <option value="clustering">Advanced Clustering</option>
                </select>
                <button onclick="runSegmentation()">Run Customer Segmentation</button>
            </div>
            <div id="segmentation-content">
                <div class="loading" id="seg-loading" style="display: none;">
                    <p>🎯 Segmenting customers using advanced algorithms...</p>
                </div>
                <div id="seg-results"></div>
            </div>
        </div>

        <!-- Survival Analysis Page -->
        <div id="survival-analysis" class="page">
            <div class="controls">
                <button onclick="runSurvivalAnalysis()">Run Survival Analysis</button>
            </div>
            <div id="survival-content">
                <div class="loading" id="survival-loading" style="display: none;">
                    <p>📈 Analyzing customer lifecycle and renewal patterns...</p>
                </div>
                <div id="survival-results"></div>
            </div>
        </div>

        <!-- World Map Page -->
        <div id="world-map" class="page">
            <div class="chart-container">
                <h3>Global User Distribution</h3>
                <div id="map"></div>
            </div>
        </div>
    </div>

    <script>
        let map;
        let charts = {};

        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            
            document.getElementById(pageId).classList.add('active');
            document.getElementById(pageId + '-btn').classList.add('active');
            
            if (pageId === 'world-map' && !map) {
                initMap();
            }
            if (pageId === 'overview') {
                loadDashboardCharts();
            }
        }

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
                            <div class="stat-label">Licenses Sold</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.quantity_activated}</div>
                            <div class="stat-label">Licenses Activated</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${(data.activation_rate * 100).toFixed(1)}%</div>
                            <div class="stat-label">Activation Rate</div>
                        </div>
                    `;
                });
        }

        function loadDashboardCharts() {
            // Product Performance Chart
            fetch('/api/product-performance')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('product-performance-chart').getContext('2d');
                    if (charts.productPerformance) charts.productPerformance.destroy();
                    
                    charts.productPerformance = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.products,
                            datasets: [{
                                label: 'Revenue',
                                data: data.revenue,
                                backgroundColor: 'rgba(102, 126, 234, 0.6)'
                            }]
                        },
                        options: { responsive: true }
                    });
                });

            // Customer Trends Chart
            fetch('/api/customer-trends')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('customer-trends-chart').getContext('2d');
                    if (charts.customerTrends) charts.customerTrends.destroy();
                    
                    charts.customerTrends = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.months,
                            datasets: [{
                                label: 'New Customers',
                                data: data.new_customers,
                                borderColor: 'rgba(238, 90, 36, 1)',
                                fill: false
                            }]
                        },
                        options: { responsive: true }
                    });
                });

            // Deployment Chart
            fetch('/api/deployment-status')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('deployment-chart').getContext('2d');
                    if (charts.deployment) charts.deployment.destroy();
                    
                    charts.deployment = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: ['Deployed', 'Not Deployed'],
                            datasets: [{
                                data: [data.deployed, data.not_deployed],
                                backgroundColor: ['rgba(81, 207, 102, 0.8)', 'rgba(255, 107, 107, 0.8)']
                            }]
                        },
                        options: { responsive: true }
                    });
                });
        }

        function generateRecommendations() {
            const type = document.getElementById('recommendation-type').value;
            const loading = document.getElementById('rec-loading');
            const results = document.getElementById('rec-results');
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            fetch(`/api/recommendations/${type}`)
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    displayRecommendations(data, type);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
        }

        function displayRecommendations(data, type) {
            const results = document.getElementById('rec-results');
            let html = `<div class="recommendations">
                <h3>AI-Powered ${type === 'customer' ? 'Customer' : 'Vendor'} Recommendations</h3>`;
            
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    html += `<div class="recommendation-item">
                        <h4>${rec.title}</h4>
                        <p>${rec.description}</p>
                        <p><strong>Confidence:</strong> ${rec.confidence}%</p>
                    </div>`;
                });
            } else {
                html += '<p>No recommendations available at this time.</p>';
            }
            
            html += '</div>';
            results.innerHTML = html;
        }

        function runChurnAnalysis() {
            const loading = document.getElementById('churn-loading');
            const results = document.getElementById('churn-results');
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            fetch('/api/churn-analysis')
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    displayChurnResults(data);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
        }

        function runDriverAnalysis() {
            const loading = document.getElementById('churn-loading');
            const results = document.getElementById('churn-results');
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            fetch('/api/driver-analysis')
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    displayDriverResults(data);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
        }

        function displayChurnResults(data) {
            const results = document.getElementById('churn-results');
            let html = `<div class="ml-results">
                <h3>Churn Prediction Results</h3>
                <div class="metric-box">
                    <h4>Model Accuracy</h4>
                    <p>${(data.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div class="metric-box">
                    <h4>High Risk Customers</h4>
                    <p>${data.high_risk_count}</p>
                </div>
                <div class="metric-box">
                    <h4>Potential Revenue at Risk</h4>
                    <p>$${data.revenue_at_risk.toLocaleString()}</p>
                </div>
            </div>`;
            
            if (data.high_risk_customers && data.high_risk_customers.length > 0) {
                html += `<div class="recommendations">
                    <h4>High-Risk Customers Requiring Immediate Attention</h4>`;
                data.high_risk_customers.forEach(customer => {
                    html += `<div class="recommendation-item">
                        <h5>${customer.name}</h5>
                        <p>Churn Probability: ${(customer.churn_probability * 100).toFixed(2)}%</p>
                        <p>Recommended Action: ${customer.recommendation}</p>
                    </div>`;
                });
                html += '</div>';
            }
            
            results.innerHTML = html;
        }

        function displayDriverResults(data) {
            const results = document.getElementById('churn-results');
            let html = `<div class="ml-results">
                <h3>Driver Analysis Results</h3>
                <h4>Key Factors Influencing Renewal Decisions</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Factor</th>
                            <th>Importance Score</th>
                            <th>Impact on Renewal</th>
                        </tr>
                    </thead>
                    <tbody>`;
            
            if (data.drivers) {
                data.drivers.forEach(driver => {
                    html += `<tr>
                        <td>${driver.feature}</td>
                        <td>${driver.importance.toFixed(3)}</td>
                        <td>${driver.impact}</td>
                    </tr>`;
                });
            }
            
            html += `</tbody></table></div>
                <div class="recommendations">
                    <h4>Strategic Recommendations</h4>`;
            
            if (data.recommendations) {
                data.recommendations.forEach(rec => {
                    html += `<div class="recommendation-item">
                        <p>${rec}</p>
                    </div>`;
                });
            }
            
            html += '</div>';
            results.innerHTML = html;
        }

        function runSegmentation() {
            const type = document.getElementById('segmentation-type').value;
            const loading = document.getElementById('seg-loading');
            const results = document.getElementById('seg-results');
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            fetch(`/api/segmentation/${type}`)
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    displaySegmentationResults(data, type);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
        }

        function displaySegmentationResults(data, type) {
            const results = document.getElementById('seg-results');
            let html = `<div class="ml-results">
                <h3>${type === 'premium' ? 'Premium vs Non-Premium' : 'Advanced Clustering'} Results</h3>`;
            
            if (data.segments) {
                data.segments.forEach((segment, index) => {
                    html += `<div class="cluster-info">
                        <h4>Segment ${index + 1}: ${segment.name}</h4>
                        <p><strong>Size:</strong> ${segment.size} customers</p>
                        <p><strong>Characteristics:</strong> ${segment.description}</p>
                        <p><strong>Average Revenue:</strong> $${segment.avg_revenue.toLocaleString()}</p>
                    </div>`;
                });
            }
            
            html += '</div>';
            results.innerHTML = html;
        }

        function runSurvivalAnalysis() {
            const loading = document.getElementById('survival-loading');
            const results = document.getElementById('survival-results');
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            fetch('/api/survival-analysis')
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    displaySurvivalResults(data);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
        }

        function displaySurvivalResults(data) {
            const results = document.getElementById('survival-results');
            let html = `<div class="ml-results">
                <h3>Customer Lifecycle & Survival Analysis</h3>
                <div class="metric-box">
                    <h4>Median Customer Lifespan</h4>
                    <p>${data.median_survival} days</p>
                </div>
                <div class="metric-box">
                    <h4>1-Year Survival Rate</h4>
                    <p>${(data.one_year_survival * 100).toFixed(2)}%</p>
                </div>
            </div>`;
            
            if (data.insights) {
                html += `<div class="recommendations">
                    <h4>Key Insights</h4>`;
                data.insights.forEach(insight => {
                    html += `<div class="recommendation-item">
                        <p>${insight}</p>
                    </div>`;
                });
                html += '</div>';
            }
            
            results.innerHTML = html;
        }

        function initMap() {
            map = L.map('map').setView([20, 0], 2);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            
            fetch('/api/user-locations')
                .then(response => response.json())
                .then(locations => {
                    locations.forEach(loc => {
                        L.circleMarker([loc.latitude, loc.longitude], {
                            radius: Math.log(loc.user_count + 1) * 3,
                            fillColor: '#667eea',
                            color: '#764ba2',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.7
                        }).addTo(map).bindPopup(`${loc.city}, ${loc.country}<br>Users: ${loc.user_count}`);
                    });
                });
        }

        // Initialize dashboard
        window.onload = function() {
            loadOverviewStats();
            loadDashboardCharts();
        };
    </script>
</body>
</html>
'''

# New API endpoints for advanced analytics

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/overview')
def overview():
    """Get overview statistics"""
    try:
        total_sold = int(data['entitlements']['purchase_quantity'].sum())
        total_activated = int(data['activations']['quantity'].sum())
        activation_rate = total_activated / total_sold if total_sold > 0 else 0
        
        stats = {
            'total_entitlements': len(data['entitlements']),
            'total_customers': len(data['customers']),
            'total_products': len(data['products']),
            'quantity_sold': total_sold,
            'quantity_activated': total_activated,
            'activation_rate': activation_rate
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/product-performance')
def product_performance():
    """Get product performance metrics"""
    try:
        # Calculate revenue per product (assuming price data)
        product_revenue = data['entitlements'].groupby('product_id')['purchase_quantity'].sum()
        product_names = data['products'].set_index('product_id')['name']
        
        top_products = product_revenue.head(10)
        products = [product_names.get(pid, f'Product {pid}')[:20] for pid in top_products.index]
        revenue = top_products.values.tolist()
        
        return jsonify({'products': products, 'revenue': revenue})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/customer-trends')
def customer_trends():
    """Get customer acquisition trends"""
    try:
        # Group by month
        monthly_customers = data['entitlements'].groupby(
            data['entitlements']['purchase_date'].dt.to_period('M')
        )['customer_id'].nunique()
        
        months = [str(period) for period in monthly_customers.index]
        new_customers = monthly_customers.values.tolist()
        
        return jsonify({'months': months, 'new_customers': new_customers})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/deployment-status')
def deployment_status():
    """Get deployment status overview"""
    try:
        total_sold = data['entitlements']['purchase_quantity'].sum()
        total_activated = data['activations']['quantity'].sum()
        
        deployed = int(total_activated)
        not_deployed = int(total_sold - total_activated)
        
        return jsonify({'deployed': deployed, 'not_deployed': not_deployed})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/recommendations/<rec_type>')
def generate_recommendations(rec_type):
    """Generate AI recommendations using collaborative filtering"""
    try:
        recommendations = []
        
        if rec_type == 'customer':
            # Customer product recommendations based on purchase patterns
            customer_product_matrix = data['entitlements'].pivot_table(
                index='customer_id', 
                columns='product_id', 
                values='purchase_quantity', 
                fill_value=0
            )
            
            # Simple collaborative filtering
            for customer_id in customer_product_matrix.index[:5]:  # Top 5 customers
                customer_products = customer_product_matrix.loc[customer_id]
                purchased_products = customer_products[customer_products > 0].index
                
                # Find similar customers
                similarity_scores = {}
                for other_customer in customer_product_matrix.index:
                    if other_customer != customer_id:
                        other_products = customer_product_matrix.loc[other_customer]
                        # Simple cosine similarity
                        dot_product = np.dot(customer_products, other_products)
                        norm_a = np.linalg.norm(customer_products)
                        norm_b = np.linalg.norm(other_products)
                        if norm_a > 0 and norm_b > 0:
                            similarity_scores[other_customer] = dot_product / (norm_a * norm_b)
                
                if similarity_scores:
                    most_similar = max(similarity_scores, key=similarity_scores.get)
                    similar_customer_products = customer_product_matrix.loc[most_similar]
                    
                    # Recommend products that similar customer bought but current customer hasn't
                    recommendations.append({
                        'title': f'Recommendation for Customer {customer_id}',
                        'description': f'Based on similar customers, consider promoting products not yet purchased',
                        'confidence': int(similarity_scores[most_similar] * 100)
                    })
        
        else:  # vendor recommendations
            # Vendor recommendations based on product performance
            product_performance = data['entitlements'].groupby('product_id').agg({
                'purchase_quantity': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            top_products = product_performance.nlargest(3, 'purchase_quantity')
            
            for _, product in top_products.iterrows():
                product_name = data['products'][data['products']['product_id'] == product['product_id']]['name'].iloc[0]
                recommendations.append({
                    'title': f'Promote {product_name}',
                    'description': f'High-performing product with {product["purchase_quantity"]} units sold to {product["customer_id"]} customers',
                    'confidence': 85
                })
        
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/churn-analysis')
def churn_analysis():
    """Run churn prediction analysis"""
    try:
        # Create features for churn prediction
        customer_features = data['entitlements'].groupby('customer_id').agg({
            'purchase_quantity': 'sum',
            'purchase_date': ['min', 'max', 'count']
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'total_quantity', 'first_purchase', 'last_purchase', 'purchase_count']
        
        # Calculate days since last purchase
        customer_features['days_since_last_purchase'] = (
            datetime.now() - pd.to_datetime(customer_features['last_purchase'])
        ).dt.days
        
        # Calculate customer lifetime (days between first and last purchase)
        customer_features['customer_lifetime'] = (
            pd.to_datetime(customer_features['last_purchase']) - 
            pd.to_datetime(customer_features['first_purchase'])
        ).dt.days
        
        # Define churn based on no purchase in last 180 days
        customer_features['is_churned'] = (customer_features['days_since_last_purchase'] > 180).astype(int)
        
        # Prepare features for ML model
        feature_columns = ['total_quantity', 'purchase_count', 'days_since_last_purchase', 'customer_lifetime']
        X = customer_features[feature_columns].fillna(0)
        y = customer_features['is_churned']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Predict churn probability for all customers
        churn_probabilities = model.predict_proba(X)[:, 1]
        customer_features['churn_probability'] = churn_probabilities
        
        # High risk customers
        high_risk = customer_features[customer_features['churn_probability'] > 0.7]
        high_risk_customers = []
        
        for _, customer in high_risk.head(5).iterrows():
            customer_name = data['customers'][data['customers']['customer_id'] == customer['customer_id']]['name'].iloc[0]
            
            recommendation = "Immediate outreach recommended"
            if customer['days_since_last_purchase'] > 120:
                recommendation = "Re-engagement campaign with special offers"
            elif customer['purchase_count'] == 1:
                recommendation = "Onboarding improvement and follow-up"
            
            high_risk_customers.append({
                'name': customer_name,
                'churn_probability': customer['churn_probability'],
                'recommendation': recommendation
            })
        
        # Calculate potential revenue at risk
        revenue_at_risk = high_risk['total_quantity'].sum() * 100  # Assuming $100 per unit
        
        return jsonify({
            'accuracy': accuracy,
            'high_risk_count': len(high_risk),
            'revenue_at_risk': revenue_at_risk,
            'high_risk_customers': high_risk_customers
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/driver-analysis')
def driver_analysis():
    """Run driver analysis using logistic regression"""
    try:
        # Create renewal dataset
        renewal_features = data['entitlements'].groupby('customer_id').agg({
            'purchase_quantity': 'sum',
            'purchase_date': ['min', 'max', 'count']
        }).reset_index()
        
        renewal_features.columns = ['customer_id', 'total_quantity', 'first_purchase', 'last_purchase', 'purchase_count']
        
        # Calculate features
        renewal_features['days_since_last_purchase'] = (
            datetime.now() - pd.to_datetime(renewal_features['last_purchase'])
        ).dt.days
        
        renewal_features['customer_lifetime'] = (
            pd.to_datetime(renewal_features['last_purchase']) - 
            pd.to_datetime(renewal_features['first_purchase'])
        ).dt.days
        
        # Get activation data
        customer_activations = data['activations'].merge(
            data['entitlements'][['entitlement_id', 'customer_id']], 
            on='entitlement_id'
        ).groupby('customer_id')['quantity'].sum().reset_index()
        
        renewal_features = renewal_features.merge(
            customer_activations, on='customer_id', how='left'
        ).fillna(0)
        renewal_features.rename(columns={'quantity': 'activation_quantity'}, inplace=True)
        
        # Calculate activation rate
        renewal_features['activation_rate'] = renewal_features['activation_quantity'] / renewal_features['total_quantity']
        renewal_features['activation_rate'] = renewal_features['activation_rate'].fillna(0)
        
        # Define renewal target (renewed if purchased within last 365 days)
        renewal_features['will_renew'] = (renewal_features['days_since_last_purchase'] <= 365).astype(int)
        
        # Prepare features for logistic regression
        feature_columns = ['total_quantity', 'purchase_count', 'customer_lifetime', 'activation_rate']
        X = renewal_features[feature_columns].fillna(0)
        y = renewal_features['will_renew']
        
        # Train logistic regression model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_scaled, y)
        
        # Get feature importance (coefficients)
        coefficients = lr_model.coef_[0]
        feature_importance = list(zip(feature_columns, coefficients))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Interpret coefficients
        drivers = []
        recommendations = []
        
        for feature, coef in feature_importance:
            impact = "Positive" if coef > 0 else "Negative"
            
            if feature == 'activation_rate' and abs(coef) > 0.5:
                recommendations.append("Focus on improving software activation and user onboarding")
            elif feature == 'total_quantity' and coef > 0:
                recommendations.append("Encourage larger initial purchases through volume discounts")
            elif feature == 'customer_lifetime' and coef > 0:
                recommendations.append("Invest in long-term customer relationship building")
            elif feature == 'purchase_count' and coef > 0:
                recommendations.append("Develop strategies to encourage repeat purchases")
            
            drivers.append({
                'feature': feature.replace('_', ' ').title(),
                'importance': abs(coef),
                'impact': f"{impact} impact on renewal likelihood"
            })
        
        return jsonify({
            'drivers': drivers,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/segmentation/<seg_type>')
def customer_segmentation(seg_type):
    """Run customer segmentation analysis"""
    try:
        segments = []
        
        if seg_type == 'premium':
            # Premium vs Non-Premium segmentation
            customer_stats = data['entitlements'].groupby('customer_id').agg({
                'purchase_quantity': 'sum'
            }).reset_index()
            
            # Define premium threshold (top 20% by quantity)
            threshold = customer_stats['purchase_quantity'].quantile(0.8)
            
            premium_customers = customer_stats[customer_stats['purchase_quantity'] >= threshold]
            non_premium_customers = customer_stats[customer_stats['purchase_quantity'] < threshold]
            
            segments.append({
                'name': 'Premium Customers',
                'size': len(premium_customers),
                'description': f'High-value customers with {threshold:.0f}+ licenses',
                'avg_revenue': premium_customers['purchase_quantity'].mean() * 100
            })
            
            segments.append({
                'name': 'Standard Customers',
                'size': len(non_premium_customers),
                'description': f'Regular customers with <{threshold:.0f} licenses',
                'avg_revenue': non_premium_customers['purchase_quantity'].mean() * 100
            })
        
        else:  # clustering
            # Advanced clustering using K-means
            customer_features = data['entitlements'].groupby('customer_id').agg({
                'purchase_quantity': 'sum',
                'purchase_date': 'count'
            }).reset_index()
            
            customer_features.columns = ['customer_id', 'total_quantity', 'purchase_frequency']
            
            # Get activation data
            customer_activations = data['activations'].merge(
                data['entitlements'][['entitlement_id', 'customer_id']], 
                on='entitlement_id'
            ).groupby('customer_id')['quantity'].sum().reset_index()
            
            customer_features = customer_features.merge(
                customer_activations, on='customer_id', how='left'
            ).fillna(0)
            customer_features.rename(columns={'quantity': 'activation_quantity'}, inplace=True)
            
            # Prepare features for clustering
            X = customer_features[['total_quantity', 'purchase_frequency', 'activation_quantity']]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            customer_features['cluster'] = clusters
            
            # Analyze clusters
            for i in range(3):
                cluster_data = customer_features[customer_features['cluster'] == i]
                
                avg_quantity = cluster_data['total_quantity'].mean()
                avg_frequency = cluster_data['purchase_frequency'].mean()
                avg_activation = cluster_data['activation_quantity'].mean()
                
                if avg_quantity > customer_features['total_quantity'].mean():
                    name = "High-Volume Customers"
                    description = f"Large purchasers with high activation rates"
                elif avg_frequency > customer_features['purchase_frequency'].mean():
                    name = "Frequent Buyers"
                    description = f"Regular repeat customers"
                else:
                    name = "Occasional Users"
                    description = f"Infrequent, low-volume customers"
                
                segments.append({
                    'name': name,
                    'size': len(cluster_data),
                    'description': description,
                    'avg_revenue': avg_quantity * 100
                })
        
        return jsonify({'segments': segments})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/survival-analysis')
def survival_analysis():
    """Run survival analysis on customer lifecycle"""
    try:
        # Prepare survival data
        customer_data = data['entitlements'].groupby('customer_id').agg({
            'purchase_date': ['min', 'max']
        }).reset_index()
        
        customer_data.columns = ['customer_id', 'first_purchase', 'last_purchase']
        
        # Calculate duration (customer lifetime in days)
        customer_data['duration'] = (
            pd.to_datetime(customer_data['last_purchase']) - 
            pd.to_datetime(customer_data['first_purchase'])
        ).dt.days + 1  # Add 1 to avoid zero duration
        
        # Event indicator (1 if churned, 0 if still active)
        # Assume churned if no purchase in last 180 days
        days_since_last = (datetime.now() - pd.to_datetime(customer_data['last_purchase'])).dt.days
        customer_data['event'] = (days_since_last > 180).astype(int)
        
        # Simple survival analysis calculations
        durations = customer_data['duration'].values
        events = customer_data['event'].values
        
        # Calculate median survival time
        median_survival = np.median(durations[events == 1]) if np.any(events == 1) else np.median(durations)
        
        # Estimate 1-year survival rate
        one_year_customers = customer_data[customer_data['duration'] >= 365]
        one_year_survival = len(one_year_customers[one_year_customers['event'] == 0]) / len(one_year_customers) if len(one_year_customers) > 0 else 0
        
        insights = [
            f"Average customer lifespan is {median_survival:.0f} days",
            f"Customers who stay beyond 1 year have {(one_year_survival * 100):.1f}% retention rate",
            "Early intervention within first 90 days is critical for retention"
        ]
        
        if median_survival < 365:
            insights.append("Customer retention needs immediate attention - median lifespan is under 1 year")
        
        return jsonify({
            'median_survival': int(median_survival),
            'one_year_survival': one_year_survival,
            'insights': insights
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/user-locations')
def user_locations():
    """Get user locations for world map"""
    try:
        location_counts = data['users'].groupby(['city', 'country', 'latitude', 'longitude']).size().reset_index(name='user_count')
        locations = location_counts.to_dict('records')
        return jsonify(locations)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_data()
    app.run(debug=True, host='0.0.0.0', port=5000)