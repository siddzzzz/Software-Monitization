from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import json
from datetime import datetime
import os

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
    <title>License Management Dashboard</title>
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
        .container { padding: 20px; max-width: 1200px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .page { display: none; }
        .page.active { display: block; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .controls { margin-bottom: 20px; }
        .controls select { padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
        #map { height: 500px; width: 100%; }
        .top-list { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .top-item { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>License Management Dashboard</h1>
    </div>
    
    <div class="nav">
        <button onclick="showPage('overview')" class="active" id="overview-btn">Overview</button>
        <button onclick="showPage('product-trends')" id="product-trends-btn">Product Trends</button>
        <button onclick="showPage('usage-trends')" id="usage-trends-btn">Usage Trends</button>
        <button onclick="showPage('deployment')" id="deployment-btn">Deployment %</button>
        <button onclick="showPage('world-map')" id="world-map-btn">World Map</button>
        <button onclick="showPage('top-customers')" id="top-customers-btn">Top Customers</button>
        <button onclick="showPage('top-products')" id="top-products-btn">Top Products</button>
    </div>

    <div class="container">
        <!-- Overview Page -->
        <div id="overview" class="page active">
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be loaded here -->
            </div>
        </div>

        <!-- Product Trends Page -->
        <div id="product-trends" class="page">
            <div class="chart-container">
                <h3>Product Quantity Trends Over Time</h3>
                <div class="controls">
                    <select id="product-select" onchange="updateProductChart()">
                        <option value="">Select Product</option>
                    </select>
                </div>
                <canvas id="product-chart"></canvas>
            </div>
        </div>

        <!-- Usage Trends Page -->
        <div id="usage-trends" class="page">
            <div class="chart-container">
                <h3>Usage Trends Over Time</h3>
                <div class="controls">
                    <select id="usage-product-select" onchange="updateUsageChart()">
                        <option value="">Select Product</option>
                    </select>
                </div>
                <canvas id="usage-chart"></canvas>
            </div>
        </div>

        <!-- Deployment Percentage Page -->
        <div id="deployment" class="page">
            <div class="chart-container">
                <h3>Deployment Percentage by Product</h3>
                <canvas id="deployment-chart"></canvas>
            </div>
        </div>

        <!-- World Map Page -->
        <div id="world-map" class="page">
            <div class="chart-container">
                <h3>User Locations Worldwide</h3>
                <div id="map"></div>
            </div>
        </div>

        <!-- Top Customers Page -->
        <div id="top-customers" class="page">
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
        </div>

        <!-- Top Products Page -->
        <div id="top-products" class="page">
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
            if (pageId === 'world-map' && !map) {
                initMap();
            }
            if (pageId === 'deployment') {
                loadDeploymentChart();
            }
            if (pageId === 'top-customers') {
                loadTopCustomersCharts();
            }
            if (pageId === 'top-products') {
                loadTopProductsCharts();
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

        function updateProductChart() {
            const productId = document.getElementById('product-select').value;
            if (!productId) return;
            
            fetch(`/api/product-trends/${productId}`)
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('product-chart').getContext('2d');
                    if (charts.productChart) charts.productChart.destroy();
                    
                    charts.productChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [{
                                label: 'Cumulative Quantity Sold',
                                data: data.quantities,
                                borderColor: '#3498db',
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                });
        }

        function updateUsageChart() {
            const productId = document.getElementById('usage-product-select').value;
            if (!productId) return;
            
            fetch(`/api/usage-trends/${productId}`)
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('usage-chart').getContext('2d');
                    if (charts.usageChart) charts.usageChart.destroy();
                    
                    charts.usageChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [{
                                label: 'Cumulative Users',
                                data: data.users,
                                borderColor: '#e74c3c',
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                });
        }

        function loadDeploymentChart() {
            fetch('/api/deployment-percentage')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('deployment-chart').getContext('2d');
                    if (charts.deploymentChart) charts.deploymentChart.destroy();
                    
                    charts.deploymentChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.products,
                            datasets: [{
                                label: 'Deployment %',
                                data: data.percentages,
                                backgroundColor: '#2ecc71'
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: { beginAtZero: true, max: 100 }
                            }
                        }
                    });
                });
        }

        function initMap() {
            map = L.map('map').setView([20, 0], 2);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            
            fetch('/api/user-locations')
                .then(response => response.json())
                .then(locations => {
                    locations.forEach(loc => {
                        L.circleMarker([loc.latitude, loc.longitude], {
                            radius: 5,
                            fillColor: '#3498db',
                            color: '#2980b9',
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        }).addTo(map).bindPopup(`${loc.city}, ${loc.country}<br>Users: ${loc.user_count}`);
                    });
                });
        }

        function loadTopCustomersCharts() {
            // Top customers by purchase
            fetch('/api/top-customers-purchase')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('customers-purchase-chart').getContext('2d');
                    if (charts.customersPurchaseChart) charts.customersPurchaseChart.destroy();
                    
                    charts.customersPurchaseChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.customers,
                            datasets: [{
                                label: 'Purchase Quantity',
                                data: data.quantities,
                                backgroundColor: '#3498db'
                            }]
                        },
                        options: { responsive: true }
                    });
                });

            // Top customers by activation
            fetch('/api/top-customers-activation')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('customers-activation-chart').getContext('2d');
                    if (charts.customersActivationChart) charts.customersActivationChart.destroy();
                    
                    charts.customersActivationChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.customers,
                            datasets: [{
                                label: 'Activation Quantity',
                                data: data.quantities,
                                backgroundColor: '#e74c3c'
                            }]
                        },
                        options: { responsive: true }
                    });
                });
        }

        function loadTopProductsCharts() {
            // Top products by purchase
            fetch('/api/top-products-purchase')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('products-purchase-chart').getContext('2d');
                    if (charts.productsPurchaseChart) charts.productsPurchaseChart.destroy();
                    
                    charts.productsPurchaseChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.products,
                            datasets: [{
                                label: 'Purchase Quantity',
                                data: data.quantities,
                                backgroundColor: '#9b59b6'
                            }]
                        },
                        options: { responsive: true }
                    });
                });

            // Top products by activation
            fetch('/api/top-products-activation')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('products-activation-chart').getContext('2d');
                    if (charts.productsActivationChart) charts.productsActivationChart.destroy();
                    
                    charts.productsActivationChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.products,
                            datasets: [{
                                label: 'Activation Quantity',
                                data: data.quantities,
                                backgroundColor: '#f39c12'
                            }]
                        },
                        options: { responsive: true }
                    });
                });
        }

        // Initialize dashboard
        window.onload = function() {
            loadOverviewStats();
            loadProductOptions();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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

@app.route('/api/product-trends/<product_id>')
def product_trends(product_id):
    """Get product quantity trends over time"""
    try:
        product_entitlements = data['entitlements'][data['entitlements']['product_id'] == product_id]
        product_entitlements = product_entitlements.sort_values('purchase_date')
        
        # Calculate cumulative quantities
        product_entitlements['cumulative_qty'] = product_entitlements['purchase_quantity'].cumsum()
        
        dates = product_entitlements['purchase_date'].dt.strftime('%Y-%m-%d').tolist()
        quantities = product_entitlements['cumulative_qty'].tolist()
        
        return jsonify({'dates': dates, 'quantities': quantities})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/usage-trends/<product_id>')
def usage_trends(product_id):
    """Get usage trends for a product"""
    try:
        # Get entitlements for the product
        product_entitlements = data['entitlements'][data['entitlements']['product_id'] == product_id]
        
        # Get activations for these entitlements
        activations = data['activations'][data['activations']['entitlement_id'].isin(product_entitlements['entitlement_id'])]
        
        # Get users for these activations
        users = data['users'][data['users']['activation_id'].isin(activations['activation_id'])]
        users = users.sort_values('first_login_date')
        
        # Calculate cumulative user count
        users['cumulative_users'] = range(1, len(users) + 1)
        
        dates = users['first_login_date'].dt.strftime('%Y-%m-%d').tolist()
        user_counts = users['cumulative_users'].tolist()
        
        return jsonify({'dates': dates, 'users': user_counts})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/deployment-percentage')
def deployment_percentage():
    """Get deployment percentage by product"""
    try:
        # Calculate deployment percentage (activated quantity / sold quantity)
        product_stats = []
        
        for _, product in data['products'].iterrows():
            product_entitlements = data['entitlements'][data['entitlements']['product_id'] == product['product_id']]
            sold_qty = product_entitlements['purchase_quantity'].sum()
            
            if sold_qty > 0:
                activations = data['activations'][data['activations']['entitlement_id'].isin(product_entitlements['entitlement_id'])]
                activated_qty = activations['quantity'].sum()
                
                deployment_pct = (activated_qty / sold_qty) * 100
                product_stats.append({
                    'product': product['name'][:20] + '...' if len(product['name']) > 20 else product['name'],
                    'percentage': round(deployment_pct, 2)
                })
        
        product_stats.sort(key=lambda x: x['percentage'], reverse=True)
        product_stats = product_stats[:10]  # Top 10
        
        products = [p['product'] for p in product_stats]
        percentages = [p['percentage'] for p in product_stats]
        
        return jsonify({'products': products, 'percentages': percentages})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/user-locations')
def user_locations():
    """Get user locations for world map"""
    try:
        # Group users by city/country and count
        location_counts = data['users'].groupby(['city', 'country', 'latitude', 'longitude']).size().reset_index(name='user_count')
        locations = location_counts.to_dict('records')
        return jsonify(locations)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/top-customers-purchase')
def top_customers_purchase():
    """Get top 10 customers by purchase quantity"""
    try:
        customer_purchases = data['entitlements'].groupby('customer_id')['purchase_quantity'].sum().reset_index()
        customer_purchases = customer_purchases.merge(data['customers'][['customer_id', 'name']], on='customer_id')
        customer_purchases = customer_purchases.sort_values('purchase_quantity', ascending=False).head(10)
        
        customers = [name[:15] + '...' if len(name) > 15 else name for name in customer_purchases['name'].tolist()]
        quantities = customer_purchases['purchase_quantity'].tolist()
        
        return jsonify({'customers': customers, 'quantities': quantities})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/top-customers-activation')
def top_customers_activation():
    """Get top 10 customers by activation quantity"""
    try:
        # Join entitlements with activations to get customer activation data
        ent_act = data['entitlements'].merge(data['activations'], on='entitlement_id')
        customer_activations = ent_act.groupby('customer_id')['quantity'].sum().reset_index()
        customer_activations = customer_activations.merge(data['customers'][['customer_id', 'name']], on='customer_id')
        customer_activations = customer_activations.sort_values('quantity', ascending=False).head(10)
        
        customers = [name[:15] + '...' if len(name) > 15 else name for name in customer_activations['name'].tolist()]
        quantities = customer_activations['quantity'].tolist()
        
        return jsonify({'customers': customers, 'quantities': quantities})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/top-products-purchase')
def top_products_purchase():
    """Get top 10 products by purchase quantity"""
    try:
        product_purchases = data['entitlements'].groupby('product_id')['purchase_quantity'].sum().reset_index()
        product_purchases = product_purchases.merge(data['products'][['product_id', 'name']], on='product_id')
        product_purchases = product_purchases.sort_values('purchase_quantity', ascending=False).head(10)
        
        products = [name[:15] + '...' if len(name) > 15 else name for name in product_purchases['name'].tolist()]
        quantities = product_purchases['purchase_quantity'].tolist()
        
        return jsonify({'products': products, 'quantities': quantities})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/top-products-activation')
def top_products_activation():
    """Get top 10 products by activation quantity"""
    try:
        # Join entitlements with activations to get product activation data
        ent_act = data['entitlements'].merge(data['activations'], on='entitlement_id')
        product_activations = ent_act.groupby('product_id')['quantity'].sum().reset_index()
        product_activations = product_activations.merge(data['products'][['product_id', 'name']], on='product_id')
        product_activations = product_activations.sort_values('quantity', ascending=False).head(10)
        
        products = [name[:15] + '...' if len(name) > 15 else name for name in product_activations['name'].tolist()]
        quantities = product_activations['quantity'].tolist()
        
        return jsonify({'products': products, 'quantities': quantities})
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    load_data()
    app.run(debug=True, host='0.0.0.0', port=5000)