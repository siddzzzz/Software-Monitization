from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy import stats
import os
import logging
import io
import base64
from datetime import datetime

app = Flask(__name__)

DATA_DIR = 'software_monetization_dataset'
OUTPUT_DIR = 'eda_outputs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda_analysis.log'),
        logging.StreamHandler()
    ]
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
FIGSIZE = (12, 8)
def load_datasets():
    datasets = {}
    files = [
        'vendors.csv', 'customers.csv', 'products.csv', 'licenses.csv',
        'usage_history.csv', 'renewal_history.csv', 'customer_summary.csv',
        'product_performance.csv', 'vendor_performance.csv'
    ]
    
    for file in files:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{file}')
            datasets[file.replace('.csv', '')] = df
            logging.info(f"Loaded {file}: {df.shape}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file}")
        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
    
    if not datasets:
        logging.error("No datasets loaded successfully")
    return datasets

def basic_data_overview(datasets):
    try:
        overview_stats = []
        for name, df in datasets.items():
            stats_dict = {
                'Dataset': name,
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Memory_Usage_MB': df.memory_usage(deep=True).sum() / (1024**2),
                'Missing_Values': df.isnull().sum().sum(),
                'Missing_Percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'Duplicate_Rows': df.duplicated().sum(),
                'Numeric_Columns': df.select_dtypes(include=[np.number]).shape[1],
                'Categorical_Columns': df.select_dtypes(include=['object']).shape[1],
                'Date_Columns': df.select_dtypes(include=['datetime64']).shape[1]
            }
            overview_stats.append(stats_dict)
        
        overview_df = pd.DataFrame(overview_stats)
        logging.info("Generated dataset overview")
        return overview_df
    except Exception as e:
        logging.error(f"Error in basic_data_overview: {str(e)}")
        return pd.DataFrame()

def visualize_missing_data(df, title="Missing Data Pattern"):
    try:
        if df.isnull().sum().sum() == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{title} - Missing Data Analysis', fontsize=16)
        
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=axes[0])
        axes[0].set_title('Missing Data Heatmap')
        
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            missing_counts.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Missing Data Count by Column')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.info(f"Generated missing data visualization for {title}")
        return image_base64
    except Exception as e:
        logging.error(f"Error in visualize_missing_data: {str(e)}")
        return None

def analyze_distributions(df, title="Dataset"):
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df[col].hist(bins=50, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.info(f"Generated distribution visualization for {title}")
        return image_base64
    except Exception as e:
        logging.error(f"Error in analyze_distributions: {str(e)}")
        return None

def correlation_analysis(df, title="Dataset"):
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title(f'{title} - Correlation Matrix')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.info(f"Generated correlation visualization for {title}")
        return image_base64
    except Exception as e:
        logging.error(f"Error in correlation_analysis: {str(e)}")
        return None

def detect_anomalies(df, title="Dataset"):
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        df_numeric = df[numeric_cols].dropna()
        original_indices = df.index[df[numeric_cols].notna().all(axis=1)].tolist()
        
        z_scores = np.abs(stats.zscore(df_numeric))
        statistical_outliers = (z_scores > 3).any(axis=1)
        statistical_outlier_indices = df_numeric.index[statistical_outliers].tolist()
        
        if len(df_numeric) > 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            isolation_outliers = iso_forest.fit_predict(df_numeric) == -1
        else:
            isolation_outliers = np.zeros(len(df_numeric), dtype=bool)
        isolation_outlier_indices = df_numeric.index[isolation_outliers].tolist()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if len(numeric_cols) > 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(df_numeric)
            pc1, pc2 = pca_data[:, 0], pca_data[:, 1]
        else:
            pc1, pc2 = df_numeric.iloc[:, 0], df_numeric.iloc[:, 1]
        
        axes[0].scatter(pc1[~statistical_outliers], pc2[~statistical_outliers], 
                        c='blue', alpha=0.6, label='Normal')
        axes[0].scatter(pc1[statistical_outliers], pc2[statistical_outliers], 
                        c='red', alpha=0.8, label='Outliers')
        axes[0].set_title('Statistical Outliers (Z-score > 3)')
        axes[0].legend()
        
        axes[1].scatter(pc1[~isolation_outliers], pc2[~isolation_outliers], 
                        c='blue', alpha=0.6, label='Normal')
        axes[1].scatter(pc1[isolation_outliers], pc2[isolation_outliers], 
                        c='red', alpha=0.8, label='Outliers')
        axes[1].set_title('Isolation Forest Outliers')
        axes[1].legend()
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.info(f"Generated anomaly visualization for {title}")
        
        return {
            'image': image_base64,
            'summary': pd.DataFrame({
                'Statistical_Outliers': [statistical_outliers.sum()],
                'Isolation_Forest_Outliers': [isolation_outliers.sum()],
                'Total_Records': [len(df_numeric)]
            }),
            'statistical_outlier_indices': statistical_outlier_indices,
            'isolation_outlier_indices': isolation_outlier_indices
        }
    except Exception as e:
        logging.error(f"Error in detect_anomalies: {str(e)}")
        return None

def customer_segmentation(customer_summary_df):
    try:
        if customer_summary_df.empty:
            return None
        
        features = ['Total_Purchased', 'Total_Activated', 'Total_Contract_Value', 
                    'Avg_Satisfaction', 'Total_Support_Tickets']
        available_features = [f for f in features if f in customer_summary_df.columns]
        
        if len(available_features) < 2:
            return None
        
        segmentation_data = customer_summary_df[available_features].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(segmentation_data)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_summary_df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        customer_summary_df['Cluster'].value_counts().plot(kind='bar', ax=axes[0])
        axes[0].set_title('Cluster Distribution')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Count')
        
        scatter = axes[1].scatter(
            customer_summary_df[available_features[0]], 
            customer_summary_df[available_features[1]], 
            c=customer_summary_df['Cluster'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[1].set_xlabel(available_features[0])
        axes[1].set_ylabel(available_features[1])
        axes[1].set_title('Cluster Visualization')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.info("Generated customer segmentation visualization")
        
        return {
            'image': image_base64,
            'summary': customer_summary_df.groupby('Cluster')[available_features].agg(['mean', 'count']).round(2)
        }
    except Exception as e:
        logging.error(f"Error in customer_segmentation: {str(e)}")
        return None

def create_interactive_dashboard(datasets):
    try:
        if 'licenses' not in datasets:
            return None
        
        df = datasets['licenses']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Purchase Trends', 'Activation vs Deployment', 
                           'Contract Value Distribution', 'Customer Satisfaction')
        )
        
        if 'Days_since_last_quantity_purchased' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['Days_since_last_quantity_purchased'], 
                            name='Days Since Last Purchase',
                            nbinsx=30),
                row=1, col=1
            )
        
        if 'Number_of_quantities_activated' in df.columns and 'Percentage_of_quantities_deployed' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Number_of_quantities_activated'], 
                          y=df['Percentage_of_quantities_deployed'],
                          mode='markers',
                          name='Activation vs Deployment',
                          opacity=0.6),
                row=1, col=2
            )
        
        if 'Contract_Value' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['Contract_Value'], 
                            name='Contract Value',
                            nbinsx=30),
                row=2, col=1
            )
        
        if 'Satisfaction_Score' in df.columns:
            satisfaction_counts = df['Satisfaction_Score'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=satisfaction_counts.index, 
                       y=satisfaction_counts.values,
                       name='Satisfaction Score'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Software Monetization Dataset Dashboard",
            showlegend=False,
            height=800
        )
        
        logging.info("Generated interactive dashboard")
        return fig.to_html(full_html=False)
    except Exception as e:
        logging.error(f"Error in create_interactive_dashboard: {str(e)}")
        return None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/overview')
def get_overview():
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        if not datasets:
            logging.error("No datasets loaded for overview")
            return jsonify({'error': 'No datasets loaded'}), 500
        
        overview_df = basic_data_overview(datasets)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'data': overview_df.to_dict(orient='records'),
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_overview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/missing-data/<dataset>')
def get_missing_data(dataset):
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        if dataset not in datasets:
            logging.error(f"Dataset {dataset} not found")
            return jsonify({'error': f'Dataset {dataset} not found'}), 404
        
        df = datasets[dataset]
        missing_viz = visualize_missing_data(df, f"{dataset.title()} Dataset")
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'visualization': {
                'name': f'missing_data_{dataset}.png',
                'type': 'image',
                'data': missing_viz
            } if missing_viz else None,
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_missing_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/distributions/<dataset>')
def get_distributions(dataset):
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        if dataset not in datasets:
            logging.error(f"Dataset {dataset} not found")
            return jsonify({'error': f'Dataset {dataset} not found'}), 404
        
        df = datasets[dataset]
        dist_viz = analyze_distributions(df, f"{dataset.title()} Dataset")
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'visualization': {
                'name': f'distributions_{dataset}.png',
                'type': 'image',
                'data': dist_viz
            } if dist_viz else None,
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_distributions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations/<dataset>')
def get_correlations(dataset):
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        if dataset not in datasets:
            logging.error(f"Dataset {dataset} not found")
            return jsonify({'error': f'Dataset {dataset} not found'}), 404
        
        df = datasets[dataset]
        corr_viz = correlation_analysis(df, f"{dataset.title()} Dataset")
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'visualization': {
                'name': f'correlation_{dataset}.png',
                'type': 'image',
                'data': corr_viz
            } if corr_viz else None,
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_correlations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomalies/<dataset>')
def get_anomalies(dataset):
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        if dataset not in datasets:
            logging.error(f"Dataset {dataset} not found")
            return jsonify({'error': f'Dataset {dataset} not found'}), 404
        
        df = datasets[dataset]
        anomaly_results = detect_anomalies(df, f"{dataset.title()} Dataset")
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'visualization': {
                'name': f'anomalies_{dataset}.png',
                'type': 'image',
                'data': anomaly_results['image']
            } if anomaly_results else None,
            'summary': anomaly_results['summary'].to_dict(orient='records') if anomaly_results else [],
            'statistical_outlier_indices': anomaly_results['statistical_outlier_indices'] if anomaly_results else [],
            'isolation_outlier_indices': anomaly_results['isolation_outlier_indices'] if anomaly_results else [],
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_anomalies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/customer-segmentation')
def get_customer_segmentation():
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        if 'customer_summary' not in datasets:
            logging.error("customer_summary dataset not found")
            return jsonify({'error': 'customer_summary dataset not found'}), 404
        
        seg_results = customer_segmentation(datasets['customer_summary'])
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'visualization': {
                'name': 'customer_segmentation.png',
                'type': 'image',
                'data': seg_results['image']
            } if seg_results else None,
            'summary': seg_results['summary'].to_dict(orient='records') if seg_results else [],
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_customer_segmentation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interactive-dashboard')
def get_interactive_dashboard():
    try:
        start_time = datetime.now()
        datasets = load_datasets()
        dashboard_html = create_interactive_dashboard(datasets)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60
        
        return jsonify({
            'visualization': {
                'name': 'interactive_dashboard.html',
                'type': 'html',
                'data': dashboard_html
            } if dashboard_html else None,
            'execution_time': execution_time
        })
    except Exception as e:
        logging.error(f"Error in get_interactive_dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask")
    app.run(debug=True, host='0.0.0.0', port=5000)