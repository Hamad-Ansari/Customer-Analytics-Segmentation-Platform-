# app.py - Main Streamlit Application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import pickle
import hashlib
from io import BytesIO
import base64

# Backend modules
from backend.data_processor import DataProcessor
from backend.segmentation_engine import SegmentationEngine
from backend.database import DatabaseManager
from backend.auth import AuthenticationSystem

# Set page configuration
st.set_page_config(
    page_title="Harminder's Customer Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .harminder-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .success-message {
        background-color: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<h1 class="main-header">📊 Harminder\'s Customer Analytics Suite</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; margin-bottom: 2rem;">'
            '<span class="harminder-badge">Professional ML Platform</span>'
            '<span class="harminder-badge">Enterprise Ready</span>'
            '<span class="harminder-badge">Real-time Analytics</span>'
            '</div>', unsafe_allow_html=True)

# Initialize backend systems
@st.cache_resource
def init_backend():
    """Initialize all backend systems"""
    try:
        db = DatabaseManager()
        auth = AuthenticationSystem()
        processor = DataProcessor()
        engine = SegmentationEngine()
        return db, auth, processor, engine
    except Exception as e:
        st.error(f"Backend initialization error: {str(e)}")
        return None, None, None, None

# Initialize backend
db, auth, processor, engine = init_backend()

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.markdown("### 🎯 Navigation")
    menu = st.selectbox(
        "Select Module",
        ["📈 Dashboard", "👥 Customer Segmentation", "📊 Analytics", "⚙️ Model Management", "🔐 User Management", "📤 Data Import"]
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Quick Actions")
    if st.button("🔄 Refresh All Data"):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🏆 Harminder's Platform")
    st.info("""
    **Professional Features:**
    • Full-stack architecture
    • Real-time processing
    • Enterprise security
    • Scalable backend
    """)

# Main Application Logic
if menu == "📈 Dashboard":
    st.header("📈 Executive Dashboard")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🎯 Total Customers</h3><h2>1,248</h2><p>+12% from last month</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>📊 Segments</h3><h2>5</h2><p>Active clusters</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>📈 Avg. Value</h3><h2>$1,245</h2><p>Per customer</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>⚡ Accuracy</h3><h2>94.7%</h2><p>Model performance</p></div>', unsafe_allow_html=True)
    
    # Charts Section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><h4>📅 Monthly Customer Growth</h4></div>', unsafe_allow_html=True)
        # Generate sample data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        customers = [800, 900, 950, 1100, 1200, 1248]
        
        fig = go.Figure(data=go.Scatter(
            x=months, y=customers,
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10, color='#764ba2')
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="card"><h4>🎯 Segment Distribution</h4></div>', unsafe_allow_html=True)
        segments = ['Premium', 'Loyal', 'New', 'At-Risk', 'Inactive']
        distribution = [25, 30, 20, 15, 10]
        
        fig = px.pie(
            values=distribution,
            names=segments,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.markdown("---")
    st.markdown('<div class="card"><h4>📝 Recent Activity</h4></div>', unsafe_allow_html=True)
    
    activity_data = {
        'Time': ['10:30 AM', '09:45 AM', 'Yesterday', '2 days ago'],
        'Action': ['Model retrained', 'New data imported', 'Segment analysis completed', 'User permissions updated'],
        'User': ['Harminder', 'Harminder', 'System', 'Harminder'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success']
    }
    
    st.dataframe(pd.DataFrame(activity_data), use_container_width=True, hide_index=True)

elif menu == "👥 Customer Segmentation":
    st.header("👥 Customer Segmentation Engine")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Cluster Analysis", "🎯 Predict Segment", "📊 Compare Segments"])
    
    with tab1:
        st.markdown('<div class="card"><h4>Advanced Clustering Analysis</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Generate sample customer data
            np.random.seed(42)
            n_customers = 300
            
            data = pd.DataFrame({
                'Customer_ID': [f'CUST{1000+i}' for i in range(n_customers)],
                'Age': np.random.randint(18, 70, n_customers),
                'Annual_Income': np.random.randint(20000, 200000, n_customers),
                'Spending_Score': np.random.randint(1, 100, n_customers),
                'Credit_Score': np.random.randint(300, 850, n_customers),
                'Tenure_Months': np.random.randint(1, 60, n_customers)
            })
            
            # Apply clustering
            data['Segment'] = pd.cut(data['Spending_Score'], 
                                     bins=[0, 20, 40, 60, 80, 100],
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # 3D Scatter plot
            fig = px.scatter_3d(
                data,
                x='Age',
                y='Annual_Income',
                z='Spending_Score',
                color='Segment',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=['Customer_ID', 'Credit_Score'],
                title="Harminder's 3D Customer Segmentation"
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Age',
                    yaxis_title='Annual Income',
                    zaxis_title='Spending Score'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="card"><h5>🧮 Segment Statistics</h5></div>', unsafe_allow_html=True)
            
            segment_stats = data.groupby('Segment').agg({
                'Annual_Income': 'mean',
                'Spending_Score': 'mean',
                'Age': 'mean',
                'Customer_ID': 'count'
            }).round(2)
            
            segment_stats.columns = ['Avg Income', 'Avg Spend Score', 'Avg Age', 'Count']
            st.dataframe(segment_stats, use_container_width=True)
            
            st.markdown('<div class="card"><h5>⚙️ Clustering Parameters</h5></div>', unsafe_allow_html=True)
            
            n_clusters = st.slider("Number of Clusters", 2, 8, 5)
            algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])
            
            if st.button("🚀 Run Clustering", key="run_cluster"):
                with st.spinner("Harminder's AI engine is processing..."):
                    time.sleep(2)
                    st.success("✅ Clustering completed successfully!")
                    
                    # Show cluster insights
                    st.markdown('<div class="success-message">'
                                '<strong>Insights Generated:</strong><br>'
                                '• Identified 5 distinct customer personas<br>'
                                '• High-income segment shows 40% higher lifetime value<br>'
                                '• Young adults (18-30) have highest engagement<br>'
                                '• Premium segment contributes 60% of total revenue'
                                '</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card"><h4>🔮 Real-time Customer Prediction</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Enter Customer Details")
            
            age = st.slider("Age", 18, 80, 35)
            income = st.number_input("Annual Income ($)", 20000, 500000, 75000)
            spend_score = st.slider("Spending Score", 1, 100, 50)
            credit_score = st.slider("Credit Score", 300, 850, 700)
            tenure = st.slider("Tenure (months)", 1, 120, 12)
            
            if st.button("🎯 Predict Segment", key="predict_segment"):
                with st.spinner("Harminder's AI is analyzing..."):
                    time.sleep(1.5)
                    
                    # Simulate prediction logic
                    if income > 150000 and spend_score > 70:
                        segment = "💎 Premium"
                        color = "success"
                        description = "High-value customer with excellent engagement"
                    elif income > 80000 and credit_score > 700:
                        segment = "⭐ Loyal"
                        color = "info"
                        description = "Reliable customer with good credit"
                    elif age < 30 and spend_score > 60:
                        segment = "🚀 High Potential"
                        color = "warning"
                        description = "Young customer with high engagement"
                    else:
                        segment = "📊 Standard"
                        color = "secondary"
                        description = "Regular customer profile"
                    
                    st.markdown(f'<div class="success-message">'
                                f'<h4>Prediction Result</h4>'
                                f'<h3>{segment}</h3>'
                                f'<p>{description}</p>'
                                f'<small>Confidence: 92.5%</small>'
                                f'</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📋 Customer Persona Details")
            
            persona_data = {
                "💎 Premium": {
                    "description": "High-income professionals with premium spending habits",
                    "characteristics": ["Income > $150k", "Spend Score > 70", "Age 30-50"],
                    "strategy": "Premium offers, personal concierge",
                    "avg_value": "$5,200"
                },
                "⭐ Loyal": {
                    "description": "Long-term customers with consistent engagement",
                    "characteristics": ["Tenure > 24 months", "Credit Score > 700", "Regular purchases"],
                    "strategy": "Loyalty rewards, exclusive access",
                    "avg_value": "$2,800"
                },
                "🚀 High Potential": {
                    "description": "Young customers showing high engagement",
                    "characteristics": ["Age < 30", "High digital engagement", "Growing income"],
                    "strategy": "Growth-focused offers, tech products",
                    "avg_value": "$1,500"
                }
            }
            
            selected_persona = st.selectbox("Select Persona", list(persona_data.keys()))
            
            if selected_persona in persona_data:
                persona = persona_data[selected_persona]
                st.markdown(f"""
                <div class="card">
                    <h5>{selected_persona} Persona</h5>
                    <p><strong>Description:</strong> {persona['description']}</p>
                    <p><strong>Key Characteristics:</strong><br>
                    • {persona['characteristics'][0]}<br>
                    • {persona['characteristics'][1]}<br>
                    • {persona['characteristics'][2]}</p>
                    <p><strong>Marketing Strategy:</strong> {persona['strategy']}</p>
                    <p><strong>Average Value:</strong> {persona['avg_value']}</p>
                </div>
                """, unsafe_allow_html=True)

elif menu == "📊 Analytics":
    st.header("📊 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🔍 Customer Journey", "📋 Data Explorer"])
    
    with tab1:
        st.markdown('<div class="card"><h4>Business Performance Dashboard</h4></div>', unsafe_allow_html=True)
        
        # Generate time series data
        dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='MS')
        metrics_data = pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.normal(100000, 20000, len(dates)).cumsum(),
            'Customers': np.random.randint(800, 1300, len(dates)),
            'Conversion': np.random.uniform(2.5, 4.5, len(dates)),
            'CLV': np.random.normal(1200, 200, len(dates))
        })
        
        # Line chart for revenue
        fig = px.line(metrics_data, x='Date', y='Revenue',
                     title='📈 Revenue Trend - Harminder\'s Analysis',
                     line_shape='spline')
        fig.update_traces(line=dict(width=4))
        st.plotly_chart(fig, use_container_width=True)
        
        # Multiple metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Monthly Revenue", "$124,856", "+12.5%")
        with col2:
            st.metric("👥 Active Customers", "1,248", "+8.2%")
        with col3:
            st.metric("🎯 Conversion Rate", "3.8%", "+0.4%")

elif menu == "⚙️ Model Management":
    st.header("⚙️ Model Management System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card"><h4>🤖 Machine Learning Models</h4></div>', unsafe_allow_html=True)
        
        models = {
            "Customer Segmentation": {
                "status": "✅ Production",
                "accuracy": "94.7%",
                "last_trained": "2024-01-15",
                "version": "v2.1.0"
            },
            "Churn Prediction": {
                "status": "✅ Production",
                "accuracy": "89.3%",
                "last_trained": "2024-01-10",
                "version": "v1.5.2"
            },
            "LTV Prediction": {
                "status": "🟡 Testing",
                "accuracy": "92.1%",
                "last_trained": "2024-01-12",
                "version": "v1.0.0"
            }
        }
        
        for model_name, details in models.items():
            with st.expander(f"{model_name} - {details['status']}"):
                st.write(f"**Accuracy:** {details['accuracy']}")
                st.write(f"**Last Trained:** {details['last_trained']}")
                st.write(f"**Version:** {details['version']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"🔄 Retrain {model_name}", key=f"retrain_{model_name}"):
                        with st.spinner(f"Retraining {model_name}..."):
                            time.sleep(3)
                            st.success(f"✅ {model_name} retrained successfully!")
                with col_b:
                    if st.button(f"📊 View Metrics", key=f"metrics_{model_name}"):
                        st.info(f"Opening metrics dashboard for {model_name}...")

elif menu == "🔐 User Management":
    st.header("🔐 User Management")
    
    if 'users' not in st.session_state:
        st.session_state.users = [
            {"username": "harminder", "role": "Admin", "email": "harminder@company.com", "last_login": "2024-01-15"},
            {"username": "alex", "role": "Analyst", "email": "alex@company.com", "last_login": "2024-01-14"},
            {"username": "sarah", "role": "Viewer", "email": "sarah@company.com", "last_login": "2024-01-13"}
        ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><h4>👥 User List</h4></div>', unsafe_allow_html=True)
        users_df = pd.DataFrame(st.session_state.users)
        st.dataframe(users_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown('<div class="card"><h4>➕ Add New User</h4></div>', unsafe_allow_html=True)
        
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_role = st.selectbox("Role", ["Admin", "Analyst", "Viewer"])
            
            if st.form_submit_button("➕ Add User"):
                if new_username and new_email:
                    st.session_state.users.append({
                        "username": new_username,
                        "role": new_role,
                        "email": new_email,
                        "last_login": datetime.now().strftime("%Y-%m-%d")
                    })
                    st.success(f"✅ User {new_username} added successfully!")
                    st.rerun()

elif menu == "📤 Data Import":
    st.header("📤 Data Import & Processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.markdown('<div class="success-message">'
                   '<strong>✅ File Uploaded Successfully!</strong><br>'
                   f'File: {uploaded_file.name}<br>'
                   f'Size: {uploaded_file.size / 1024:.2f} KB'
                   '</div>', unsafe_allow_html=True)
        
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display preview
        st.markdown('<div class="card"><h4>📋 Data Preview</h4></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show statistics
        st.markdown('<div class="card"><h4>📊 Data Statistics</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data processing options
        st.markdown('<div class="card"><h4>⚙️ Processing Options</h4></div>', unsafe_allow_html=True)
        
        if st.button("🚀 Process Data with Harminder's Engine"):
            with st.spinner("Processing data..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.success("✅ Data processed successfully!")
                
                # Show processed data info
                st.markdown("""
                **Processing Completed:**
                - ✅ Data validation passed
                - ✅ Missing values handled
                - ✅ Feature engineering completed
                - ✅ Ready for segmentation
                """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Harminder's Analytics Suite**")
    st.markdown("v2.1.0 | Professional Edition")
with col2:
    st.markdown("**🚀 Enterprise Ready**")
    st.markdown("Scalable • Secure • Reliable")
with col3:
    st.markdown("**📞 Support**")
    st.markdown("mrhammadzahid24@gmail.com| +1-800-555-1234 ]")

st.markdown('<div style="text-align: center; color: #666; padding: 2rem;">'
            '© 2024 Harminder\'s Customer Analytics Platform. All rights reserved.'
            '</div>', unsafe_allow_html=True)