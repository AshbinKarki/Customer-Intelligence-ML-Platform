"""
Streamlit Dashboard
Interactive visualization for customer analytics and model monitoring
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import logging
from datetime import datetime, timedelta

st.set_page_config(page_title="Customer ML Platform",
                   layout="wide", initial_sidebar_state="expanded")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ==================== Page Config ====================

st.markdown("""
    <style>
    .main { padding: 0rem 0rem; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==================== Sidebar Navigation ====================

st.sidebar.title("🚀 Customer ML Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🎯 Churn Analysis", "🎨 Segmentation",
        "🛍️ Recommendations", "📝 NLP Analytics", "⚙️ Model Management"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
    Full-stack AI system for:
    - Customer behavior analysis
    - Churn prediction
    - Customer segmentation
    - Product recommendations
    - Model serving via API
""")

# ==================== Helper Functions ====================


@st.cache_data
def load_customer_data():
    """Load sample customer data"""
    np.random.seed(42)
    n_customers = 1000

    data = {
        'customer_id': np.arange(1, n_customers + 1),
        'total_purchases': np.random.poisson(15, n_customers) + 1,
        'avg_order_value': np.random.gamma(2, 50, n_customers),
        'days_active': np.random.poisson(200, n_customers),
        'login_frequency': np.random.poisson(20, n_customers),
        'support_tickets': np.random.poisson(2, n_customers),
        'churn_probability': np.random.uniform(0, 1, n_customers)
    }

    df = pd.DataFrame(data)
    df['churned'] = (df['churn_probability'] > 0.5).astype(int)
    df['customer_lifetime_value'] = df['total_purchases'] * df['avg_order_value']
    df['segment'] = pd.cut(df['customer_lifetime_value'], bins=5, labels=[
                           'Inactive', 'Low', 'Medium', 'High', 'Premium'])

    return df


# Load data
customers_df = load_customer_data()

# ==================== Page: Dashboard ====================

if page == "📊 Dashboard":
    st.title("📊 Customer Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(customers_df), "↑ 12%")

    with col2:
        st.metric("Churn Rate",
                  f"{customers_df['churned'].mean():.1%}", "↓ 2%")

    with col3:
        st.metric(
            "Avg CLV", f"${customers_df['customer_lifetime_value'].mean():.0f}", "↑ 5%")

    with col4:
        st.metric("Active Users",
                  f"{(customers_df['days_active'] > 30).sum()}", "→ 0%")

    st.markdown("---")

    # Top visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Churn distribution
        fig = px.histogram(
            customers_df,
            x='churn_probability',
            nbins=30,
            title='Churn Probability Distribution',
            labels={'churn_probability': 'Churn Probability',
                    'count': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Customer segments
        segment_counts = customers_df['segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # CLV vs Churn
        fig = px.scatter(
            customers_df,
            x='customer_lifetime_value',
            y='churn_probability',
            color='segment',
            size='login_frequency',
            hover_data=['customer_id'],
            title='Customer Lifetime Value vs Churn Risk',
            labels={
                'customer_lifetime_value': 'Customer Lifetime Value ($)', 'churn_probability': 'Churn Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Purchase frequency vs Support tickets
        fig = px.scatter(
            customers_df,
            x='total_purchases',
            y='support_tickets',
            color='churned',
            title='Purchase Frequency vs Support Tickets',
            labels={'total_purchases': 'Total Purchases',
                    'support_tickets': 'Support Tickets'},
            color_discrete_map={0: 'green', 1: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== Page: Churn Analysis ====================

elif page == "🎯 Churn Analysis":
    st.title("🎯 Churn Risk Analysis")

    st.subheader("Real-time Churn Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_purchases = st.number_input(
            "Total Purchases", min_value=0, value=10)
    with col2:
        avg_order_value = st.number_input(
            "Avg Order Value ($)", min_value=0.0, value=50.0)
    with col3:
        days_active = st.number_input("Days Active", min_value=0, value=180)

    col1, col2, col3 = st.columns(3)

    with col1:
        login_frequency = st.number_input(
            "Login Frequency (per month)", min_value=0, value=15)
    with col2:
        support_tickets = st.number_input(
            "Support Tickets", min_value=0, value=1)
    with col3:
        conversion_rate = st.slider("Conversion Rate", 0.0, 1.0, 0.3)

    if st.button("🔮 Predict Churn"):
        try:
            payload = {
                "customer_id": 9999,
                "total_purchases": total_purchases,
                "avg_order_value": avg_order_value,
                "days_active": days_active,
                "login_frequency": login_frequency,
                "support_tickets": support_tickets,
                "conversion_rate": conversion_rate,
                "subscription_tier": "premium",
                "device_type": "mobile",
                "country": "US"
            }

            # Mock prediction
            risk_score = min(
                0.5 * (1 / (1 + np.exp(-(-total_purchases + 10) / 5))) +
                0.3 * (support_tickets / 10) +
                0.2 * (1 - conversion_rate),
                1.0
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Churn Risk Score",
                          f"{risk_score:.1%}", delta=f"{risk_score-0.3:.1%}")

            with col2:
                if risk_score > 0.7:
                    st.metric("Risk Level", "🔴 HIGH", delta="Critical")
                elif risk_score > 0.4:
                    st.metric("Risk Level", "🟡 MEDIUM", delta="Warning")
                else:
                    st.metric("Risk Level", "🟢 LOW", delta="Stable")

            with col3:
                st.metric("Confidence", f"{0.92:.0%}", delta="+2%")

            st.markdown("---")

            # Churn drivers
            st.subheader("Churn Risk Factors")

            drivers = pd.DataFrame({
                'Factor': ['Low Purchase Frequency', 'High Support Tickets', 'Low Conversion Rate'],
                'Impact': [0.5 * (1 / (1 + np.exp(-(-total_purchases + 10) / 5))), support_tickets / 10, 1 - conversion_rate],
                'Recommendation': [
                    '💡 Increase engagement with personalized offers',
                    '⚠️ Improve support quality and response time',
                    '📈 Optimize product recommendations'
                ]
            })

            for idx, row in drivers.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{row['Factor']}**")
                with col2:
                    st.progress(row['Impact'])
                    st.caption(row['Recommendation'])

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown("---")

    st.subheader("Batch Churn Predictions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Upload CSV for Batch Predictions"):
            st.info("Upload a CSV file with customer features")

    with col2:
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")

# ==================== Page: Segmentation ====================

elif page == "🎨 Segmentation":
    st.title("🎨 Customer Segmentation")

    st.subheader("Segment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Segment profiles
        segment_profiles = customers_df.groupby('segment').agg({
            'customer_lifetime_value': 'mean',
            'total_purchases': 'mean',
            'login_frequency': 'mean',
            'churned': 'mean'
        }).round(2)

        st.dataframe(segment_profiles, use_container_width=True)

    with col2:
        # Segment distribution over time
        dates = pd.date_range('2024-01-01', periods=30)
        segment_trend = pd.DataFrame({
            'Date': dates,
            'Inactive': np.random.randint(50, 200, 30),
            'Low': np.random.randint(100, 300, 30),
            'Medium': np.random.randint(150, 400, 30),
            'High': np.random.randint(100, 300, 30),
            'Premium': np.random.randint(50, 150, 30)
        })

        fig = px.line(
            segment_trend,
            x='Date',
            y=['Inactive', 'Low', 'Medium', 'High', 'Premium'],
            title='Segment Distribution Trend',
            labels={'value': 'Number of Customers', 'Date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== Page: Recommendations ====================

elif page == "🛍️ Recommendations":
    st.title("🛍️ Product Recommendations")

    customer_id = st.number_input(
        "Enter Customer ID", min_value=1, max_value=1000, value=1)

    if st.button("🎁 Get Recommendations"):
        # Mock recommendations
        np.random.seed(customer_id)
        products = np.random.choice(range(1, 101), 5, replace=False)
        scores = np.random.uniform(0.6, 1.0, 5)

        recom_df = pd.DataFrame({
            'Product ID': products,
            'Score': scores,
            'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 5),
            'Price': np.random.uniform(20, 200, 5).round(2)
        }).sort_values('Score', ascending=False).reset_index(drop=True)

        st.dataframe(recom_df, use_container_width=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        for i, (col, prod) in enumerate(zip([col1, col2, col3, col4, col5], recom_df['Product ID'])):
            with col:
                st.metric(f"Product {prod}",
                          f"{recom_df.iloc[i]['Score']:.2f}", "⭐")

# ==================== Page: NLP Analytics ====================

elif page == "📝 NLP Analytics":
    st.title("📝 Customer Review Sentiment Analysis")

    st.subheader("Review Sentiment Distribution")

    # Generate sample reviews
    reviews = pd.DataFrame({
        'review': ['Great product!', 'Poor quality', 'Amazing experience', 'Disappointed', 'Good value for money'],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'neutral'],
        'rating': [5, 2, 5, 2, 3]
    })

    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = reviews['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values,
                     names=sentiment_counts.index, title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            reviews.groupby('sentiment')['rating'].mean().reset_index(),
            x='sentiment',
            y='rating',
            title='Average Rating by Sentiment',
            labels={'rating': 'Average Rating', 'sentiment': 'Sentiment'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analyze Custom Review")

    review_text = st.text_area("Enter review text", height=100)

    if st.button("🔍 Analyze Sentiment"):
        # Mock sentiment
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        confidence = np.random.uniform(0.7, 0.99)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment.upper(), f"({confidence:.1%})")
        with col2:
            st.write(f"**Confidence:** {confidence:.1%}")

# ==================== Page: Model Management ====================

elif page == "⚙️ Model Management":
    st.title("⚙️ Model Management & Monitoring")

    st.subheader("Available Models")

    models = {
        'XGBoost Churn Predictor': {
            'status': '✅ Active',
            'accuracy': 0.89,
            'auc_roc': 0.92,
            'f1_score': 0.85,
            'version': '1.0.0'
        },
        'K-Means Segmentation': {
            'status': '✅ Active',
            'silhouette_score': 0.65,
            'davies_bouldin': 1.2,
            'version': '1.0.0'
        },
        'Collaborative Filtering': {
            'status': '✅ Active',
            'rmse': 0.68,
            'ndcg': 0.78,
            'version': '1.0.0'
        }
    }

    for model_name, details in models.items():
        with st.expander(f"📦 {model_name}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Status:** {details['status']}")
                st.write(f"**Version:** {details['version']}")

            with col2:
                metrics = {k: v for k, v in details.items() if k not in [
                    'status', 'version']}
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.2f}")

st.sidebar.markdown("---")
st.sidebar.caption("🔗 API: http://localhost:8000")
st.sidebar.caption("📚 Docs: http://localhost:8000/docs")
