"""
Data Generation Module
Generates synthetic customer data for training and testing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_customer_data(n_customers=5000, n_days=365):
    """Generate synthetic customer behavior data"""

    np.random.seed(42)

    # Generate basic customer info
    customer_ids = np.arange(1, n_customers + 1)
    countries = np.random.choice(
        ['US', 'UK', 'DE', 'FR', 'CA', 'AU'], n_customers)
    device_types = np.random.choice(
        ['mobile', 'desktop', 'tablet'], n_customers, p=[0.5, 0.35, 0.15])
    subscription_tiers = np.random.choice(
        ['free', 'basic', 'premium', 'enterprise'], n_customers, p=[0.3, 0.4, 0.25, 0.05])

    # Generate behavioral features
    total_purchases = np.random.poisson(lam=15, size=n_customers) + 1
    avg_order_value = np.random.gamma(shape=2, scale=50, size=n_customers)
    days_active = np.random.poisson(lam=200, size=n_customers)
    login_frequency = np.random.poisson(lam=20, size=n_customers)
    support_tickets = np.random.poisson(lam=2, size=n_customers)

    # Browse-to-purchase ratio (engagement metric)
    browse_count = total_purchases * np.random.uniform(5, 15, n_customers)
    conversion_rate = total_purchases / (browse_count + 1)

    # Feature engineering
    customer_lifetime_value = total_purchases * avg_order_value
    avg_session_duration = days_active / (login_frequency + 1)

    # Churn target (with relationship to features)
    churn_prob = (
        # Low purchases -> higher churn
        0.5 * (1 / (1 + np.exp(-(-total_purchases + 10) / 5))) +
        # High support tickets -> higher churn
        0.3 * (support_tickets / (support_tickets.max() + 1)) +
        0.2 * (1 - conversion_rate)  # Low conversion -> higher churn
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churned = (np.random.random(n_customers) < churn_prob).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'country': countries,
        'device_type': device_types,
        'subscription_tier': subscription_tiers,
        'total_purchases': total_purchases,
        'avg_order_value': avg_order_value,
        'days_active': days_active,
        'login_frequency': login_frequency,
        'support_tickets': support_tickets,
        'browse_count': browse_count.astype(int),
        'conversion_rate': conversion_rate,
        'customer_lifetime_value': customer_lifetime_value,
        'avg_session_duration': avg_session_duration,
        'churned': churned
    })

    # Add temporal features
    df['signup_date'] = [datetime.now() - timedelta(days=int(d))
                         for d in df['days_active']]
    df['last_purchase_date'] = [df['signup_date'].iloc[i] + timedelta(days=int(np.random.randint(1, df['days_active'].iloc[i])))
                                for i in range(len(df))]
    df['days_since_last_purchase'] = (
        datetime.now() - df['last_purchase_date']).dt.days
    df['date'] = datetime.now().date()

    return df


def generate_product_data(n_products=100):
    """Generate synthetic product data for recommendations"""

    np.random.seed(42)

    product_ids = np.arange(1, n_products + 1)
    categories = np.random.choice(
        ['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_products)
    prices = np.random.exponential(scale=50, size=n_products) + 10
    ratings = np.random.uniform(3, 5, n_products)
    popularity = np.random.poisson(lam=100, size=n_products)

    df = pd.DataFrame({
        'product_id': product_ids,
        'category': categories,
        'price': prices,
        'rating': ratings,
        'popularity': popularity,
        'created_date': [datetime.now() - timedelta(days=int(d)) for d in np.random.exponential(scale=365, size=n_products)]
    })

    return df


def generate_purchase_history(n_customers=5000, n_products=100, n_transactions=20000):
    """Generate purchase history for collaborative filtering"""

    np.random.seed(42)

    customer_ids = np.random.choice(
        np.arange(1, n_customers + 1), n_transactions)
    product_ids = np.random.choice(
        np.arange(1, n_products + 1), n_transactions)
    ratings = np.random.randint(1, 6, n_transactions)
    purchase_dates = [datetime.now() - timedelta(days=int(d))
                      for d in np.random.exponential(scale=180, size=n_transactions)]

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'product_id': product_ids,
        'rating': ratings,
        'purchase_date': purchase_dates
    })

    return df.drop_duplicates(subset=['customer_id', 'product_id'], keep='last')


def generate_reviews(n_reviews=2000):
    """Generate customer reviews for NLP sentiment analysis"""

    np.random.seed(42)

    positive_phrases = [
        "Great product! Highly recommend", "Excellent quality and fast shipping",
        "Love this! Will buy again", "Exceeded my expectations",
        "Best purchase ever made", "Amazing customer service"
    ]

    negative_phrases = [
        "Terrible product, waste of money", "Poor quality, very disappointed",
        "Arrived damaged", "Not as described", "Worst experience ever",
        "Unresponsive customer support"
    ]

    neutral_phrases = [
        "It's okay, nothing special", "Average product for the price",
        "Does what it says", "Acceptable quality", "Could be better"
    ]

    sentiments = []
    reviews = []
    ratings = []

    for _ in range(n_reviews):
        rand = np.random.random()
        if rand < 0.5:
            review = np.random.choice(positive_phrases)
            sentiment = 'positive'
            rating = np.random.randint(4, 6)
        elif rand < 0.85:
            review = np.random.choice(negative_phrases)
            sentiment = 'negative'
            rating = np.random.randint(1, 3)
        else:
            review = np.random.choice(neutral_phrases)
            sentiment = 'neutral'
            rating = 3

        sentiments.append(sentiment)
        reviews.append(review)
        ratings.append(rating)

    df = pd.DataFrame({
        'customer_id': np.random.randint(1, 5001, n_reviews),
        'product_id': np.random.randint(1, 101, n_reviews),
        'review_text': reviews,
        'sentiment': sentiments,
        'rating': ratings,
        'review_date': [datetime.now() - timedelta(days=int(d)) for d in np.random.exponential(scale=180, size=n_reviews)]
    })

    return df


def main():
    """Generate all datasets and save to disk"""

    os.makedirs('data/raw', exist_ok=True)

    print("Generating customer data...")
    customers = generate_customer_data(n_customers=5000)
    customers.to_csv('data/raw/customers.csv', index=False)
    print(f"✓ Generated {len(customers)} customers")

    print("Generating product data...")
    products = generate_product_data(n_products=100)
    products.to_csv('data/raw/products.csv', index=False)
    print(f"✓ Generated {len(products)} products")

    print("Generating purchase history...")
    purchases = generate_purchase_history(
        n_customers=5000, n_products=100, n_transactions=20000)
    purchases.to_csv('data/raw/purchase_history.csv', index=False)
    print(f"✓ Generated {len(purchases)} purchases")

    print("Generating customer reviews...")
    reviews = generate_reviews(n_reviews=2000)
    reviews.to_csv('data/raw/reviews.csv', index=False)
    print(f"✓ Generated {len(reviews)} reviews")

    print("\n✅ All datasets generated successfully!")
    print(f"Customer data shape: {customers.shape}")
    print(f"\nCustomer data sample:\n{customers.head()}")


if __name__ == "__main__":
    main()
