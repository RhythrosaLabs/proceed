import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to generate sample data
def generate_sample_data(days=30):
    categories = ['Groceries', 'Rent', 'Utilities', 'Entertainment', 'Transportation', 'Shopping', 'Healthcare', 'Education']
    data = []
    start_date = datetime.now() - timedelta(days=days)
    for i in range(days):
        date = start_date + timedelta(days=i)
        for _ in range(random.randint(1, 5)):  # 1 to 5 transactions per day
            category = random.choice(categories)
            amount = round(random.uniform(10, 200), 2)
            data.append({'Date': date, 'Category': category, 'Amount': amount})
    return pd.DataFrame(data)

# Function to calculate financial health score
def calculate_financial_health(df, budget):
    total_spent = df['Amount'].sum()
    total_budget = sum(budget.values())
    overspend_ratio = total_spent / total_budget
    category_balance = sum([1 for cat, bud in budget.items() if df[df['Category'] == cat]['Amount'].sum() <= bud])
    category_balance_ratio = category_balance / len(budget)
    health_score = 100 - (overspend_ratio * 50) + (category_balance_ratio * 50)
    return max(0, min(100, health_score))

# Function to get personalized tips based on spending patterns
def get_personalized_tips(df, budget):
    tips = []
    for category, allocated in budget.items():
        spent = df[df['Category'] == category]['Amount'].sum()
        if spent > allocated:
            tips.append(f"You've overspent in {category}. Try to cut back next month.")
        elif spent < 0.5 * allocated:
            tips.append(f"You're doing great in {category}! Consider saving the extra money.")
    return tips

# Function to predict future expenses using KMeans clustering
def predict_expenses(df):
    X = df.groupby('Date').agg({'Amount': 'sum'}).reset_index()
    X['DayOfWeek'] = X['Date'].dt.dayofweek
    X['DayOfMonth'] = X['Date'].dt.day
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['Amount', 'DayOfWeek', 'DayOfMonth']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    X['Cluster'] = kmeans.fit_predict(X_scaled)
    
    next_week = pd.date_range(start=X['Date'].max() + timedelta(days=1), periods=7)
    future_X = pd.DataFrame({'Date': next_week, 'DayOfWeek': next_week.dayofweek, 'DayOfMonth': next_week.day})
    future_X_scaled = scaler.transform(future_X[['DayOfWeek', 'DayOfMonth']])
    future_X['Cluster'] = kmeans.predict(future_X_scaled)
    
    cluster_avg_spending = X.groupby('Cluster')['Amount'].mean()
    future_X['PredictedAmount'] = future_X['Cluster'].map(cluster_avg_spending)
    
    return future_X[['Date', 'PredictedAmount']]

# Main function to run the Streamlit app
def main():
    st.title("Gamified AI-Powered Personal Finance Dashboard")

    # Sidebar for user input
    st.sidebar.header("Settings")
    days = st.sidebar.slider("Number of days to analyze", 7, 90, 30)
    
    # Generate or load data
    if 'finance_data' not in st.session_state:
        st.session_state.finance_data = generate_sample_data(days)
    df = st.session_state.finance_data

    # Financial Health Score
    st.header("Your Financial Health")
    default_budget = {cat: 1000 for cat in df['Category'].unique()}
    budget = {}
    for category in df['Category'].unique():
        budget[category] = st.sidebar.number_input(f"Budget for {category}", min_value=0.0, value=1000.0, step=50.0)
    
    health_score = calculate_financial_health(df, budget)
    st.metric("Financial Health Score", f"{health_score:.2f}")
    
    # Gamification elements
    if health_score >= 90:
        st.success("Congratulations! You're a Financial Maestro! 🏆")
    elif health_score >= 70:
        st.info("Great job! You're on your way to becoming a Money Master! 💪")
    elif health_score >= 50:
        st.warning("You're doing okay, but there's room for improvement. Keep it up! 👍")
    else:
        st.error("Looks like you need to pay more attention to your finances. You can do it! 💡")

    # Overview
    st.header("Financial Overview")
    total_spent = df['Amount'].sum()
    avg_daily_spend = total_spent / days
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"${total_spent:.2f}")
    col2.metric("Average Daily Spend", f"${avg_daily_spend:.2f}")
    col3.metric("Transactions", len(df))

    # Spending by Category
    st.header("Spending Insights")
    category_spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    fig = px.pie(values=category_spending.values, names=category_spending.index, title="Spending Distribution")
    st.plotly_chart(fig)

    # Spending Over Time with Predictions
    st.header("Spending Trend and Predictions")
    daily_spending = df.groupby('Date')['Amount'].sum().reset_index()
    predictions = predict_expenses(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_spending['Date'], y=daily_spending['Amount'], mode='lines', name='Actual Spending'))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['PredictedAmount'], mode='lines', name='Predicted Spending', line=dict(dash='dot')))
    fig.update_layout(title="Daily Spending Trend with Future Predictions", xaxis_title="Date", yaxis_title="Amount")
    st.plotly_chart(fig)

    # Budget Planning and Analysis
    st.header("Budget Analysis")
    budget_comparison = []
    for category, allocated in budget.items():
        spent = df[df['Category'] == category]['Amount'].sum()
        budget_comparison.append({
            'Category': category,
            'Budget': allocated,
            'Spent': spent,
            'Remaining': allocated - spent
        })
    budget_df = pd.DataFrame(budget_comparison)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=budget_df['Category'], y=budget_df['Budget'], name='Budget'))
    fig.add_trace(go.Bar(x=budget_df['Category'], y=budget_df['Spent'], name='Spent'))
    fig.update_layout(title="Budget vs. Actual Spending", barmode='group')
    st.plotly_chart(fig)

    # Personalized Tips
    st.header("Personalized Financial Tips")
    tips = get_personalized_tips(df, budget)
    for tip in tips:
        st.info(tip)

    # Transaction History
    st.header("Recent Transactions")
    st.dataframe(df.sort_values('Date', ascending=False).head(10))

    # AI-Powered Insights
    st.header("AI-Powered Insights")
    highest_spending_day = daily_spending.loc[daily_spending['Amount'].idxmax()]
    st.write(f"Your highest spending day was {highest_spending_day['Date'].strftime('%Y-%m-%d')} with ${highest_spending_day['Amount']:.2f} spent.")
    
    most_frequent_category = df['Category'].mode().values[0]
    st.write(f"Your most frequent spending category is {most_frequent_category}.")

    # Savings Challenge
    st.header("Savings Challenge")
    savings_goal = st.number_input("Set a savings goal for next month", min_value=0.0, value=500.0, step=50.0)
    if st.button("Start Savings Challenge"):
        st.session_state.savings_goal = savings_goal
        st.success(f"Great! Your savings goal for next month is ${savings_goal:.2f}. Let's see if you can beat it!")

    if hasattr(st.session_state, 'savings_goal'):
        projected_savings = sum(budget.values()) - predictions['PredictedAmount'].sum()
        if projected_savings >= st.session_state.savings_goal:
            st.success(f"You're on track to meet your savings goal! Projected savings: ${projected_savings:.2f}")
        else:
            st.warning(f"You might miss your savings goal. Projected savings: ${projected_savings:.2f}. Try to cut back on some expenses!")

if __name__ == "__main__":
    main()
