import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Function to generate sample data
def generate_sample_data(days=30):
    categories = ['Groceries', 'Rent', 'Utilities', 'Entertainment', 'Transportation']
    data = []
    start_date = datetime.now() - timedelta(days=days)
    for i in range(days):
        date = start_date + timedelta(days=i)
        for _ in range(random.randint(1, 5)):  # 1 to 5 transactions per day
            category = random.choice(categories)
            amount = round(random.uniform(10, 200), 2)
            data.append({'Date': date, 'Category': category, 'Amount': amount})
    return pd.DataFrame(data)

# Main function to run the Streamlit app
def main():
    st.title("Personal Finance Dashboard and Budgeting Tool")

    # Sidebar for user input
    st.sidebar.header("Settings")
    days = st.sidebar.slider("Number of days to analyze", 7, 90, 30)
    
    # Generate or load data
    if 'finance_data' not in st.session_state:
        st.session_state.finance_data = generate_sample_data(days)
    df = st.session_state.finance_data

    # Overview
    st.header("Financial Overview")
    total_spent = df['Amount'].sum()
    avg_daily_spend = total_spent / days
    st.metric("Total Spent", f"${total_spent:.2f}")
    st.metric("Average Daily Spend", f"${avg_daily_spend:.2f}")

    # Spending by Category
    st.header("Spending by Category")
    category_spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    fig = px.pie(values=category_spending.values, names=category_spending.index, title="Spending Distribution")
    st.plotly_chart(fig)

    # Spending Over Time
    st.header("Spending Over Time")
    daily_spending = df.groupby('Date')['Amount'].sum().reset_index()
    fig = px.line(daily_spending, x='Date', y='Amount', title="Daily Spending Trend")
    st.plotly_chart(fig)

    # Transaction History
    st.header("Recent Transactions")
    st.dataframe(df.sort_values('Date', ascending=False).head(10))

    # Budget Planning
    st.header("Budget Planning")
    categories = df['Category'].unique()
    budget_data = {}
    for category in categories:
        budget_data[category] = st.number_input(f"Budget for {category}", min_value=0.0, value=500.0, step=50.0)

    if st.button("Analyze Budget"):
        budget_comparison = []
        for category, budget in budget_data.items():
            spent = df[df['Category'] == category]['Amount'].sum()
            budget_comparison.append({
                'Category': category,
                'Budget': budget,
                'Spent': spent,
                'Remaining': budget - spent
            })
        budget_df = pd.DataFrame(budget_comparison)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=budget_df['Category'], y=budget_df['Budget'], name='Budget'))
        fig.add_trace(go.Bar(x=budget_df['Category'], y=budget_df['Spent'], name='Spent'))
        fig.update_layout(title="Budget vs. Actual Spending", barmode='group')
        st.plotly_chart(fig)

        st.dataframe(budget_df)

if __name__ == "__main__":
    main()
