import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 🔹 Step 1: Generate Dummy Data for NRI Business Analysis
np.random.seed(42)

# Dummy NRI Product Team Revenue & Income Data
nri_data = pd.DataFrame({
    'branch_id': np.arange(1, 51),
    'nri_customers': np.random.randint(100, 500, 50),
    'revenue_generated': np.random.randint(500000, 2000000, 50),
    'income_generated': np.random.randint(200000, 800000, 50),
    'operational_cost': np.random.randint(100000, 500000, 50),
    'loan_defaults': np.random.randint(0, 30, 50),
    'term_deposits': np.random.randint(100, 1000, 50),
    'market_competition_index': np.random.uniform(0.1, 1.0, 50)
})

# 🔹 Step 2: Analyze Revenue, Income, and Cost Efficiency
nri_data['profit_margin'] = nri_data['revenue_generated'] - nri_data['operational_cost']
nri_data['cost_efficiency'] = nri_data['profit_margin'] / nri_data['operational_cost']

# 🔹 Step 3: Credit Risk Analysis (Loan Defaults Prediction)
X = nri_data[['nri_customers', 'term_deposits', 'market_competition_index']]
y = nri_data['loan_defaults']

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

mse = mean_squared_error(y, predictions)
print("📊 Mean Squared Error for Loan Default Prediction:", mse)

# 🔹 Step 4: Customer Segmentation using KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
nri_data['customer_segment'] = kmeans.fit_predict(X)

# 🔹 Step 5: Generate MIS Reports Automatically
mis_report = nri_data[['branch_id', 'nri_customers', 'revenue_generated', 'income_generated', 'profit_margin', 'cost_efficiency', 'customer_segment']]
mis_report.to_csv("NRI_Business_MIS_Report.csv", index=False)
print("📄 MIS Report Generated and Saved as 'NRI_Business_MIS_Report.csv'")

# 🔹 Step 6: Visualization of Profit Margin by Branch
plt.figure(figsize=(10, 6))
plt.bar(nri_data['branch_id'], nri_data['profit_margin'], color='skyblue')
plt.title("Profit Margin by Branch")
plt.xlabel("Branch ID")
plt.ylabel("Profit Margin")
plt.show()
