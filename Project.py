import pandas as pd

# Loading dataset
df = pd.read_csv("/Users/rohanbali/Desktop/Advanced Mathematical Stats/Project/globalterrorismdb_0718dist.csv", encoding='latin1')

# Handling missing values in 'nkill' column
df['nkill'] = df['nkill'].fillna(0)

# Categorize=ing attacks
df['AttackCategory'] = pd.cut(
    df['nkill'],
    bins=[-float('inf'), 2, 10, float('inf')],
    labels=['Minor', 'Small', 'Major']
)

# Verifying the categorization
print(df[['iyear', 'nkill', 'AttackCategory']].head())

# Verifying the abundance of data for better accuracy
print(df.info())

import matplotlib.pyplot as plt

# Counting number of attacks by year and category
attacks_by_year = df.groupby(['iyear', 'AttackCategory']).size().unstack(fill_value=0)

# Scatter plot for Major and Minor attacks
plt.figure(figsize=(12, 6))

plt.scatter(attacks_by_year.index, attacks_by_year['Major'], color='red', label='Major Attacks')
plt.scatter(attacks_by_year.index, attacks_by_year['Minor'], color='blue', label='Minor Attacks')

plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.title('Year vs Number of Attacks (Major and Minor)')
plt.legend()
plt.grid(True)
plt.show()

# Calculating moving averages for trend detection
attacks_by_year['Major_MA'] = attacks_by_year['Major'].rolling(window=5).mean()
attacks_by_year['Minor_MA'] = attacks_by_year['Minor'].rolling(window=5).mean()

# Plotting trends
plt.figure(figsize=(12, 6))

plt.plot(attacks_by_year.index, attacks_by_year['Major_MA'], color='red', label='Major Attacks (Trend)')
plt.plot(attacks_by_year.index, attacks_by_year['Minor_MA'], color='blue', label='Minor Attacks (Trend)')

plt.xlabel('Year')
plt.ylabel('Number of Attacks (Moving Average)')
plt.title('Trend Analysis: Year vs Number of Attacks')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np

# Calculating total attacks per year
attacks_by_year['Total'] = attacks_by_year.sum(axis=1)

# Preparing data for regression
X = attacks_by_year.index.values.reshape(-1, 1)  # Year as the independent variable
y = attacks_by_year['Total'].values  # Total attacks as the dependent variable

# Fitting the regression model
model = LinearRegression()
model.fit(X, y)

# Predictions for regression line
y_pred = model.predict(X)

# Scatter plot with regression line
plt.figure(figsize=(12, 6))

plt.scatter(X, y, color='blue', label='Total Attacks')
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('Year')
plt.ylabel('Number of Total Attacks')
plt.title('Linear Regression: Total Attacks vs Year')
plt.legend()
plt.grid(True)
plt.show()

# Output regression details
print(f'Regression Coefficient (Slope): {model.coef_[0]}')
print(f'Regression Intercept: {model.intercept_}')
print(f'R-squared: {model.score(X, y)}')