import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

housing_data = pd.read_csv('housing_prices.csv')

# Check for missing values
print(housing_data.isnull().sum())

# Remove duplicates
housing_data.drop_duplicates(inplace=True)

# Handle outliers
q1 = housing_data['price'].quantile(0.25)
q3 = housing_data['price'].quantile(0.75)
iqr = q3 - q1
housing_data = housing_data[(housing_data['price'] > q1 - 1.5 * iqr) & (housing_data['price'] < q3 + 1.5 * iqr)]

# Convert categorical variables into numerical
housing_data = pd.get_dummies(housing_data)

# Perform feature scaling
scaler = StandardScaler()
housing_data[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']] = scaler.fit_transform(housing_data[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']])

# Split the dataset into training and testing sets

X = housing_data.drop('price', axis=1)
y = housing_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering
housing_data['age'] = 2023 - housing_data['yr_built']
housing_data['dist_city_center'] = np.sqrt((housing_data['lat'] - 47.6) ** 2 + (housing_data['long'] + 122.3) ** 2)
housing_data['crime_rate'] = np.log(housing_data['crime_rate'])

# Print the first few rows of the preprocessed data
print(housing_data.head())

# Train a Linear Regression model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate the Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score

y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R2 Score:', r2)



df = pd.read_csv('housing_prices.csv')

# Histogram of target variable
plt.hist(df['price'], bins=20)
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# Scatter plot of sqft_living vs price
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.show()

# Box plot of bedrooms vs price
sns.boxplot(x='bedrooms', y='price', data=df)
plt.show()


