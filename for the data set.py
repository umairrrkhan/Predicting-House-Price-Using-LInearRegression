import numpy as np
import pandas as pd

# Create a sample dataset
n = 1000
np.random.seed(42)
df = pd.DataFrame({
    'id': range(1, n+1),
    'date': pd.date_range(start='2020-01-01', periods=n),
    'bedrooms': np.random.randint(low=1, high=6, size=n),
    'bathrooms': np.random.uniform(low=1.0, high=4.0, size=n),
    'sqft_living': np.random.randint(low=500, high=5000, size=n),
    'sqft_lot': np.random.randint(low=5000, high=10000, size=n),
    'floors': np.random.choice([1, 1.5, 2], size=n),
    'waterfront': np.random.choice([0, 1], size=n),
    'view': np.random.choice([0, 1, 2, 3, 4], size=n),
    'condition': np.random.choice([1, 2, 3, 4, 5], size=n),
    'grade': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n),
    'sqft_above': np.random.randint(low=500, high=5000, size=n),
    'sqft_basement': np.random.randint(low=0, high=1000, size=n),
    'yr_built': np.random.randint(low=1900, high=2022, size=n),
    'yr_renovated': np.random.choice([0, 1, 2, 3, 4], size=n),
    'zipcode': np.random.randint(low=98001, high=98199, size=n),
    'lat': np.random.uniform(low=47.1, high=47.7, size=n),
    'long': np.random.uniform(low=-122.5, high=-121.5, size=n),
    'sqft_living15': np.random.randint(low=500, high=5000, size=n),
    'sqft_lot15': np.random.randint(low=5000, high=10000, size=n),
    'price': np.random.randint(low=100000, high=1000000, size=n),
    'crime_rate': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15], size=n)
})

# Save the dataset to a CSV file
df.to_csv('housing_prices.csv', index=False)





