# python data\create_csv.py

import os
import pandas as pd

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Sample data for sentiment analysis
data = {
    'text': [
        'I love this product!',       # Positive
        'This is the worst service.', # Negative
        'Had a great time today.',    # Positive
        'I am so disappointed.',      # Negative
        'Absolutely fantastic!',      # Positive
        'Terrible experience.',       # Negative
        'The food was amazing.',      # Positive
        'I will never come back here.', # Negative
        'Great value for money.',     # Positive
        'Worst purchase ever.',       # Negative
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = 'data/tweets.csv'
df.to_csv(csv_path, index=False)

print(f"CSV file created at {csv_path}")
