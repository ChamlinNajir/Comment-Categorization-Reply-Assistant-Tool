import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('comment_dataset.csv')

# 2. Peek at the first few rows to ensure it loaded correctly
print("--- First 5 Rows of Data ---")
print(df.head())

# 3. Check the column names
print("\n--- Column Names ---")
print(df.columns)

# 4. Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 5. Visualize the Category Distribution
# This tells us if we have balanced classes or if one category dominates
plt.figure(figsize=(10,6))
sns.countplot(x='label', data=df) # Replace 'label' with whatever your category column is named
plt.title('Distribution of Comment Categories')
plt.xticks(rotation=45)
plt.show()

