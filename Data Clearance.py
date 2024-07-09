# Run code with command below in terminal
# Command: python3 Assignment2.py stock_data.csv
# The file path have to be in the same folder as the .py file

# Libraries 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


#Question 1
print("Question 1")
file_path = 'stock_data.csv'
df = pd.read_csv(file_path)
print(df.head())

#Question 2
print("Question 2")
all_names = sorted(df['Name'].unique())
print(f"Total number of unique names: {len(all_names)}")
# Display the first 5 names
print("\nFirst 5 names:")
print(all_names[:5])

# Display the last 5 names
print("\nLast 5 names:")
print(all_names[-5:])

#Question 3
print("Question 3")
df['date'] = pd.to_datetime(df['date'])
# Find names to be removed based on date criteria
names_to_remove = df.groupby('Name')['date'].agg([min, max])
names_to_remove = names_to_remove[(names_to_remove['min'] > '2014-07-01') | (names_to_remove['max'] < '2017-06-30')].index
# Remove rows with names to be removed
df_filtered = df[~df['Name'].isin(names_to_remove)]

# Display the names that were removed
print(f"Names removed: {names_to_remove}")

# Display the total number of names left
print(f"\nNumber of names left: {df_filtered['Name'].nunique()}")

#Question 4
print("Question 4")
# Identify the set of dates for each stock
dates_by_name = df_filtered.groupby('Name')['date'].apply(set)

# Find the intersection of dates for all stocks
common_dates = set.intersection(*dates_by_name)

# Filter dates based on specified criteria
start_date = pd.Timestamp('2014-07-01')
end_date = pd.Timestamp('2017-06-30')
filtered_dates = [date for date in common_dates if start_date <= date <= end_date]

# Filter DataFrame based on names and common dates
df_filtered_common_dates = df_filtered[df_filtered['date'].isin(filtered_dates)]

# Display the number of dates left
print(f"Number of dates left: {len(filtered_dates)}")

# Display the first 5 dates
print("\nFirst 5 dates:")
print(filtered_dates[:5])

# Display the last 5 dates
print("\nLast 5 dates:")
print(filtered_dates[-5:])
# Display the total number of dates left
print(f"Number of dates left: {len(filtered_dates)}")

# Display the first 5 dates
print("\nFirst 5 dates:")
for date in filtered_dates[:5]:
    print(date.strftime('%Y-%m-%d'))
# Display the last 5 dates
print("\nLast 5 dates:")
for date in filtered_dates[-5:]:
    print(date.strftime('%Y-%m-%d'))

#Question 5
print("Question 5")

# Filter DataFrame based on names and dates
df_filtered = df[df['Name'].isin(df_filtered['Name']) & df['date'].isin(filtered_dates)]

# Create a new DataFrame with 'Name' as columns, 'Date' as index, and 'Close' as values
pivot_df = pd.pivot_table(df_filtered, values='close', index='date', columns='Name', aggfunc='first')

# Display the new DataFrame
print("New DataFrame with filtered names and dates:")
print(pivot_df)

# Display the new DataFrame
print(pivot_df)

# Question 6
print("Question 6")

# Create a new DataFrame for returns
returns_df = pd.DataFrame()

# Calculate returns for each group (stock) using the provided formula
for name, group in df_filtered.groupby('Name'):
    group['Return'] = 100 * ((group['close'] - group['close'].shift(1)) / group['close'].shift(1))
    returns_df = pd.concat([returns_df, group[['Name', 'date', 'Return']]], ignore_index=True)

# Remove the first row for each stock (no previous date)
pivot_returns_df = returns_df.pivot(index='date', columns='Name', values='Return')
pivot_returns_df = pivot_returns_df.iloc[1:]

# Display the DataFrame with returns pivoted 
print("DataFrame with returns:")
print(pivot_returns_df)


#Question 7
print("Question 7")
# Standardize the data
standardized_returns = (pivot_returns_df - pivot_returns_df.mean()) / pivot_returns_df.std()

# Initialize PCA and fit_transform on the standardized data
pca = PCA()
principal_components = pca.fit_transform(standardized_returns)

# Get the eigenvalues and eigenvectors
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Create a DataFrame to store the results
pc_df = pd.DataFrame(data={'Eigenvalue': eigenvalues, 'Eigenvector': eigenvectors.tolist()})

# Sort the DataFrame by eigenvalues in descending order
pc_df = pc_df.sort_values(by='Eigenvalue', ascending=False)

# Print the top five principal components
print("Top 5 Principal Components:")
print(pc_df.head(5))


print("Question 8")

# Extract explained variance ratios
explained_var_ratios = pca.explained_variance_ratio_

# Calculate cumulative explained variance ratios
cumulative_var_ratios = explained_var_ratios.cumsum()

# Plot the first 20 explained variance ratios
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), explained_var_ratios[:20], marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratios for the First 20 Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

print("Question 9")
# Calculate cumulative variance ratios
cumulative_var_ratios = np.cumsum(explained_var_ratios)

# Plot cumulative variance ratios
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_var_ratios) + 1), cumulative_var_ratios, marker='o', linestyle='-', color='g')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Cumulative Variance')
plt.title('Cumulative Variance Ratios for Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Ratio')
plt.legend()
plt.grid(True)
plt.show()

# Question 10
print("Question 10")

# Normalize the DataFrame from step (6)
normalized_returns_df = (pivot_returns_df - pivot_returns_df.mean()) / pivot_returns_df.std()

# Initialize PCA and fit_transform on the normalized data
pca_normalized = PCA()
principal_components_normalized = pca_normalized.fit_transform(normalized_returns_df)

# Get the eigenvalues and eigenvectors for normalized data
eigenvalues_normalized = pca_normalized.explained_variance_
eigenvectors_normalized = pca_normalized.components_

# Create a DataFrame to store the results for normalized data
pc_df_normalized = pd.DataFrame(data={'Eigenvalue': eigenvalues_normalized, 'Eigenvector': eigenvectors_normalized.tolist()})

# Sort the DataFrame by eigenvalues in descending order for normalized data
pc_df_normalized = pc_df_normalized.sort_values(by='Eigenvalue', ascending=False)

# Print the top five principal components for normalized data
print("Top 5 Principal Components for Normalized Data:")
print(pc_df_normalized.head(5))

# Extract explained variance ratios for normalized data
explained_var_ratios_normalized = pca_normalized.explained_variance_ratio_

# Calculate cumulative explained variance ratios for normalized data
cumulative_var_ratios_normalized = explained_var_ratios_normalized.cumsum()

# Plot the first 20 explained variance ratios for normalized data
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), explained_var_ratios_normalized[:20], marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratios for the First 20 Principal Components (Normalized Data)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

