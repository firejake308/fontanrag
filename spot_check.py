import pandas as pd

# Read the CSV file
df = pd.read_csv('latest_notes_results.csv')

# Filter rows that have at least one True value
# Convert boolean columns to boolean type if they're not already
bool_columns = df.select_dtypes(include=['bool']).columns
if len(bool_columns) == 0:
    print('Converting non-boolean columns')
    # If no boolean columns found, try to convert string 'True'/'False' to boolean
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'True': True, 'False': False})

# Filter rows with at least one True value
print(len(df))
filtered_df = df[df.any(axis=1)]
print(len(filtered_df))

# Transpose the data
transposed_df = filtered_df.set_index('Filepath').transpose().astype(bool)

# Save to new CSV
transposed_df.to_csv('spot_check.csv', header=False)
   
transposed_df.index = transposed_df.index.rename('')
transposed_df.loc[:,transposed_df.sum(axis=0) > 2].sample(n=30, axis=1, random_state=42).to_csv('spot_check30.csv')

print("Processing complete. Results saved to spot_check.csv")