import pandas as pd
from extract_all import extract_all, CONDITIONS
import time

def read_text_file(filepath):
    """Read the contents of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return ""

def main():
    # Read the validation set CSV
    df = pd.read_csv('validation_set.csv').set_index('Filepath').T.reset_index(names='Filepath')
    df.index = df.index.rename('idx')

    # record start time
    start_time = time.time()
    
    # Process each row
    results = []
    for idx, row in df.iterrows():
        # Read the text file
        text = read_text_file(row['Filepath'])
        print(row['Filepath'])

        # extract_all returns a dictionary where each key matches a column of df (except for df['Filepath'])
        results.append({'Filepath': row['Filepath'], **extract_all([text])[0]})
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy for each condition, which is a separate column in df_results
    correct = 0
    total = 0

    for condition in CONDITIONS:
        correct += (results_df[condition] == df[condition]).sum()
        total += len(df[condition])

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Save results to CSV
    results_df.to_csv('validation_results.csv', index=False)
    print("Results saved to validation_results.csv")

    print("Elapsed time", time.time() - start_time)

if __name__ == "__main__":
    main() 