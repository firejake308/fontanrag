import pandas as pd
from extract_boolean import extract_boolean
import os

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
    df = pd.read_csv('validation_set.csv')
    
    # Process each row
    results = []
    for idx, row in df.iterrows():
        # Read the text file
        text = read_text_file(row['Filepath'])
        
        # Check for atrial fibrillation
        print(row['Filepath'])
        afib_result = extract_boolean([text], 'smoking')[0]
        
        # Convert None to False
        if afib_result is None:
            afib_result = False
            
        # Compare with label
        label = bool(row['Label'])
        is_correct = afib_result == label
        
        results.append({
            'Filepath': row['Filepath'],
            'Predicted': afib_result,
            'Label': label,
            'Correct': is_correct
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy
    accuracy = results_df['Correct'].mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Save results to CSV
    results_df.to_csv('validation_results.csv', index=False)
    print("Results saved to validation_results.csv")

if __name__ == "__main__":
    main() 