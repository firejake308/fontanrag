import pandas as pd
from extract_all import extract_all
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
    # Read the latest notes CSV
    df = pd.read_csv('latest_notes_per_patient.csv')
    
    # Prepend the base path to file names
    df['Filepath'] = 'CardioIMS/Notes/NoteText/' + df['File Name']
    
    # record start time
    start_time = time.time()
    
    # Process each row
    results = []
    for idx, row in df.iterrows():
        # Read the text file
        text = read_text_file(row['Filepath'])
        print(f"Processing {row['Filepath']}")

        # extract_all returns a dictionary with condition results
        results.append({'Filepath': row['Filepath'], **extract_all([text])[0]})
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv('latest_notes_results.csv', index=False)
    print("Results saved to latest_notes_results.csv")

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 