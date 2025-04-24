import os
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import defaultdict

def search_files(directory):
    """
    Search for 'fib' (case-insensitive) or 'AF' (case-sensitive) in all .txt files in the given directory.
    Returns a dictionary mapping search terms to lists of filenames that contain each term.
    """
    file_categories = {
        'afib': [],
        'a-fib': [],
        'fibrillation': [],
        'AF': []
    }
    file_count = 0
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_count += 1
                if file_count % 1000 == 0:
                    print(f"Checked {file_count} files...")
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'afib' in content.lower():
                            file_categories['afib'].append(file)
                        elif 'a-fib' in content.lower():
                            file_categories['a-fib'].append(file)
                        elif 'fibrillation' in content.lower():
                            file_categories['fibrillation'].append(file)
                        elif 'AF' in content:
                            file_categories['AF'].append(file)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    print(f"Total files checked: {file_count}")
    for category, files in file_categories.items():
        print(f"Found '{category}': {len(files)} files")
    
    return file_categories

def stratified_split(file_categories):
    """
    Perform stratified split based on search terms.
    Takes 5 files per category for validation, 5 for test, and ignores the rest.
    Returns validation and test lists of filenames.
    """
    validation_files = []
    test_files = []
    
    for category, files in file_categories.items():
        if len(files) >= 8:
            # Randomly shuffle the files
            random.shuffle(files)
            # Take first 5 for validation, next 5 for test
            validation_files.extend(files[:4])
            test_files.extend(files[4:8])
        else:
            print(f"Warning: Category '{category}' has only {len(files)} files, skipping")
    
    return validation_files, test_files

def main():
    # Directory containing the text files
    directory = 'CardioIMS/Notes/NoteText'
    
    # Search for files containing the different terms
    file_categories = search_files(directory)
    
    if any(file_categories.values()):
        # Perform stratified split
        val_files, test_files = stratified_split(file_categories)
        
        # Save validation set filenames
        with open('validation_set.txt', 'w') as f:
            for file in val_files:
                f.write(f"{file}\n")
        
        # Save test set filenames
        with open('test_set.txt', 'w') as f:
            for file in test_files:
                f.write(f"{file}\n")
        
        print(f"\nValidation set ({len(val_files)} files) saved to validation_set.txt")
        print(f"Test set ({len(test_files)} files) saved to test_set.txt")
    else:
        print("No files containing the search terms were found.")

if __name__ == "__main__":
    main() 