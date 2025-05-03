import os
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import defaultdict

def search_files(directory):
    """
    Search for phrases related to smoking in all .txt files in the given directory.
    Returns a dictionary mapping search terms to lists of filenames that contain each term.
    """
    file_categories = {
        'smoking': [],
        'smoker': [],
        'cigar': [],
        'tobacco': [],
        'ppd': [],
        'packs per day': [],
        # 'vape': [],
        # 'vaping': [],
        # 'nicotin': [],
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
                        for key in file_categories.keys():
                            if key in content.lower():
                                file_categories[key].append(file)
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
        # Group files by patient
        patient_files = defaultdict(list)
        for file in files:
            try:
                patient_name = file.split('_ADMISSION_')[0]
                patient_files[patient_name].append(file)
            except IndexError:
                print(f"Warning: File '{file}' does not match the expected naming format, skipping")
        
        # Flatten the list while ensuring no two files come from the same patient
        unique_patient_files = []
        for patient, patient_file_list in patient_files.items():
            unique_patient_files.extend(patient_file_list[:1])  # Take one file per patient
        
        if len(unique_patient_files) >= 8:
            # Randomly shuffle the files
            random.shuffle(unique_patient_files)
            # Take first 4 for validation, next 4 for test
            validation_files.extend(unique_patient_files[:4])
            test_files.extend(unique_patient_files[4:8])
        elif len(unique_patient_files) >= 2:
            # Randomly shuffle the files
            random.shuffle(unique_patient_files)
            # Take first 1 for validation, next 1 for test
            validation_files.extend(unique_patient_files[:1])
            test_files.extend(unique_patient_files[1:2])
        else:
            print(f"Warning: Category '{category}' has only {len(unique_patient_files)} unique patient files, ")
    
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