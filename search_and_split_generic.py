import os
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import defaultdict

def search_files(directory, search_queries: List[str]):
    """
    Search for the `search_queries` in all .txt files in the given directory.
    Returns a dictionary mapping search terms to lists of filenames that contain each term.
    """
    file_categories = {q: [] for q in search_queries}
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
    Returns a dictionary mapping each category to a list of up to 2 random file contents,
    ensuring files come from different patients.
    """
    category_contents = {}
    
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
        
        if len(unique_patient_files) >= 1:
            # Randomly shuffle the files
            random.shuffle(unique_patient_files)
            # Take up to 2 files
            selected_files = unique_patient_files[:2]
            
            # Read the contents of selected files
            file_contents = []
            for file in selected_files:
                try:
                    file_path = os.path.join('CardioIMS/Notes/NoteText', file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Find the position of the search term
                        if category in content.lower():
                            pos = content.lower().find(category)
                            # Get 1000 chars before and after
                            start = max(0, pos - 1000)
                            end = min(len(content), pos + 1000)
                            file_contents.append("..." + content[start:end] + "...")
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            category_contents[category] = file_contents
        else:
            print(f"Warning: Category '{category}' has no unique patient files")
            category_contents[category] = []
    
    return category_contents

def search_generic(search_queries: List[str]):
    """
    Search for the `search_queries` in all of the clinical notes using case-insensitive exact match.
    Returns a list containing excerpts from a random selection of matching notes.
    """
    # Directory containing the text files
    directory = 'CardioIMS/Notes/NoteText'
    
    # Search for files containing the different terms
    file_categories = search_files(directory, search_queries)
    
    if any(file_categories.values()):
        # Perform stratified split
        category_contents = stratified_split(file_categories)
        
        # Flatten category_contents into a single list
        flattened_contents = []
        for contents in category_contents.values():
            flattened_contents.extend(contents)
        
        return flattened_contents
    else:
        return "No files containing the search terms were found."