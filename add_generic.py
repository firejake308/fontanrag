import re
from pathlib import Path
import sqlite3
from datetime import datetime
from extract_boolean import extract_boolean
import sys
import pandas as pd


def add_generic(condition_name: str, column_name: str):
    '''Reads through the latest notes per patient from latest_notes_per_patient.csv and extracts a boolean
    variable representing whether or not the patient in the note has the supplied
    `condition_name`. Saves the result in MedicalRecords.db under the supplied
    `column_name`'''
    
    # Read the latest notes CSV
    df = pd.read_csv('latest_notes_per_patient.csv')
    
    # Prepend the base path to file names
    df['Filepath'] = 'CardioIMS/Notes/NoteText/' + df['File Name']
    
    # create a list to store all file data
    file_data = []

    # Create a list to store files that need LLM processing
    llm_batch = []
    llm_files = []

    # Process each file from the CSV
    for filepath in df['Filepath']:
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                text = f.read()
            
            file_name = Path(filepath).name
            
            # Add to batch for LLM processing
            llm_batch.append(text)
            llm_files.append(file_name)
            
            # Process batch when it reaches size 16
            if len(llm_batch) == 16:
                vals = extract_boolean(llm_batch, condition_name)
                for i, (fname, val) in enumerate(zip(llm_files, vals)):
                    # print(fname, val)
                    file_data.append({
                        'file_name': fname,
                        'val': val,
                    })
                # Clear the batches
                llm_batch = []
                llm_files = []

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    # Process any remaining files in the batch
    if llm_batch:
        vals = extract_boolean(llm_batch, condition_name)
        for i, (fname, val) in enumerate(zip(llm_files, vals)):
            file_data.append({
                'file_name': fname,
                'val': val,
            })

    try:
        # open MedicalRecords.db
        conn = sqlite3.connect("MedicalRecords.db")
        cursor = conn.cursor()

        # check if columns exist before adding them
        cursor.execute("PRAGMA table_info(notes)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE notes ADD COLUMN {column_name} BOOLEAN")

        # insert the sign_time and addend_time into the notes table
        for data in file_data:
            cursor.execute(
                f"UPDATE notes SET {column_name} = ? WHERE file_name = ?",
                (data['val'], data['file_name'])
            )

        # commit the changes and close the connection
        conn.commit()
        print(f"{column_name} added to the notes table")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    # Check if command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python add_generic.py <condition_name> <column_name>")
        sys.exit(1)

    add_generic(condition_name=sys.argv[1], column_name=sys.argv[2])
