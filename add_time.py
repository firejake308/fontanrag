import re
from pathlib import Path
import sqlite3
from datetime import datetime
from extract_time import extract_note_time

sign_time = r"This document was electronically signed by the provider .+ on (\d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d [AP]M) in the Epic medical database."
addend_time = r"This document was electronically addended by the provider .+ on (\d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d [AP]M) in the Epic medical database."
# the non-Epic notes have a non-uniform date format, which will require an LLM

source_dir = Path("CardioIMS/Notes/NoteText")

# create a list to store all file data
file_data = []

# Create a list to store files that need LLM processing
llm_batch = []
llm_files = []

# search for sign_time, and if that is not found, addend_time, in each .txt file in source_dir
for file in source_dir.glob("*.txt"):
    with open(file, "r", encoding='utf-8') as f:
        text = f.read()
    
    file_name = file.name
    sign_time_match = re.search(sign_time, text)
    addend_time_match = re.search(addend_time, text)
    
    if not sign_time_match and not addend_time_match:
        # Add to batch for LLM processing
        llm_batch.append(text)
        llm_files.append(file_name)
        
        # Process batch when it reaches size 16
        if len(llm_batch) == 8:
            note_times = extract_note_time(llm_batch)
            for i, (fname, note_time) in enumerate(zip(llm_files, note_times)):
                print(fname, note_time)
                if note_time is not None:
                    try:
                        note_date = datetime.strptime(note_time, '%m/%d/%Y').date()
                    except ValueError:
                        try:
                            # sometimes the date is in the format MM/DD/YY
                            note_date = datetime.strptime(note_time[:-2]+'20'+note_time[-2:], '%m/%d/%Y').date()
                        except ValueError:
                            # and sometimes the LLM returns None
                            note_date = None
                    file_data.append({
                        'file_name': fname,
                        'sign_time': None,
                        'addend_time': None,
                        'note_time': note_date,
                    })
            # Clear the batches
            llm_batch = []
            llm_files = []
    else:
        # store the data for this file
        sign_time_str = sign_time_match.group(1) if sign_time_match else None
        addend_time_str = addend_time_match.group(1) if addend_time_match else None
        
        # Convert strings to datetime objects
        sign_time_date = datetime.strptime(sign_time_str.split()[0], '%m/%d/%Y').date() if sign_time_str else None
        addend_time_date = datetime.strptime(addend_time_str.split()[0], '%m/%d/%Y').date() if addend_time_str else None
        
        file_data.append({
            'file_name': file_name,
            'sign_time': sign_time_date,
            'addend_time': addend_time_date,
            'note_time': None
        })


if llm_batch:
    # extract the note times from the LLM batch
    note_times = [extract_note_time(text) for text in llm_batch]
    for i, (fname, note_time) in enumerate(zip(llm_files, note_times)):
        print(fname, note_time)
        if note_time is not None:
            file_data.append({
                'file_name': fname,
                'sign_time': None,
                'addend_time': None,
                'note_time': note_time
            })

try:
    # open MedicalRecords.db
    conn = sqlite3.connect("MedicalRecords.db")
    cursor = conn.cursor()

    # check if columns exist before adding them
    cursor.execute("PRAGMA table_info(notes)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'sign_time' not in columns:
        cursor.execute("ALTER TABLE notes ADD COLUMN sign_time DATE")
    if 'addend_time' not in columns:
        cursor.execute("ALTER TABLE notes ADD COLUMN addend_time DATE") 
    if 'note_time' not in columns:
        cursor.execute("ALTER TABLE notes ADD COLUMN note_time DATE")

    # insert the sign_time and addend_time into the notes table
    for data in file_data:
        cursor.execute(
            "UPDATE notes SET sign_time = ?, addend_time = ?, note_time = ? WHERE file_name = ?",
            (data['sign_time'], data['addend_time'], data['note_time'], data['file_name'])
        )

    # commit the changes and close the connection
    conn.commit()
    print("Sign, addend, and note times added to the notes table")

except sqlite3.Error as e:
    print(f"Database error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
