import sqlite3
import pandas as pd
from pathlib import Path

# SQL script to initialize the database
sql_script = open('init_db.sql', 'r').read()

# Initialize SQLite database
conn = sqlite3.connect('MedicalRecords.db')
cursor = conn.cursor()

# Execute SQL script to create tables
cursor.executescript(sql_script)

# Load Excel file into a DataFrame
excel_file = 'CardioIMS\Donhchadh OSullivan MD-CardioIMS-Large Language Model Pilot Fontan patient notes-SCTASK0136992\Patient List.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(excel_file, engine='openpyxl')

# Populate the patients table with data from the Excel file
df.to_sql('patients', conn, if_exists='append', index=False)

# populate the notes table with data from the NoteText folder
source_dir = Path("CardioIMS/Notes/NoteText")
# for each file in source_dir, extract the file name, mrn, and note text
# Store them in a list so that we can INSERT INTO the notes table
notes_list = []
for file in source_dir.glob("*.txt"):
    try:
        # extract the file name and MRN
        file_name = file.name
        mrn = file.stem.split('_')[2]  # Extract MRN from the filename pattern
        notes_list.append((file_name, mrn))
    except IndexError as e:
        print(f"Error reading file: {file.stem}")
        raise e

# Insert the notes into the notes table
cursor.executemany("INSERT INTO notes (FILE_NAME, MRN) VALUES (?, ?)", notes_list)


# Commit changes and close the connection
conn.commit()
conn.close()

print("Database initialized and patients table populated successfully.")