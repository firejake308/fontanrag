import sqlite3
import os
import csv

def execute_sql_file(cursor, sql_file):
    with open(sql_file, 'r') as file:
        sql_script = file.read()
        cursor.executescript(sql_script)

def get_latest_notes_per_patient():
    # Connect to the database
    conn = sqlite3.connect('MedicalRecords.db')
    cursor = conn.cursor()
    
    try:
        # Execute the SQL file
        execute_sql_file(cursor, 'note_date.sql')
        
        # Query for the latest note per patient
        cursor.execute("""
            WITH ranked_notes AS (
                SELECT 
                    FILE_NAME,
                    MRN,
                    CASE
                        WHEN sign_time IS NOT NULL THEN sign_time
                        WHEN addend_time IS NOT NULL THEN addend_time
                        WHEN note_time IS NOT NULL THEN note_time
                        ELSE NULL
                    END AS note_date,
                    ROW_NUMBER() OVER (PARTITION BY MRN ORDER BY 
                        CASE
                            WHEN sign_time IS NOT NULL THEN sign_time
                            WHEN addend_time IS NOT NULL THEN addend_time
                            WHEN note_time IS NOT NULL THEN note_time
                            ELSE NULL
                        END DESC
                    ) as rn
                FROM notes
                WHERE MRN IS NOT NULL
            )
            SELECT FILE_NAME, MRN, note_date
            FROM ranked_notes
            WHERE rn = 1
            ORDER BY MRN
        """)
        
        results = cursor.fetchall()
        
        # Write results to CSV
        with open('latest_notes_per_patient.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'MRN', 'Note Date'])  # Header
            writer.writerows(results)
            
        print(f"Successfully wrote {len(results)} records to latest_notes_per_patient.csv")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    get_latest_notes_per_patient() 