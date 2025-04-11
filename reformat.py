import json
import os
from pathlib import Path

# Define source and target directories
source_dir = Path("CardioIMS/Notes/Admission/json")
target_dir = Path("CardioIMS/Notes/NoteText")

# Create target directory if it doesn't exist
target_dir.mkdir(parents=True, exist_ok=True)

# Process each JSON file
for json_file in source_dir.glob("*.json"):
    try:
        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text field
        text = data.get('text', '')
        
        # Create output filename (same name but with .txt extension)
        output_file = target_dir / f"{json_file.stem}.txt"
        
        # Write text to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"Processed {json_file.name}")
        
    except Exception as e:
        print(f"Error processing {json_file.name}: {str(e)}")

print("Processing complete!")
