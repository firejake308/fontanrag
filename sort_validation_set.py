def sort_validation_set():
    try:
        # Read the validation set file
        with open('validation_set.txt', 'r') as f:
            lines = f.readlines()
        
        # Prepend path to each line and remove any existing newlines
        lines = [f"CardioIMS/Notes/NoteText/{line.strip()}\n" for line in lines]
        
        # Sort the lines alphabetically
        sorted_lines = sorted(lines)
        
        # Write the sorted lines back to the file
        with open('validation_set.txt', 'w') as f:
            f.writelines(sorted_lines)
            
        print("Validation set has been sorted alphabetically with paths prepended.")
        
    except FileNotFoundError:
        print("Error: validation_set.txt not found. Please run search_and_split.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sort_validation_set() 