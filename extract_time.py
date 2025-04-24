from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration
import torch

# Create tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create pipeline for text generation
pipe = pipeline("text-generation", 
               model="meta-llama/Llama-3.1-8B-Instruct", 
               torch_dtype=torch.bfloat16,
               tokenizer=tokenizer,
               use_fast=True,
               device="cuda:1")

def extract_note_time(texts: List[str]) -> List[str]:
    # Format the prompt, truncating to the first 8000 characters
    messages =  [
        [{
            "role": "user", 
            "content": [{
                "type": "text", 
                "text": f"Tell me what date the following note was written on. Reply with only the date, formatted as "
                f"MM/DD/YYYY, or NULL if you are not sure. Do not include anything other than a date or NULL. Here is the note:\n---\n{text[:8000]}"
            }]
        }]
    for text in texts]
    # Generate response
    with torch.no_grad():
        responses = pipe(messages, max_new_tokens=32)

    # Extract the generated text
    out = [response[0]["generated_text"][-1]["content"].strip()
           for response in responses]
    # replace NULL with None
    out = [None if x == "NULL" else x for x in out]
    return out


# for testing and debugging
if __name__ == "__main__":
    texts = [
        "This document was electronically signed by the provider John Doe on 04/11/2025 10:00:00 AM in the Epic medical database. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "This document was electronically addended by the provider John Doe on 04/15/2025 10:00:00 AM in the Epic medical database.",
        "This document was electronically signed by the provider John Doe on 04/11/2025 10:00:00 AM in the Epic medical database.",
        "This document was electronically addended by the provider John Doe on 04/15/2025 10:00:00 AM in the Epic medical database.",
        "This document was electronically signed by the provider John Doe on 04/11/2025 10:00:00 AM in the Epic medical database.",
        "This document was electronically addended by the provider John Doe on 04/15/2025 10:00:00 AM in the Epic medical database.",
        "This document was electronically signed by the provider John Doe on 04/11/2025 10:00:00 AM in the Epic medical database.",
        "This document was electronically addended by the provider John Doe on 04/15/2025 10:00:00 AM in the Epic medical database.",
        "Date: 04/11/2025 10:00:00 AM   Title: Test Note 1",
        "Date: 04/15/2025 10:00:00 AM   Title: Test Note 2",
        "Date: 04/11/2025 10:00:00 AM   Title: Test Note 3",
        "Date: 04/15/2025 10:00:00 AM   Title: Test Note 4",
        "Date: 04/11/2025 10:00:00 AM   Title: Test Note 5",
        "Date: 04/15/2025 10:00:00 AM   Title: Test Note 6",
        "Date: 04/11/2025 10:00:00 AM   Title: Test Note 7",
        "Date: 04/15/2025 10:00:00 AM   Title: Test Note 8",
        "Cardiac ICU Note\nDate: April 11, 2025",
        "Cardiac ICU Note\nDate: 4/15/25",

    ]
    print(extract_note_time(texts))

