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
               tokenizer=tokenizer,
               use_fast=True,
               torch_dtype='auto',
               device_map="auto")

def extract_boolean(texts: List[str], condition: str) -> List[str]:
    '''
    Extract whether the texts contain evidence for a given condition. Return a list of True/False/None.
    '''
    # Format the prompt, truncating to the first 8000 characters
    messages = [
        {
            "role": "user", 
            "content": f"Tell me if the following note contains definitive evidence that the patient has {condition}. Reply with only 'Yes', 'No', or 'Not sure'. Here is the note:\n---\n{text[:8000]}"
        }
    for text in texts]
    
    # Generate response
    with torch.no_grad():
        responses = pipe(messages, max_new_tokens=32)

    # Extract the generated text
    out = [response[0]["generated_text"][-1]["content"].strip()
           for response in responses]
    
    # convert to boolean
    processed = []
    for x in out:
        x_lower = x.lower()
        if x_lower.startswith('yes'):
            processed.append(True)
        elif x_lower.startswith('no'):
            processed.append(False) 
        elif x_lower.startswith('not sure'):
            processed.append(None)
        else:
            print(f"Unexpected LLM output: {x}")
            processed.append(None)
    return processed


# for testing and debugging
if __name__ == "__main__":
    texts = [
        "The patient is a 50 year old male with a history of hypertension and diabetes. He is currently admitted to the hospital for a heart attack. He is currently on a heart monitor and a heart monitor is currently beeping.",
        "TRANSTHORACIC ECHO\nDate: 04/11/2025 10:00:00 AM\nFindings\nEchocardiogram demonstrates severe left ventricular dysfunction with a left ventricular ejection fraction of 20%.",
        "Assessment: Acute heart failure exacerbation. Recommend hospitalization for further evaluation and treatment. \nPlan: Continue heart failure medications. Monitor daily.",
        "The patient is currently on a heart monitor and a heart monitor is currently beeping.",
        "The patient is currently on a heart monitor and a heart monitor is currently beeping.",
        "The patient is currently on a heart monitor and a heart monitor is currently beeping.",
        "The patient is currently on a heart monitor and a heart monitor is currently beeping.",
        "The patient is currently on a heart monitor and a heart monitor is currently beeping.",
        "The patient is currently on a heart monitor and a heart monitor is currently beeping.",
    ]
    print(extract_boolean([texts[0]], "heart failure"))

