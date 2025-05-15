from typing import List
from transformers import pipeline, AutoTokenizer
import torch
import random
import json

# Create tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Override the default tokenizer.apply_chat_template method
original_apply_chat_template = tokenizer.apply_chat_template
def custom_apply_chat_template(*args, enable_thinking=False, **kwargs):
    return original_apply_chat_template(*args, enable_thinking=enable_thinking, **kwargs)
tokenizer.apply_chat_template = custom_apply_chat_template

# Create pipeline for text generation
pipe = pipeline("text-generation", 
               model="Qwen/Qwen3-8B",
               tokenizer=tokenizer,
               use_fast=True,
               device_map="auto")

cached_abbreviations = []

CONDITIONS = [
    'heart failure', 'atrial arrhythmia', 'atrial fibrillation', 'atrial flutter', 'ventricular arrhythmia',
    'sustained ventricular tachycardia', 'non-sustained ventricular tachycardia', 
    'ventricular fibrillation/unplanned cardiac arrest', 'chronic hypertension', 'hyperlipidemia', 'diabetes',
    'smoking', 'obesity', 'coronary artery disease', 'venous thrombosis', 'intracardiac thrombus', 'stroke',
    'pulmonary embolism', 'obstructive sleep apnea', 'chronic kidney disease', 'infective endocarditis', 'protein-losing enteropathy', 'cirrhosis'
]

def extract_all(texts: List[str]) -> List[str]:
    '''
    Extract all 22 conditions from the given texts. Returns a dictionary of True/False values.
    '''
    conditions_list = "CONDITIONS:\n - " + '\n - '.join(CONDITIONS)

    # Format the prompt, truncating the note to the last 8000 characters of the note
    messages = []
    for text in texts:
        truncated_text = text if len(text) < 8000 else text[:4000] + "\n...\n" + text[-4000:]
        messages.append(
        [
            {
                "role": "user",
                "content": f"Read the following note from a pediatric cardiology unit. Does the note explicitly state that the patient has, or has ever had, any of the conditions in the list below?\n" + conditions_list 
                + f"\n\nRemember that medical abbreviations depend on context, and that presence of a condition in either the past or the present should both result in an answer of True. Include only personal history and not family history. Here is the note:\n---\n{truncated_text}\n---\nNow that you have read the note, decide whether or not the note explicitly states that the patient has, or has ever had, each condition. Answer in the form of a JSON dictionary of the form\n" + 
'''{
    "heart failure": false,
    "atrial arrhythmia": true,
    "atrial fibrillation": false,
    "atrial flutter": true,
    ...
}
'''
                + "Set each condition to true if and only if you are absolutely sure the patient has it. Set it to false if the condition has been ruled out, or if it is not mentioned in the note, or if there is any ambiguity."
            }])
    
    # Generate response
    with torch.no_grad():
        responses = pipe(messages, max_new_tokens=1600)

    # Extract the generated text
    out = [response[0]["generated_text"][-1]["content"].strip()
           for response in responses]
    
    # convert to dictionary
    processed = []
    for x in out:
        start = x.find('{')
        end = x.rfind('}')
        if start != -1 and end != -1:
            json_substr = x[start:end+1]
            try:
                processed.append(json.loads(json_substr))
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                print(json_substr)
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
    print(extract_all([texts[0]]))

