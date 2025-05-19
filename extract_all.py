from typing import List
from transformers import pipeline, AutoTokenizer
import torch
import random
import json
import sys

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

# Split conditions into arrhythmias and other conditions
ARRHYTHMIA_CONDITIONS = [
    'atrial arrhythmia', 'atrial fibrillation', 'atrial flutter', 'ventricular arrhythmia',
    'sustained ventricular tachycardia', 'non-sustained ventricular tachycardia', 
    'ventricular fibrillation/unplanned cardiac arrest'
]

OTHER_CONDITIONS = [
    'heart failure', 'chronic hypertension', 'hyperlipidemia', 'diabetes',
    'smoking', 'obesity', 'coronary artery disease', 'venous thrombosis', 'intracardiac thrombus', 'stroke',
    'pulmonary embolism', 'obstructive sleep apnea', 'chronic kidney disease', 'infective endocarditis', 'protein-losing enteropathy', 'cirrhosis'
]

CONDITIONS = [*ARRHYTHMIA_CONDITIONS, *OTHER_CONDITIONS]

def extract_all(texts: List[str], verbose=False) -> List[dict]:
    '''
    Extract all conditions from the given texts in two separate runs - one for arrhythmias and one for other conditions.
    Returns a list of dictionaries with True/False values.
    '''
    arrhythmia_list = "CONDITIONS:\n - " + '\n - '.join(ARRHYTHMIA_CONDITIONS)
    other_conditions_list = "CONDITIONS:\n - " + '\n - '.join(OTHER_CONDITIONS)

    # Format the prompts for both runs
    arrhythmia_messages = []
    other_conditions_messages = []
    
    for text in texts:
        truncated_text = text if len(text) < 8000 else text[:4000] + "\n...\n" + text[-4000:]
        
        # Message for arrhythmias
        arrhythmia_messages.append([
            {
                "role": "user",
                "content": "Read the following note from a pediatric cardiology unit."
                " Does the note explicitly state that the patient has, or has ever had,"
                f" any of the arrhythmias in the list below?\n{arrhythmia_list}\n\n"
                "Include only personal history and not family history."
                " A few tips: Atrial arrythmias include atrial fibrillation, atrial flutter, and also supraventricular tachycardia (SVT), intraatrial reentrant tachycardia (IART), AVNRT, or sinus bradycardia."
                " If you are not sure whether a ventricular tachycardia was sustained or non-sustained, apply the heuristic that defibrillation is most often used for sustained V-tach."
                " AV blocks and junctional rhythms do not count as either atrial or ventricular arrhythmias.\n\n"
                f"Here is the note:\n---\n{truncated_text}\n---\n"
                "Now that you have read the note, decide whether or not the note explicitly states that the patient has, or has ever had, each arrhythmia."
                " Answer in the form of a JSON dictionary of the form\n" + 
'''{
    "atrial arrhythmia": true/false,
    "atrial fibrillation": true/false,
    "atrial flutter": true/false,
    ...
}
'''
                + "Set each arrhythmia to true if and only if you are absolutely sure the patient has it. Set it to false if the condition has been ruled out, or if it is not mentioned in the note, or if there is any ambiguity."
            }])
        
        # Message for other conditions
        other_conditions_messages.append([
            {
                "role": "user",
                "content": f"Read the following note from a pediatric cardiology unit. Does the note explicitly state fthat the patient has, or has ever had, any of the conditions in the list below?\n{other_conditions_list}"
                "\n\nRemember that medical abbreviations depend on context,"
                " and that presence of a condition in either the past or the present should both result in an answer of true."
                " Include only personal history and not family history."
                "\nA few tips: If a patient has failing physiology or depressed ventricular function, those are equivalent to explicitly stating that the patient has heart failure."
                " Also, to help clarify the different thrombotic conditions: a thrombus in one of the atria or ventricles is an intracardiac thrombus; a thrombus in any of the veins prior to the right atrium is a venous thrombosis; and a thrombus in one of the coronary arteries would indicate coronary artery disease, not an intracardiac thrombus."
                " DVT prophylaxis does not count as venous thrombosis; venous thrombosis is only true when the patient actually has a clot."
                f"\nHere is the note:\n---\n{truncated_text}\n---\n"
                "Now that you have read the note, decide whether or not the note explicitly states that the patient has, or has ever had, each condition."
                " Answer in the form of a JSON dictionary of the form\n" + 
'''{
    "heart failure": true/false,
    "chronic hypertension": true/false,
    "hyperlipidemia": true/false,
    ...
}
'''
                + "Set each condition to true if and only if you are absolutely sure the patient has it. Set it to false if the condition has been ruled out, or if it is not mentioned in the note, or if there is any ambiguity."
                # " Explain your reasoning after the JSON"
            }])
    
    # Generate responses for both runs
    with torch.no_grad():
        arrhythmia_responses = pipe(arrhythmia_messages, max_new_tokens=800)
        other_conditions_responses = pipe(other_conditions_messages, max_new_tokens=800)

    # Extract the generated text from both runs
    arrhythmia_out = [response[0]["generated_text"][-1]["content"].strip()
                     for response in arrhythmia_responses]
    other_conditions_out = [response[0]["generated_text"][-1]["content"].strip()
                          for response in other_conditions_responses]
    
    # Process both sets of responses
    processed = []
    for arrhythmia_text, other_conditions_text in zip(arrhythmia_out, other_conditions_out):
        if verbose:
            print(arrhythmia_text)
            print(other_conditions_text)
        # Process arrhythmia results
        arrhythmia_start = arrhythmia_text.find('{')
        arrhythmia_end = arrhythmia_text.rfind('}')
        arrhythmia_dict = {}
        if arrhythmia_start != -1 and arrhythmia_end != -1:
            try:
                arrhythmia_dict = json.loads(arrhythmia_text[arrhythmia_start:arrhythmia_end+1])
            except Exception as e:
                print(f"Error parsing arrhythmia JSON: {e}")
                print(arrhythmia_text[arrhythmia_start:arrhythmia_end+1])
        
        # Process other conditions results
        other_start = other_conditions_text.find('{')
        other_end = other_conditions_text.rfind('}')
        other_dict = {}
        if other_start != -1 and other_end != -1:
            try:
                other_dict = json.loads(other_conditions_text[other_start:other_end+1])
            except Exception as e:
                print(f"Error parsing other conditions JSON: {e}")
                print(other_conditions_text[other_start:other_end+1])
        
        # Combine both dictionaries
        combined_dict = {**arrhythmia_dict, **other_dict}
        processed.append(combined_dict if combined_dict else None)
    
    return processed

# for testing and debugging
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f'Reading {sys.argv[1]}')
        with open(sys.argv[1]) as f:
            extract_all([f.read()], verbose=True)
    else:
        print('Using dummy data')
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

