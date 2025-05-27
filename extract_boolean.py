from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration
import torch

# Create tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Override the default tokenizer.apply_chat_template method to disable thinking
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

def extract_boolean(texts: List[str], condition: str) -> List[str]:
    '''
    Extract whether the texts contain evidence for a given condition. Return a list of True/False/None.
    '''

    # Format the prompt, truncating to the first 8000 characters
    messages = [
        [
            {
                "role": "user",
                "content": f"Read the following note from a pediatric cardiology unit. Does the note explicitly state that the patient has, or has ever had, {condition}? Remember that medical abbreviations depend on context, and that presence of {condition} in either the past or the present should both result in a final answer of Yes. Include only personal history and not family history. Here is the note:\n---\n{text[-8000:]}\n---\nNow that you have read the note, does it explicitly state that the patient has, or has ever had, {condition}? Explain your reasoning, and then report your final answer on a separate line containing only 'Final Answer: Yes/No/Not sure'."
            }]
    for text in texts]
    
    # Generate response
    with torch.no_grad():
        responses = pipe(messages, max_new_tokens=512)

    # Extract the generated text
    out = [response[0]["generated_text"][-1]["content"].strip()
           for response in responses]
    
    # convert to boolean
    processed = []
    for x in out:
        # print(x)
        x_lower = x.lower().split('final answer:')[-1].strip()
        if x_lower.startswith('yes'):
            processed.append(True)
        # order matters here!
        elif x_lower.startswith('not sure'):
            processed.append(None)
        elif x_lower.startswith('no'):
            processed.append(False) 
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
    print(extract_boolean(texts, "heart failure"))

