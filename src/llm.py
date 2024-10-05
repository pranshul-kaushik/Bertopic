import os

import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_LIST = ["meta-llama/Meta-Llama-3.1-8B-Instruct"]
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL = os.environ.get("MODEL_ID")

TITLE = "<h1><center>Meta-Llama3.1-8B</center></h1>"

PLACEHOLDER = """
<center>
<p>Hi! How can I help you today?</p>
</center>
"""


CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""

device = "cuda" # for GPU usage or "cpu" for CPU usage

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= "nf4")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config)

@spaces.GPU()
def chat(
    message: str, 
    history: list,
    system_prompt: str,
    temperature: float = 0.8, 
    max_new_tokens: int = 1024, 
    top_p: float = 1.0, 
    top_k: int = 20, 
    penalty: float = 1.2,
):
    print(f'message: {message}')
    print(f'history: {history}')

    # Construct the conversation context
    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": answer},
        ])

    conversation.append({"role": "user", "content": message})

    # Tokenize the conversation input
    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Define the generation parameters
    generate_kwargs = dict(
        input_ids=input_ids, 
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature == 0 else True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=penalty,
        eos_token_id=[128001,128008,128009],  # Define the end-of-sequence token
    )

    # Generate the output
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    # Decode the output into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response
