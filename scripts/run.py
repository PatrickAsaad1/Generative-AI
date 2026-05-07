import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load your trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_path = os.path.join(project_dir, "models")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)

print("Model loaded. Type 'exit' to quit.\n")
# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_new_tokens=50, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"AI: {response}\n")
