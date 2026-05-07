import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, "data", "training_data.txt")
model_path = os.path.join(project_dir, "models")

# Load training data
with open(data_path, "r", encoding="utf-8") as f:
    training_data = f.read()

print(f"Loaded {len(training_data)} characters of training data.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
tokens = tokenizer(training_data, return_tensors="pt", truncation=True, max_length=512)
print(f"Tokenized into {tokens['input_ids'].shape[1]} tokens.")

# Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print("Model loaded.")


# Dataset wrapper
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


dataset = TextDataset(tokens)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training
training_args = TrainingArguments(
    output_dir=model_path,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("Training complete. Model saved.")
