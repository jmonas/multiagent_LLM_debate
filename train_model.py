import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LossLoggingCallback(TrainerCallback):
    """ Custom callback for logging loss during training. """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            logger.info(f"Step {state.global_step}: Loss {logs['loss']}")

# Load and prepare the custom JSON dataset
class CustomDataset(Dataset): 
    def __init__(self, json_path, tokenizer, max_length=125,):
        with open(json_path, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = f"What is the answer to: {item['problem']}?"
        answer = item['round_answers'][-1][0]  # Assuming the last answer is the correct one to learn

        # Tokenize problem and answer separately to set answer as labels
        input_encodings = self.tokenizer(problem, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
        label_encodings = self.tokenizer(answer, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")

        # Prepare input_ids and labels
        input_ids = input_encodings['input_ids'].squeeze(0)
        labels = label_encodings['input_ids'].squeeze(0)

        # Masks out padding from labels, replace with -100 which is ignored by the loss function
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {'input_ids': input_ids, 'labels': labels}
    
# CUDA Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/gpfs/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.float16, cache_dir="/scratch/gpfs/jmonas/.cache/")

# Load your custom dataset
dataset_path = 'new_combined_2_agents_3_rounds_results.json'
train_dataset = CustomDataset(dataset_path, tokenizer)

# Trainer setup
training_args = TrainingArguments(
    output_dir='/scratch/gpfs/jmonas/gemma2B',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[LossLoggingCallback()]  # Add your custom callback here
)

# Start training
trainer.train()
