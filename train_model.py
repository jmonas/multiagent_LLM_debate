import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
import logging

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn as nn

import transformers
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          )
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer


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
            data = json.load(file)  # Load and parse the JSON into a dictionary

        # Assuming data is a list of dictionaries
        self.data = data  # Store the data as an attribute
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = f"What is the answer to: {item['problem']}?"
        answer = item['round_answers'][-1][0]  # Assuming this is the correct answer

        # Tokenize the problem
        tokenized_problem = self.tokenizer.encode_plus(
            problem,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        # Tokenize the answer
        tokenized_answer = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized_problem['input_ids'].flatten(),
            'attention_mask': tokenized_problem['attention_mask'].flatten(),
            'labels': tokenized_answer['input_ids'].flatten(),
            'decoder_attention_mask': tokenized_answer['attention_mask'].flatten()
        }
        
# CUDA Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/gpfs/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.float16, cache_dir="/scratch/gpfs/jmonas/.cache/")
model.to(device)  # Send model to GPU

# Load your custom dataset
dataset_path = 'new_combined_2_agents_3_rounds_results.json'
train_dataset = CustomDataset(dataset_path, tokenizer)




peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=5,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    evaluation_strategy='steps',
    eval_steps = 112,
    eval_accumulation_steps=1,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=125,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained("trained-model")