import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import itertools
import gymnasium as gym
from gymnasium import spaces
from torch.distributions import Categorical
from tqdm import tqdm
from collections import deque
import torch
import time
import gc
from itertools import product
from train import train_one_agent
from reward import calculate_reward
from llm_reward_generator import generate_multiple_prompts
from jsontopython import compile_reward
import json
from eval import evaluate_fn
from preference_collector import collect_preference
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import torch
import gc
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from transformers import TrainerCallback

# Clear any existing models from GPU
torch.cuda.empty_cache()
gc.collect()

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.reward_margins = []
        self.reward_accuracies = []
        self.losses = []
        self.epochs = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'rewards/margins' in logs:
                self.reward_margins.append(logs['rewards/margins'])
            if 'rewards/accuracies' in logs:
                self.reward_accuracies.append(logs['rewards/accuracies'])
            if 'loss' in logs:
                self.losses.append(logs['loss'])
            if 'epoch' in logs:
                self.epochs.append(logs['epoch'])

def load_preferences(filename):
    """
    Load preferences from JSON file
    """
    with open(filename, 'r') as f:
        preferences = json.load(f)
    
    return preferences

filename = "/home/aagr657/Documents/ISARC_2026/dpo_preferences.json"
preference_dataset = load_preferences(filename)
filename = "/home/aagr657/Documents/ISARC_2026/dpo_preferences_new.json"
preference_dataset_new = load_preferences(filename)
filename = "/home/aagr657/Documents/ISARC_2026/dpo_preferences_new2.json"
preference_dataset_new2 = load_preferences(filename)
filename = "/home/aagr657/Documents/ISARC_2026/dpo_preferences_new3.json"
preference_dataset_new3 = load_preferences(filename)
filename = "/home/aagr657/Documents/ISARC_2026/dpo_preferences_new4.json"
preference_dataset_new4 = load_preferences(filename)

combined_data = preference_dataset + preference_dataset_new + preference_dataset_new2 + preference_dataset_new3 + preference_dataset_new4
dataset = Dataset.from_list(combined_data)
# print(dataset)

# Load your base model (same as before)
model_name = "Qwen/Qwen2.5-3B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             device_map="auto", 
                                             torch_dtype=torch.float16)

base_model.config.use_cache = False

# Apply LoRA
"""lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)"""

# model = get_peft_model(model, lora_config)

updated_model = PeftModel.from_pretrained(base_model, "/home/aagr657/Documents/ISARC_2026/dpo_reward_model_iter3/checkpoint-240", is_trainable=True)

# Update training config
training_args = DPOConfig(
    output_dir="./dpo_reward_model_iter4",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    logging_steps=5,
    save_steps=50,
    beta=0.1,
    gradient_checkpointing=False,
    fp16=True  # Use mixed precision
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

metrics_callback = MetricsCallback()

trainer = DPOTrainer(
    model=updated_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[metrics_callback]    # Changed from 'tokenizer' to 'processing_class'
)

# Start training
trainer.train()

np.save('/home/aagr657/Documents/ISARC_2026/reward_margins_4.npy', metrics_callback.reward_margins)