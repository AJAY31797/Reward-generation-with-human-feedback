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
from reward import calculate_reward

"""
def collect_preference(schedules, objectives, parallel_activities, reward_jsons, prompt):
    
    n = len(schedules)
    
    for i in range(n):
        print("\n" + "="*50)
        print(f"SCHEDULE {i+1}:")
        print("="*50)
        print(schedules[i].to_string(index=False))
        print(f"\nOBJECTIVES {i+1}:")
        print(objectives[i].to_string(index=False))
        print(f"\nPARALLEL ACTIVITIES {i+1}:")
        print(parallel_activities[i])
    
    # Get valid choice
    valid_choices = [str(i+1) for i in range(n)]
    choice = input(f"\nWhich schedule do you prefer? ({'/'.join(valid_choices)}): ").strip()
    
    while choice not in valid_choices:
        choice = input(f"Invalid. Choose {'/'.join(valid_choices)}: ").strip()
    
    chosen_idx = int(choice) - 1
    
    # Create preference pairs (chosen vs all others)
    preferences = []
    for i in range(n):
        if i != chosen_idx:
            preferences.append({
                "prompt":prompt,
                "chosen": reward_jsons[chosen_idx],
                "rejected": reward_jsons[i]
            })
    
    return preferences
"""

def collect_preference(schedules, objectives, parallel_activities, reward_jsons, prompt):
    
    n = len(schedules)
    schedule_costs = []
    schedule_times = []

    for i in range(n):
        # print(objectives[i]['Schedule_cost'].values[0])
        # print(objectives[i]['Schedule_time'].values[0])
        schedule_costs.append(objectives[i]['Schedule_cost'].values[0])
        schedule_times.append(objectives[i]['Schedule_time'].values[0])


    if schedule_costs[0]<schedule_costs[1] and schedule_times[0]<schedule_times[1]:
        choice = 0
    elif schedule_costs[1]<schedule_costs[0] and schedule_times[1]<schedule_times[0]:
        choice = 1
    elif schedule_costs[0]==schedule_costs[1] and schedule_times[0]<schedule_times[1]:
        choice = 0
    elif schedule_costs[1]==schedule_costs[0] and schedule_times[1]<schedule_times[0]:
        choice = 1
    elif schedule_costs[0]<schedule_costs[1] and schedule_times[0]==schedule_times[1]:
        choice = 0
    elif schedule_costs[1]<schedule_costs[0] and schedule_times[1]==schedule_times[0]:
        choice = 1
    else:
        choice = 0 if parallel_activities[0]>parallel_activities[1] else 1

    chosen_idx = choice
    
    # Create preference pairs (chosen vs all others)
    preferences = []
    for i in range(n):
        if i != chosen_idx:
            preferences.append({
                "prompt":prompt,
                "chosen": reward_jsons[chosen_idx],
                "rejected": reward_jsons[i]
            })
    
    return preferences