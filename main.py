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

number_of_schedule_options = 2

def main(seed_value):
    seed = seed_value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    num_activity = 20
    n_elements = num_activity # Initialize with your number of elements

    relations = np.zeros((20, 20))
    precedence_relations = {4:[1,2],
                 5:[1,2],
                 6:[3],
                 7:[3],
                 8:[4,5,7],
                 9:[4],
                 10:[5,6,7],
                 11:[6],
                 12:[8,10],
                 13:[8],
                 14:[9],
                 15:[11],
                 16:[12],
                 17:[14],
                 18:[13,15],
                 19:[15],
                 20:[16]
    }
    
    # Assume that key supports i. So key should be completed first. 
    for key in precedence_relations.keys():
        for i in precedence_relations[key]:
            relations[key-1][i-1] = 1

    resource_requirements =  {
                0: [5, 3, 2],  
                1: [4, 5, 3],  
                2: [2, 5, 2],  
                3: [1, 4, 4],
                4: [4, 2, 4],  
                5: [5, 5, 4],  
                6: [5, 3, 2],  
                7: [2, 3, 2],
                8: [1, 4, 4],  
                9: [2, 3, 4],  
                10: [3, 3, 2],  
                11: [4, 1, 4],
                12: [5, 5, 4],  
                13: [2, 2, 2],  
                14: [5, 1, 4],
                15: [3, 5, 3],
                16: [2, 3, 3],  
                17: [5, 4, 4],  
                18: [4, 2, 6],  
                19: [0, 4, 1] 
            }
    
    # Triangular distribution of durations
    activity_times =  { # Remember these durations are in days
    0:5,
    1:5,
    2:3,
    3:4,
    4:2,
    5:1,
    6:6,
    7:6,
    8:1,
    9:3,
    10:3,
    11:3,
    12:3,
    13:6,
    14:4,
    15:3,
    16:3,
    17:4,
    18:1,
    19:4
    }

    resource_1_cost = 378.40 # Per day cost of skilled labor as per RS Means
    resource_2_cost = 378.40 # Per day cost of skilled labor as per RS Means
    resource_3_cost = 293.20 # Per day cost of unskilled labor as per RS Means
    resource_1_capacity = 8 
    resource_2_capacity = 8
    resource_3_capacity = 8

    action_space = np.load("/home/aagr657/Documents/ISARC_2026/action_space.npy")

    # Suppose we define N=3 different weight vectors
    weight_vector = [1,1] # Equal weightage to all the three

    project_context = """A residential building construction project, involving 20 activities. Need to create a schedule that minimizes time and cost."""
    
    reward_jsons, prompt = generate_multiple_prompts(project_context, number_of_schedule_options)
    schedules = []
    objectives = []
    parallel_activites = []

    for i, reward_json_str in enumerate(reward_jsons):  # Changed from 'json' to 'reward_json_str'
        reward_json = json.loads(reward_json_str)
        reward_fn = compile_reward(reward_json)
        env_config = {
            "n_elements" : n_elements, 
            "resource_1_cost" : resource_1_cost, 
            "resource_2_cost" : resource_2_cost, 
            "resource_3_cost" : resource_3_cost, 
            "resource_1_capacity" : resource_1_capacity, 
            "resource_2_capacity" : resource_2_capacity, 
            "resource_3_capacity" : resource_3_capacity, 
            "resource_requirements" : resource_requirements, 
            "activity_times" : activity_times, 
            "precedence_relations" : relations,
            "action_space" : action_space, 
            "weight_vector" : weight_vector,
            "reward_fn": reward_fn 
        }

        results = train_one_agent(env_config, weight_vector, seed_value, i) # The results of these asynchronous calls are stored in the results list.
        schedule_df, objective_df, average_parallel_activities = evaluate_fn(seed_value,i, reward_fn)
        schedules.append(schedule_df)
        objectives.append(objective_df)
        parallel_activites.append(average_parallel_activities)
         
        print(results)
    return schedules, objectives, parallel_activites, reward_jsons, prompt

def save_preferences(preferences, filename):
    """
    Save collected preferences to JSON file
    """
    with open(filename, 'w') as f:
        json.dump(preferences, f, indent=2)
    
    print(f"Saved {len(preferences)} preferences to {filename}")

def load_preferences(filename):
    """
    Load preferences from JSON file
    """
    with open(filename, 'r') as f:
        preferences = json.load(f)
    
    return preferences

def to_json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    else:
        return obj

if __name__ == "__main__":
    seed_value = 20
    iterations = 15
    final_Schedules = []
    final_Objectives = []
    final_parallel_activities = []
    all_preferences = []
    for i in range(iterations):
        #schedules, objectives, parallel_activities, reward_jsons, prompt = main(seed_value+i)
        #preference = collect_preference(schedules, objectives, parallel_activities, reward_jsons, prompt)
        #all_preferences.extend(preference)

        # Collecting the schedules and objectives for the existing reward models.
        schedules, objectives, parallel_activities, reward_jsons, prompt = main(seed_value+i)
        
        final_Schedules.extend(schedules)
        final_Objectives.extend(objectives)
        final_parallel_activities.extend(parallel_activities)
    # Preference_filename = "/home/aagr657/Documents/ISARC_2026/dpo_preferences_new4.json"
    # save_preferences(all_preferences, Preference_filename)

    results_updated_model = []
    results_base_model = []

    for i in range(len(final_Schedules)):
        results_base_model.append({
            "run_id": i,
            "schedule": to_json_safe(final_Schedules[i]),
            "objectives": to_json_safe(final_Objectives[i]),
            "parallel_activities": to_json_safe(final_parallel_activities[i])
        })

    output_path = "/home/aagr657/Documents/ISARC_2026/base_model_results.json"
    with open(output_path, "w") as f:
        json.dump(results_base_model, f, indent=2)


