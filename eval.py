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
from env import PrecastSchedulingEnv
from models import ActorCritic
from ppo_agent import Agent
import os

def count_parallel_activities(schedule_df):
    # Get all unique time points where tasks start or end
    time_points = sorted(set(schedule_df["Start times"]) | set(schedule_df["Finish times"]))
    
    parallel_counts = []
    
    for t in time_points:
        # Count active tasks at this time point
        active = ((schedule_df["Start times"] <= t) & 
                  (schedule_df["Finish times"] > t)).sum()
        parallel_counts.append(active)
    
    # max_parallel = max(parallel_counts)
    avg_parallel = np.mean(parallel_counts)
    
    return avg_parallel

def evaluate_fn(seed_value, iteration, reward_fn):
    """
    Used to compute the objective values. 
    parameters : The updated actor network parameters
    index : should be the index of the corresponding agent, whose parameters should be used for the update
    """ 
    seed = seed_value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Try with different feature sizes. Right now I am considering just one of them.
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

    # Suppose we define N=3 different weight vectors
    weight_vector = [1,1] # Equal weightage to all the three

    action_space = np.load("/home/aagr657/Documents/ISARC_2026/action_space.npy")

    node_features_dim = 5
    gcn_hidden_dim = 64
    gcn_output_dim = 64
    resource_availability_feature_size = 32
    remaining_elements_feature_size = 32
    action_mask_feature_size = 32
    structural_support_feature_size = 64
    hidden_dim = 256 
    actor_hidden_dim = 256
    critic_hidden_dim = 256
    dropout_rate = 0
    lr = 0.00025
    gamma = 0.99
    entropy_coef = 0.01
    clip_epsilon = 0.1 
    gae_lambda = 0.95 # Need to see how to decide this value.
    K_epochs = 3
    minibatch_size = 248
    max_timesteps = 1000 # Maximum timesteps per episode
    
    main_dir = f"/home/aagr657/Documents/ISARC_2026/PPO+GNN_{seed_value}_{iteration}"
    parameter_storage_location = os.path.join (main_dir, f"Validation_ISARC2026_{seed_value}")

    # parameter_storage_location = "D:/Ajay/Precast_Assembly_Scheduling/Training_models/Improved2_Singleweather_RewardNorm_CostRewardDeltat/Paper_3_Validation_Results/Validation_Paper3_4Divisions_Deterministic_Separate_11"
    checkpoint_filename = "PPO_Model_Episdoe_400.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for agent_id in range(1,2):
        agent_path = os.path.join(parameter_storage_location, f"Agent_{agent_id}")
        agent_parameters_path = os.path.join(agent_path, checkpoint_filename)
        checkpoint_original = torch.load(agent_parameters_path, map_location=device,weights_only = False)
        checkpoint_copy = checkpoint_original.copy()

        env = PrecastSchedulingEnv(n_elements = n_elements, 
                 resource_1_cost = resource_1_cost, 
                 resource_2_cost = resource_2_cost, 
                 resource_3_cost = resource_3_cost, 
                 resource_1_capacity = resource_1_capacity, 
                 resource_2_capacity = resource_2_capacity, 
                 resource_3_capacity = resource_3_capacity, 
                 resource_requirements = resource_requirements, 
                 activity_times = activity_times, 
                 precedence_relations = relations,
                 action_space = action_space, 
                 weight_vector = weight_vector,
                 reward_fn = reward_fn)
        
        # Initialize the ActorCritic
        actor_critic = ActorCritic(n_elements = n_elements, 
                    node_feature_dim = node_features_dim, 
                    gcn_hidden_dim = gcn_hidden_dim,
                    gcn_output_dim = gcn_output_dim,
                    resource_availability_feature_size = resource_availability_feature_size,
                    remaining_elements_feature_size = remaining_elements_feature_size,
                    structural_support_feature_size = structural_support_feature_size, # Need to try with another graph neural nework
                    action_mask_feature_size = action_mask_feature_size, 
                    hidden_dim = hidden_dim, 
                    actor_hidden_dim = actor_hidden_dim,
                    critic_hidden_dim = critic_hidden_dim,
                    num_actions = len(action_space),
                    dropout_rate = dropout_rate,
                    device = None)
    
        # 2) Create your PPO (or other RL) agent
        agent = Agent(actor_critic_model = actor_critic, 
                    action_space = action_space, # The whole action space should be passed. Why would you pass only valid actions here.
                    lr = lr, 
                    gamma = gamma,
                    entropy_coef = entropy_coef,
                    clip_epsilon = clip_epsilon, 
                    gae_lambda = gae_lambda, # Need to see how to decide this value.
                    K_epochs = K_epochs,
                    minibatch_size = minibatch_size,
                    batch_size = minibatch_size,
                    weight_vector = weight_vector,
                    device = None) 
        
        full_state_dict = checkpoint_copy['model_state_dict']
        agent.model.load_state_dict(full_state_dict) # This is to load the parameters of the updated state dictionary to the agent.model, which is actorcritic in my case. 
        agent.eval()

        results = np.empty((0, 2))
        for episode in tqdm(range(0, 10)):
            state = env.reset() # Reset the environment
            done = False
            episode_cost = 0
            episode_time = 0
            timestep = 0

            while not done and timestep<max_timesteps:
                with torch.no_grad():  # No gradient calculation during evaluation
                    action = agent.select_action_prediction(state, timestep)
                # Take action in the environment 
                next_state, reward, done, total_duration, cost_step, unnormalized_reward = env.step(action, timestep) # This will essentially increase the step.
                episode_cost += cost_step
                episode_time = total_duration
                    # Update state
                state = next_state

                timestep = timestep + 1
        
            results = np.append(results, [[episode_cost, episode_time]], axis=0)
        
            if episode == 9:
                mean_results_10 = np.mean(results, axis = 0)
                std_results_10 = np.std(results, axis = 0)

        # Organize your data into a dictionary
        data = {
            "Episodes": [10],
            "Mean Cost": [mean_results_10[0]],
            "Std Cost": [std_results_10[0]],
            "Mean Time": [mean_results_10[1]],
            "Std Time": [std_results_10[1]],
        }

        df = pd.DataFrame(data)
        # Save to Excel
        evaluation_excel = os.path.join(agent_path, "rl_evaluation_results.xlsx")
        df.to_excel(evaluation_excel, index=False)
        tasks = np.arange(1, n_elements + 1)  # Simpler way to create [1,2,3,...,n]

        schedule_details = pd.DataFrame({
            "Activity":tasks,
            "Start times":env.task_start_times,
            "Finish times":env.task_finish_times
        })

        average_parallel_activities = count_parallel_activities(schedule_details)

        objectives_df = pd.DataFrame({
            "Schedule_cost": [mean_results_10[0]],
            "Schedule_time": [mean_results_10[1]]
        })

        return schedule_details, objectives_df, average_parallel_activities