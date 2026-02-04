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


class PrecastSchedulingEnv(gym.Env): #This class is modeling the environment in which the agent interacts
    metadata = {'render.modes' : ['human']}

    def __init__(self, 
                 n_elements, 
                 resource_1_cost, 
                 resource_2_cost, 
                 resource_3_cost, 
                 resource_1_capacity, 
                 resource_2_capacity, 
                 resource_3_capacity, 
                 resource_requirements, 
                 activity_times, 
                 precedence_relations,
                 action_space, 
                 weight_vector,
                 reward_fn):
        super(PrecastSchedulingEnv, self).__init__() # In Python, super() is used to give you access to methods from a parent (or superclass) class. In the context of your PrecastSchedulingEnv class, which is inheriting from gym.Env, the super(PrecastSchedulingEnv, self).__init__() call is invoking the __init__ method of the parent class, gym.Env.

        self.n_elements = n_elements #Number of elements to be scheduled
        self.daily_resource_1_cost = resource_1_cost 
        self.daily_resource_2_cost = resource_2_cost 
        self.daily_resource_3_cost = resource_3_cost 
        self.reward_function = reward_fn

        self.resource_1_capacity = resource_1_capacity 
        self.resource_2_capacity = resource_2_capacity 
        self.resource_3_capacity = resource_3_capacity
        # self.resource_capacity = resource_capacity #This will be a matrix representing different types of resources that need to be used in the project
        self.resource_1_available = resource_1_capacity # Initialize them with the maximum availability
        self.resource_2_available = resource_2_capacity # Initialize them with the maximum availability, which will then be updated with the number of steps, I think. 
        self.resource_3_available = resource_3_capacity # Initialize them with the maximum availability, which will then be updated with the number of steps, I think. 

        self.resource_requirements = resource_requirements

        self.precendence_relations = precedence_relations # To check the structural support relations of the modules

        self.total_time_steps = 0 # Total timesteps
        self.current_day = 0  # Current day (increments every 8 hours)

        # Initialize the newly completed elements tracker (starts empty)
        self.newly_completed_elements_indices = torch.zeros(self.n_elements, dtype=torch.bool)  # All elements start as incomplete
        self.newly_completed_elements = 0 # Tracking the count of newly_completed_elements

        self.duration_params = activity_times
        # Duration distribution has to be in certain form, for example:

        self.min_duration = self.get_min_duration(activity_times)
        self.max_duration = self.get_max_duration(activity_times)

        # Initialize node features and edge indices for the GCN
        self.node_features = torch.zeros((self.n_elements, 5))  # [status, resource_1_needed, resource_2_needed, resource_3_needed, duration]
        self.edge_index = torch.empty((2, 0), dtype=torch.long) # Edge indices for the graph (empty at initialization)

        self.action_space = torch.from_numpy(np.array(action_space)) # Generating the action space here. 

        # So I will maintain three arrays here.
        self.in_progress_elements = np.zeros(n_elements) 
        self.completed_elements = np.zeros(n_elements)
        self.not_yet_started_elements = np.ones(n_elements)

        self.action_record = []
        self.started_elements = []
        self.finished_elements = []

        self.eps = 1e-8
        self.window_size = 500
        self.rewards_buffer = deque(maxlen=self.window_size)
        self.weight_vector = weight_vector

        self.task_start_times = np.zeros(n_elements)
        self.task_finish_times = np.zeros(n_elements)

        # self.reward_window = np.zeros(5,8)

        self.observation_space = spaces.Dict({ # The objective of this is to define the format, type, and bounds of the observations that the environment will return to the agent.
            'node_features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_elements, 5), dtype=np.float32), # So the node_features will be a 6 dimensional vector, where each value can be -infinity to +infitnity. 
            'edge_index': spaces.Box(low=0, high=1, shape=(self.n_elements, self.n_elements), dtype=np.float32),  # Variable size is not possible. Therefore, I define an adjacency matrix here, and we will use that adjacency matrix to get the edge_index in the GCN implementation. 
            'remaining_elements': spaces.MultiBinary(self.n_elements), # It will basically be the binary representation of n_elements, where 0 is placed at the places where elements are not yet started.
            'resource_availability': spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32), # The number of available resources for the three types. 
            'action_mask': spaces.MultiBinary(len(self.action_space)),  # Include action mask in the observation. This will again be a multi-binary value. 
            'precedence_relationships': spaces.Box(low=0, high=1, shape=(self.n_elements, self.n_elements), dtype=np.float32), # Making the structure support relationships a part of the state itself. 
        })

        self.reset()

    def get_max_duration(self, activity_times):
        max_value = float('-inf')
        for values in activity_times.values():
            max_value = max(values, max_value)

        return max_value
    
    def get_min_duration(self, activity_times):
        min_value = float('inf')
        for values in activity_times.values():
            min_value = min(values, min_value)

        return min_value

    def getActionSpace(self, n_elements):
        """
        Generate all valid actions where the nmber of selected elements is less than crane_capacity
        
        Returns : list of binary arrays representing valid actions"""
        valid_actions = []
        for i in range(0, n_elements):
            action_array = np.zeros(n_elements, dtype = int)
            action_array[i] = 1
            action_array_tensor = torch.tensor(action_array)
            valid_actions.append(action_array_tensor)
        valid_actions.append(torch.tensor(np.zeros(n_elements, dtype = int)))

        return valid_actions # So you are returning the valid actions from here, that is the reduced combinatorial action space. 

    def reset(self):
        # Reset the environment to the initial state
        # This will be used at the starting of each new episode.
        self.total_time_steps = 0
        self.current_day = 0 
        self.total_episode_cost = 0

        self.task_start_times = np.zeros(self.n_elements)
        self.task_finish_times = np.zeros(self.n_elements)

        # Initialize available resources at the start of the simulation
        self.resource_1_available = self.resource_1_capacity
        self.resource_2_available = self.resource_2_capacity
        self.resource_3_available = self.resource_3_capacity

        self.newly_completed_elements_indices = torch.zeros(self.n_elements, dtype=torch.bool)  # All elements start as incomplete - I don't think we need it to be a tensor. But keep it as a tensor for now.
        self.newly_completed_elements = 0 # Tracking the count of newly_completed_elements

        self.node_features = torch.zeros((self.n_elements, 5), dtype=torch.float32)  

        # Initialize edge_index as an empty list (graph starts with no edges). I think this should be kept as an empty list because it will be updated as the agent will continue to interact with the environment and schedule is created. 
        self.edge_index = torch.empty((2,0), dtype=torch.long) # Here, you are creating an empty tensor to store the edges.
        self.edge_adjacency_matrix = torch.zeros((self.n_elements, self.n_elements), dtype=torch.float32) # So this is to store the adjacency matrix of the nodes, basically another way of showing the edge matrix in the state. 
        self.edge_adjacency_matrix = torch.from_numpy(self.precendence_relations).to('cuda' if torch.cuda.is_available() else 'cpu')

        # So I will maintain three arrays here.
        self.in_progress_elements = np.zeros(self.n_elements) 
        self.completed_elements = np.zeros(self.n_elements)
        self.not_yet_started_elements = np.ones(self.n_elements)

        self.action_record = []
        self.started_elements = []
        self.finished_elements = []

        for i in range(self.n_elements):
            resource_vector = self.resource_requirements[i] #It gets the resource requirement for the element from the resource_requirement dictionary in the environment. This has to be an input to the environment. 
            
            sampled_duration = self.duration_params[i]

            self.node_features[i] = torch.tensor([0, resource_vector[0],  resource_vector[1], resource_vector[2], sampled_duration], dtype = torch.float32) # Here you are creating the node_features based on the data created above, i.e., the status[kept as zero for all elements to intiailize with], element_type, the the resource requirement, their actual duration also has to come here because you are taking a probailistic activity duration. 
        
        initial_state = self.get_state() # The get_state function should essentially return the observation space. For the reset function, it should basically provide the initial state based on the resetted information in this function. 
        return initial_state
    
    def get_state(self):

        # You create a copy of the node_features and normalize that copy so that the original node features is not changed after normalization. 
        node_features = self.node_features.clone().detach().cpu()

        # Normalize the node features
        r1 = node_features[:, 1] # Getting the values of resource1
        r2 = node_features[:, 2] # Getting the values of resource2
        r3 = node_features[:, 3] # Getting the values of resource3

        min_r1 = r1.min()
        max_r1 = r1.max()

        min_r2 = r2.min()
        max_r2 = r2.max()

        min_r3 = r3.min()
        max_r3 = r3.max()

        if (min_r1 != max_r1):
            normalized_r1 = (r1 - min_r1)/(max_r1 - min_r1)

        elif (min_r1 == max_r1):
            normalized_r1 = r1/max_r1

        if (min_r2 != max_r2):
            normalized_r2 = (r2 - min_r2)/(max_r2 - min_r2)

        elif (min_r2 == max_r2):
            normalized_r2 = r2/max_r2

        if (min_r3 != max_r3):
            normalized_r3 = (r3 - min_r3)/(max_r3 - min_r3)

        elif (min_r3 == max_r3):
            normalized_r3 = r3/max_r3

        node_features[:, 1] = normalized_r1
        node_features[:, 2] = normalized_r2
        node_features[:, 3] = normalized_r3

        node_features[:, 4] = (node_features[:, 4]-self.min_duration)/(self.max_duration - self.min_duration)
        
        edge_adjacency_marix = self.edge_adjacency_matrix.clone().detach().cpu()
        
        # Elements remaining to be assembled: Elements with status < 1 are considered "remaining"
        remaining_elements = (self.node_features[:, 0] == 0).float().cpu() # Binary mask: 1 if remaining, else 0
        
        resource_availability = torch.tensor([self.resource_1_available/self.resource_1_capacity, self.resource_2_available/self.resource_2_capacity, self.resource_3_available/self.resource_3_capacity], dtype=torch.float32).cpu()

        precedence_relationships = torch.tensor(self.precendence_relations, dtype = torch.float32).clone().detach().cpu() # So it was an array before. No I converted it into tensor. 

        valid_actions = torch.tensor(self.get_valid_action(), dtype = torch.float32).clone().detach().cpu() # So again, this has to be changed for an aray of arrays.  
        
        # Concatenate everything into the state representation
        state = {
            "node_features": node_features,
            "edge_index": edge_adjacency_marix, # Passing adjacency_matrix here because of the format created in the __init__ function. 
            "remaining_elements": remaining_elements,
            "resource_availability": resource_availability,
            "action_mask": valid_actions, # Valid actions should essentially be the part of the state itself - It can be a simple mask, which you can use in defining the initial observation space may be. 
            'precedence_relationships': precedence_relationships
        }
    
        return state
    
    def update_status_based_on_step(self, action):
        """
        The function to identify which element is going to finish next
        """
        if np.any(self.in_progress_elements!=0):

            # Get the remaining times of the in progress activities
            remaining_times = []
            completing_indices = []

            # Find the indices of elements in progress
            in_progress_indices = np.where(self.in_progress_elements == 1)[0] # This gives just the indices of the elements which are in progress. 
            
            remaining_times = self.node_features[in_progress_indices,4] # You also get the remaining time here itself

            min_remaining_time = torch.min(remaining_times) # This will provide the activity which has the shortest duration based on the current weather factor.

            for j, k in enumerate(in_progress_indices):
                if remaining_times[j] == min_remaining_time:
                    completing_indices.append(in_progress_indices[j])

            self.node_features[completing_indices,0] = 1 # Set the status to completed
            self.completed_elements[completing_indices] = 1
            self.in_progress_elements[completing_indices] = 0 # Remove from in progress
            self.resource_1_available += self.node_features[completing_indices,1].sum() # Release resource 1
            self.resource_2_available += self.node_features[completing_indices,2].sum() # Release resource 2
            self.resource_3_available += self.node_features[completing_indices,3].sum() # Release resource 3

            self.newly_completed_elements_indices[completing_indices] = 1
            self.newly_completed_elements = len(completing_indices)

            self.current_day = self.current_day + min_remaining_time
            self.task_finish_times[completing_indices] = self.current_day # Assigning the finish times of the activities in the array

            self.resource_1_assigned = self.resource_1_assigned + self.node_features[completing_indices, 1].sum()
            self.resource_2_assigned = self.resource_2_assigned + self.node_features[completing_indices, 2].sum()
            self.resource_3_assigned = self.resource_3_assigned + self.node_features[completing_indices, 3].sum()

    def step(self, action, timestep):
        """
        Processes the agent's action, updates the environment's state, and returns the next observation, reward, and done flag.

        Args:
            action (array or tensor): The action taken by the agent.

        Returns:
            next_state (dict): The next observation.
            reward (float): The reward obtained from the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.
        """

        prev_time = self.current_day

        self.resource_1_assigned = 0
        self.resource_2_assigned = 0
        self.resource_3_assigned = 0

        if isinstance(action, torch.Tensor):
            action = action.numpy() # Convert action to numpy array

        # This makes less sense for now to reinitialize it every time. 
        self.newly_completed_elements = 0  
        self.newly_completed_elements_indices = torch.zeros(self.n_elements, dtype=torch.bool) 

        # Updating the graph edges
        if self.newly_completed_elements != 0:
            completed_element_indices = self.newly_completed_elements_indices.numpy().copy() # Creating a numpy array here because the tensor might not behave accurately with the enumerate function. 
            completed_element = torch.where(completed_element_indices == True)
            started_element = np.where(action==1)[0]

            for i in completed_element:
                for j in started_element:
                    self.update_edges(i, j)

        # Start the element
        started_element = np.where(action==1)[0]
        self.task_start_times[started_element] = self.current_day # Assigning the start time to the corresponding activity. 
        self.node_features[started_element, 0] = 0.5
        self.resource_1_available = max(self.resource_1_available - torch.sum(self.node_features[started_element, 1]).item(), 0) # Reduce the resource_1 availability. Limiting it to zero minimum, it can not be negative. 
        self.resource_2_available = max(self.resource_2_available - torch.sum(self.node_features[started_element, 2]).item(), 0) # Reduce the resource_2 availability. Limiting it to zero minimum, it can not be negative.
        self.resource_3_available = max(self.resource_3_available - torch.sum(self.node_features[started_element, 3]).item(), 0) # Reduce the resource_3 availability. Limiting it to zero minimum, it can not be negative.

        self.in_progress_elements[started_element] = 1
        self.not_yet_started_elements[started_element] = 0
        self.started_elements.append(k for k in started_element)

        self.update_status_based_on_step(action) # Finish the action.

        deltat = self.current_day-prev_time

        started_elements_num = np.sum(action==1)
        resource_1_assigned = self.resource_1_assigned
        resource_2_assigned = self.resource_2_assigned
        resource_3_assigned = self.resource_3_assigned

        daily_resource_1_cost = self.daily_resource_1_cost
        daily_resource_2_cost = self.daily_resource_2_cost
        daily_resource_3_cost = self.daily_resource_3_cost

        present_day = self.current_day

        newly_completed_elements_indices = self.newly_completed_elements_indices
        newly_completed_indices_numbers = torch.sum(newly_completed_elements_indices).item()

        all_completion_status = False
        if torch.all(self.node_features[:, 0] == 1):
            all_completion_status = True

        n_elements = len(action)

        weight_vector = self.weight_vector

        incremental_cost = resource_1_assigned*deltat*(daily_resource_1_cost) + resource_2_assigned*deltat*(daily_resource_2_cost) + resource_3_assigned*deltat*(daily_resource_3_cost)
        self.total_episode_cost += incremental_cost

        reward, cost, unnormalized_reward = self.reward_function(incremental_cost,
                                                                  self.total_episode_cost,
                                                                  timestep,
                                                                  action,
                                                                  started_elements_num,
                                                                  deltat,
                                                                  present_day,
                                                                  newly_completed_indices_numbers,
                                                                  all_completion_status,
                                                                  weight_vector,
                                                                  n_elements
                                                                  )
        
        reward = self.reward_normalizer(reward)

        done = self.check_termination()
        
        return self.get_state(), reward, done, self.current_day, cost, unnormalized_reward
    
    def reward_normalizer(self, reward):

        self.rewards_buffer.append(reward) # Add the reward to the reward buffer

        if len(self.rewards_buffer) < 2:
            # If the buffer is too small, return the raw reward
            return reward

        rewards_array = np.array(self.rewards_buffer)

        mean = np.mean(rewards_array, axis=0)
        variance = np.var(rewards_array, axis=0)
        std = np.sqrt(variance)

        std[std == 0] = 1.0

        normalized = (reward - mean) / (std + self.eps)
    
        return normalized

    def update_edges(self, source, target):
        self.edge_adjacency_matrix[source][target] = 1
    
    def get_valid_action(self):
        all_possible_actions = self.action_space # This gives all the possible actions at a particular step.

        valid_action_mask = np.ones(len(all_possible_actions), dtype = int)
        
        invalid_actions = set()

        alread_started_elements = self.node_features[:, 0] > 0 #You should get a single boolean vector here. 

        for i, action in enumerate(all_possible_actions):
            if torch.any(alread_started_elements[action == 1] == True): # The means that action has been already started
                invalid_actions.add(i)
                continue

            # Check the resource availability here itself
            total_resource_1_needed = torch.sum(self.node_features[:, 1]*action)
            total_resource_2_needed = torch.sum(self.node_features[:, 2]*action)
            total_resource_3_needed = torch.sum(self.node_features[:, 3]*action)
            if self.resource_1_available < total_resource_1_needed or self.resource_2_available < total_resource_2_needed or self.resource_3_available < total_resource_3_needed:
                invalid_actions.add(i)
                continue

            # Check the structural support relationships
            for j in range(0,len(action)):
                supports_mask = self.precendence_relations[j, :].astype(bool)  # True where k supports j
                if action[j] == 1 and torch.any(self.node_features[supports_mask, 0] != 1):
                    invalid_actions.add(i)
                    break

        # Returning the indices of the valid actions only.                            
        for j in invalid_actions:
            valid_action_mask[j] = 0 

        return valid_action_mask 
 
    def render(self, mode='human'):
        # Function - Prints the current day and time, Displays available resources, Shows current weather conditions, Lists each element's status and associated attributes
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment. Defaults to 'human'.
        """
        print("\n===== Precast Scheduling Environment =====")
        print(f"Current Day: {self.current_day}  days")
        print("\nElement Status:")
        print("Index | Status       | Type | Crane Needed | Labor Needed | Actual Dur | Remaining Dur")
        status_dict = {0.0: 'Not Started', 0.5: 'In Progress', 1.0: 'Completed'}
        for i in range(self.n_elements):
            status_value = self.node_features[i, 0].item()
            actual_dur = self.node_features[i, 4].item()
            print(f"{i:5d} | {actual_dur:10.2f}")
        print("==========================================\n")

    def close(self): # Sort of useless function. 
        # But in general, its use is to perform any necessary cleanup when the environment is no longer needed.
        """
        Performs any necessary cleanup.
        """
        pass  # No action needed if there are no external resources

    def seed(self, seed=None): # By setting the seed, you ensure that the environment's behavior is deterministic across runs with the same seed.
        """
        Sets the seed for the environment's random number generators.

        Args:
            seed (int, optional): The seed value to use. If None, a random seed is chosen.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def check_termination(self):
    # Check if all elements have been assembled (i.e., their status is 1 in node_features)
        #return torch.all(self.node_features[:, 0] == 1).item()
        if np.all(self.completed_elements) == 1:
            return True
        else:
            return False
 