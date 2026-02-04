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
import os

class Agent: 
    def __init__(self, 
                 actor_critic_model, 
                 action_space, # The whole action space should be passed. Why would you pass only valid actions here.
                 lr, 
                 gamma,
                 entropy_coef,
                 clip_epsilon, 
                 gae_lambda, # Need to see how to decide this value.
                 K_epochs,
                 minibatch_size,
                 batch_size,
                 weight_vector,
                 device = None):
        """
        Initializing the Agent with the ActorCritic model and optimizer.
        
        actor_critic_model (nn.Module) : The ActorCritic model.
        device : Device to run the model on
        entropy_coef : Coefficient for entropy regularization"""

        # Automatically select device if not provided
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = actor_critic_model.to(self.device) # When the parent module is moved to a device, all the submodules should automatically be moved. 

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4) # Adding the weight decay here automatically adds the L2 regularization. 

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.mini_batch_size = minibatch_size
        self.clip_epsilon = clip_epsilon # This is basically the epsilon value that is used for clipping the updates

        self.action_space = action_space
        self.weight_vector = weight_vector

        # Memory to store experiences
        self.reset_memory()

    def reset_memory(self):
        """
        Resets the memory buffers.
        """
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []
        self.states = [] # Optional, storing stats
        self.actions = [] # Optional, storing actions
    
    def create_data_object(self, state):
        # node_features = torch.tensor(state['node_features'], dtype=torch.float32)  # Shape: [n_elements, feature_dim]
        node_features = state['node_features']
        edge_index = self.adjacency_to_edge_index(state['edge_index'])           # Shape: [2, num_edges]
        remaining_elements = state['remaining_elements'].unsqueeze(0)
        resource_availability = state['resource_availability'].unsqueeze(0)
        action_mask = state['action_mask'].unsqueeze(0)
        precedence_relationships = state['precedence_relationships'].flatten(start_dim=0).unsqueeze(0)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            remaining_elements=remaining_elements,
            resource_availability=resource_availability,
            action_mask=action_mask,
            precedence_relationships=precedence_relationships,
        )
        
        return data
    
    def create_batched_data(self, states):
        data_list = [self.create_data_object(state) for state in states]
        batched_data = Batch.from_data_list(data_list)
        batched_data = batched_data.to(self.device)  # Move to GPU or CPU as needed
        return batched_data

    def adjacency_to_edge_index(self, edge_adjacency_matrix):
        "To convert the adjacency matrix of the edges to the edge representation needed for the GCN."

        # You need to get the indices where adjacency_matrix is 1
        # Returns: edge_index: Edge indices [2, num_edges]
        if edge_adjacency_matrix.dim() != 2:
            raise ValueError(f"Expected a 2D adjacency matrix, but got {edge_adjacency_matrix.dim()}D.")

        edge_index = edge_adjacency_matrix.nonzero(as_tuple = False).t().contiguous()
        return edge_index

    def select_action(self, observation, timestep):
        """
        Selects an action based on the current observation. 
        
        observation (dict) : Current state observation from the environment.
        
        Returns : action (numpy.ndarray) - Selected action. """
        self.model.eval()
        # selected_actions = torch.zeros(n_elements, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            # Convert single state to Data object
            data = self.create_data_object(observation).to(self.device)

            # Batch the single Data object
            batched_data = Batch.from_data_list([data])

            action_mask = data.action_mask # Get the action mask from the data object
            
            action_probs, state_value = self.model(batched_data, timestep, is_training = True) # So this should return the action probabilities and the state values for each time step. 

            dist = Categorical(action_probs) # Creating a categorical distribution over the actions probabilities

            # Sampling an action
            action = dist.sample() # action here will be the index of the selected array. 
            if np.all(self.action_space[action.item()] == 0):
                if torch.sum(action_mask) > 1:
                    # If there are valid actions, sample again until a valid action is found
                    while True:
                        action = dist.sample()
                        if np.any(self.action_space[action.item()]!=0) :
                            break
            # Get log probability and entropy
            log_prob = dist.log_prob(action) # Gets the log_probability of the selected action. 
            entropy = dist.entropy() # gets the entropy of the distribution.

            selected_action = self.action_space[action.item()] # This should get the selected action from the action space. action.item() converts the 0D tensor to a number, so that it can be used as an index.

            # Storing in the memory

            self.log_probs.append(log_prob)
            self.entropies.append(entropy)
            self.values.append(state_value)
            self.actions.append(action) # This is optional I think. There is no use of this. 
            self.states.append(observation) 

        return selected_action

    def select_action_prediction(self, observation, timestep):
        """
        Selects an action based on the current observation. 
        
        observation (dict) : Current state observation from the environment.
        
        Returns : action (numpy.ndarray) - Selected action. """
        self.model.eval()
        # selected_actions = torch.zeros(n_elements, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            # Convert single state to Data object
            data = self.create_data_object(observation).to(self.device)

            action_mask = data.action_mask # Get the action mask from the data object

            # Batch the single Data object
            batched_data = Batch.from_data_list([data])
            
            action_probs, state_value = self.model(batched_data, timestep, is_training = True) # So this should return the action probabilities and the state values for each time step. 

            ranked = torch.argsort(action_probs, descending=True)

            chosen = None

            for a in ranked.view(-1):
                a = a.item()
                if action_probs.view(-1)[a].item()>0 and np.any(self.action_space[a]!=0):
                    chosen = a
                    break

            if chosen is None:
                chosen = ranked.view(-1)[0].item()
            
            action = torch.tensor(chosen, device=self.device)

            dist = Categorical(action_probs) # Creating a categorical distribution over the actions probabilities
            log_prob = dist.log_prob(action) # Gets the log_probability of the selected action. 
            entropy = dist.entropy() # gets the entropy of the distribution.

            selected_action = self.action_space[action.item()] # This should get the selected action from the action space. action.item() converts the 0D tensor to a number, so that it can be used as an index.

            self.log_probs.append(log_prob)
            self.entropies.append(entropy)
            self.values.append(state_value)
            self.actions.append(action) # This is optional I think. There is no use of this. 
            self.states.append(observation) 

        return selected_action 
    
    def eval(self):
        """
        Sets the model to evaluation mode.
        """
        self.model.eval()

    def store_transition(self, reward, done):
        # As the agent interacts with the environment, at each time step, it stores the rewards received, and whether that particular state was terminal state or not. 
        """
        Stores the reward and done flag for the current timestep.
        
        Args:
            reward (float): Reward received after taking the action.
            done (bool): Flag indicating if the episode has ended.
        """
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device).view(-1)
        # Append each component reward to its own list
        self.rewards.append(reward_tensor)
        # self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        self.dones.append(torch.tensor([done], dtype=torch.float32).to(self.device))

    def compute_returns_and_advantages(self, last_value, first_state_index, last_state_index, done):
        # Once a sequence of experiences is collected (often at the end of an episode or after a fixed number of steps), this function processes this data to compute the total discounted reward for each time step onward, and the difference between these returns and
        # and the Critic's value estimates, indicating the relative benefit of the actions taken. 
        # Should use generalized advantage function here. 

        """
        Computes discounted returns and advantages.

        Args:
            last_value (torch.Tensor): The value of the last state.
            done (bool): Flag indicating if the episode has ended.

        Returns:
            returns (torch.Tensor): Tensor of discounted returns.
            advantages (torch.Tensor): Tensor of advantages.
        """
        num_objectives = 2
        returns_list = []
        advantages_list = []

        #So now, you try to get the returns and advantages for each objective separately. 
        for i in range(0,num_objectives):

            # Access the reward corresponding to each objective
            rewards = [r[i] for r in self.rewards[first_state_index:last_state_index + 1]] # Rewards is an array here

            dones = self.dones[first_state_index:last_state_index + 1]

            # values = self.values[first_state_index:last_state_index] + [last_value.view(1,1)]

            values = [v[i] for v in self.values[first_state_index:last_state_index+1]]
            # last_value = last_value.view(-1)
            values.append(last_value[i].view(1, 1))

            returns = []
            advantages = []
            gae = 0

            for step in reversed(range(len(rewards))):  # Iterates over the timesteps in the reverse order, starting from the last one.
                delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step] # If done =1, this means it is the last step, then the step+1 portion gets ignored.
                gae = delta + self.gamma * self.gae_lambda * gae * (1-dones[step]) # It is basically computing the Advantage function in the reverse order
                advantages.append(gae) # Append the advantage value to the advantages array
                returns.append(gae + values[step])

            # Reverse the lists to maintain the original order
            returns = torch.cat(returns[::-1]).detach().view(-1, 1)
            advantages = torch.cat(advantages[::-1]).detach().view(-1, 1)

            returns_list.append(returns)
            advantages_list.append(advantages)
    
        return returns_list, advantages_list
    
    def update(self):
        "Updating the policy and value network based on collected experiences"
        "Use the PPO clipped surrogate objectives with minibatch updates"

        # Set the model to training mode
        self.model.train()
        # progress = episode / (total_episodes - 1)        # 0 → 1
        # self.entropy_coef = initial_entropy + progress * (final_entropy - initial_entropy)

        # Check if there are enough episodes
        if len(self.rewards) == 0:
            print("No experiences to update")
            return
        
        total_timesteps = len(self.rewards)

        # Initialize lists for first and last states
        first_states = []
        last_states = []

        # Iterate through the batch of dones
        for i in range(len(self.dones)):
            # Always add the first state
            if i == 0:
                first_states.append(i)
            
            # Add to last states if `dones[i]` is True
            if self.dones[i] == True:
                last_states.append(i)
                
                # Add the next state to first_states if it exists
                if i + 1 < len(self.dones):
                    first_states.append(i + 1)
            
            # Ensure the last state is added to `last_states`
            if i == len(self.dones) - 1 and i not in last_states:
                last_states.append(i)

        num_objectives = 2

        # Initialize returns and advantages as empty tensors
        returns_list = [torch.tensor([], device=self.device, dtype=torch.float32) for _ in range(num_objectives)]
        advantages_list = [torch.tensor([], device=self.device, dtype=torch.float32) for _ in range(num_objectives)]

        # Computing returns segmentwise
        for i in range(0, len(last_states)):
            with torch.no_grad():
                if self.dones[last_states[i]] == True:
                    # The case when we are ending at the terminal state
                    # We don't need to compute the values here. 
                    # If it is the terminal state, there is no value
                    # THe last value here is not exactly the last value. It is the value of the state after the terminal state. 
                    last_value = torch.tensor([[0.0],[0.0]], device='cuda', dtype=torch.float32)
                else:

                    last_state_data_list = self.create_data_object(self.states[last_states[i]]) # Getting the last state of the segment
                    last_state_batched_data = Batch.from_data_list([last_state_data_list]).to(self.device)
                    last_value_action_probs, last_value = self.model(last_state_batched_data, 1000, is_training = False)
                    last_value = last_value
            segmental_returns, segmental_advantages = self.compute_returns_and_advantages(last_value, first_states[i], last_states[i], done = self.dones[last_states[i]])

            # Append results to overall tensors
            for j in range(0, num_objectives):
                returns_list[j] = torch.cat([returns_list[j], segmental_returns[j]])
                advantages_list[j] = torch.cat([advantages_list[j], segmental_advantages[j]])

        # Plotting the advantages before the normalization makes more sense, I think. 

        mean_adv_time = advantages_list[0].mean().item()
        std_adv_time = advantages_list[0].std().item()
        mean_adv_cost = advantages_list[1].mean().item()
        std_adv_cost = advantages_list[1].std().item()
        # Normalize advantages separately
        for i in range(0, num_objectives):
            advantages_list[i] = (advantages_list[i] - advantages_list[i].mean()) / (advantages_list[i].std() + 1e-8)

        # Get the combined advantages
        advantages = self.weight_vector[0]*advantages_list[0] + self.weight_vector[1]*advantages_list[1]
        # Convert lists to tensors
        log_probs = torch.stack(self.log_probs).to(self.device)  # [N]
        entropies = torch.stack(self.entropies).to(self.device)  # [N]
        v_time = torch.stack([v[0] for v in self.values]).squeeze().unsqueeze(1).to(self.device)
        v_cost = torch.stack([v[1] for v in self.values]).squeeze().unsqueeze(1).to(self.device)

        values_list = [v_time, v_cost]
        actions = torch.stack(self.actions).to(self.device)  # [N]

        # Determine the number of mini-batching
        num_mini_batches = total_timesteps // self.mini_batch_size
        if num_mini_batches == 0:
            num_mini_batches = 1 # Atleast one batch

        for epoch in range(self.K_epochs): # So you run it for each epoch
            indices = torch.randperm(total_timesteps).to(self.device)
            for i in range(num_mini_batches):
                start = i*self.mini_batch_size
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                # Selecting mini_batch data
                # So this will get the data corresponding to the indices selected in the mini batches
                mini_log_probs = log_probs[batch_indices]
                mini_entropies = entropies[batch_indices]
                mini_values = [values[batch_indices] for values in values_list] # This would become a list of three tensors
                mini_returns = [returns[batch_indices] for returns in returns_list] # This would become a list of three tensors
                mini_advantages = advantages[batch_indices] # This would be a list of one tensor.
                mini_actions = actions[batch_indices]

                # Getting the data list
                minibatch = [self.states[j] for j in batch_indices]
                data_list = [self.create_data_object(state) for state in minibatch]

                # Creating the batch of the Data objects
                batched_data = Batch.from_data_list(data_list).to(self.device)

                # Forward pass through the model
                action_probs, state_values = self.model(batched_data, 1000, is_training = False) # So you are already having the states. But you want the model to just predict the probabilities and value functions again. 

                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(mini_actions.squeeze()) # So this gives the log probs for the same actions that the agent took from this state, but in the new policy. 
                entropy = dist.entropy()

                # Compute the ratio (r_t)
                ratios = torch.exp(new_log_probs.unsqueeze(1) - mini_log_probs.detach())

                # Compute the surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_advantages

                # Compute the actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                # print(actor_loss)

                # Compute the critic loss
                critic_loss_time = F.mse_loss(state_values[0].squeeze(), mini_returns[0].squeeze())
                critic_loss_cost = F.mse_loss(state_values[1].squeeze(), mini_returns[1].squeeze())
                critic_loss = self.weight_vector[0]*F.mse_loss(state_values[0].squeeze(), mini_returns[0].squeeze()) + self.weight_vector[1]*F.mse_loss(state_values[1].squeeze(), mini_returns[1].squeeze())
                # print(critic_loss)

                # Compute the entropy loss
                entropy_loss = -self.entropy_coef * entropy.mean()

                # Total loss
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Gradient clipping for stability
                self.optimizer.step()
        print(f"Actor loss at epoch {epoch} is {actor_loss}")
        print(f"Critic loss at epoch {epoch} is {critic_loss}")
        print(f"Entropy loss at epoch {epoch} is {entropy_loss}")
        print(f"Total loss at epoch {epoch} is {loss}")

        # Clear memory after updating
        self.reset_memory()
        return actor_loss, critic_loss, mean_adv_time, std_adv_time, mean_adv_cost, std_adv_cost, entropy_loss, torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon).mean().item(), critic_loss_time, critic_loss_cost
