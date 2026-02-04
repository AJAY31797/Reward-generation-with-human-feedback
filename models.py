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

class GCN(nn.Module): # SO I am going to use the GCN to represent the activity network till the timestep. 

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 2, dropout = 0.2):
        # in_challens - the number of input channels - basically the length of node feature vector
        # hidden_channels - number of hidden units per GCN layer.
        # out_channels - the number of output features per node. 
        # dropout - Dropout rate - Used for regularization to prevent overfitting. e.g., if it is 0.5, it means 50% of the neurons are randomly deactivated during training. 
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList() # A list to hold all GCN layers. 
        self.convs.append(GCNConv(in_channels, hidden_channels)) # First layer
        
        for _ in range(num_layers - 2): # This is only for the hidden layers.
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels)) # Outermost layer.
        self.dropout = dropout # Stores the droput rate for use during the forward pass.

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.LongTensor): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, out_channels].
        """
        
        for i, conv in enumerate(self.convs): # So it goes through all the layers
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.leaky_relu(x) # Using tanh here instead of ReLU.
                # Probably need to setup different activation layers and see the performance. 
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class ActorNet(nn.Module):
    # Separate Actor net so that the parameters can be updated later on easily for the Actor network. 
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        # e.g. an MLP for actor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        # Actor forward pass
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        logits = self.fc3(x)
        return logits # Returns the action logits

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # e.g. an MLP for critic
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Predict separate value functions for each reward component
        self.time_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Critic forward pass
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        values = [self.time_head(x), self.cost_head(x)]
        return values

class ActorCritic(nn.Module):
    def __init__(self, 
                 n_elements, 
                 node_feature_dim, 
                 gcn_hidden_dim,
                 gcn_output_dim,
                 resource_availability_feature_size,
                 remaining_elements_feature_size,
                 structural_support_feature_size,
                 action_mask_feature_size,
                 hidden_dim, 
                 actor_hidden_dim,
                 critic_hidden_dim,
                 num_actions,
                 dropout_rate,  
                 device = None): # Not giving any default value for now. I'll update it later on. 
        super(ActorCritic, self).__init__() # This basically calls the __init()__ method of the superclass.
        self.n_elements = n_elements

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate TGN
        self.gcn = GCN(in_channels = node_feature_dim, 
                       hidden_channels = gcn_hidden_dim, 
                       out_channels = gcn_output_dim, 
                       num_layers = 2, 
                       dropout = dropout_rate).to(device)
        
        # Other features
        self.additional_feature_size = resource_availability_feature_size + remaining_elements_feature_size


        self.resource_embedding = nn.Sequential(
            nn.Linear(3, resource_availability_feature_size),  # Assuming 3 resources
            nn.LeakyReLU()
        )

        self.remaining_embedding = nn.Sequential(
            nn.Linear(n_elements, remaining_elements_feature_size),
            nn.LeakyReLU()
        )

        self.structural_support_embedding = nn.Sequential(
            nn.Linear(n_elements * n_elements, structural_support_feature_size),
            nn.LeakyReLU()
        )

        self.action_mask_embedding = nn.Sequential(
            nn.Linear(num_actions, action_mask_feature_size), # Action space is of length one more than the n_elements
            nn.LeakyReLU(),
        )

        combined_embedding = gcn_output_dim  + resource_availability_feature_size + remaining_elements_feature_size + structural_support_feature_size + action_mask_feature_size

        self.shared_head = nn.Sequential(
            nn.Linear(combined_embedding, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.actor_head = ActorNet(input_dim = hidden_dim,
                                   hidden_dim = actor_hidden_dim,
                                   num_actions = num_actions)
        
        self.critic_head = CriticNet(input_dim = hidden_dim,
                                     hidden_dim = critic_hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Let's try with LeakyReLU here. 
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a = 0.01) # Trying with LeakyReLu here for the activation. 
                nn.init.constant_(m.bias, 0)

    def forward(self, data, timestep, is_training):
        """
        Forward pass of the actor critic network.
        Args: observation (dict): Observation from the environment.
        
        Returns: action_probs: Probabilities of each action.
        state_value : Estimated value of the current state.clear
        
        """
        # Extract node features and edge_index
        node_features = data.x                  # [total_nodes_in_batch, node_feature_dim]
        edge_index = data.edge_index            # [2, total_edges_in_batch]
        batch = data.batch                      # [total_nodes_in_batch]

        node_embedding = self.gcn(node_features, edge_index) # This outputs the node level embedding of size [n_elements, output_dimension]
        
        graph_embedding = global_mean_pool(node_embedding, batch)  # [gcn_output_dim] - Global mean pooling to get the graph level embedding -  Do you really need this?

        # Separate Embeddings
        resource_embedded = self.resource_embedding(data.resource_availability)  # [batch_size, resource_availability_feature_size]
        remaining_embedded = self.remaining_embedding(data.remaining_elements)  # [batch_size, remaining_elements_feature_size]
        precedence_embedded = self.structural_support_embedding(data.precedence_relationships)# [batch_size, 400]
        action_mask = data.action_mask
        action_mask_embedded = self.action_mask_embedding(data.action_mask)

        # Combine all features
        combined_embedding = torch.cat([
            graph_embedding, #GCN embedding 
            resource_embedded, # resource_availability_embedding
            remaining_embedded, # remaining_element_feature_size
            precedence_embedded,
            action_mask_embedded # precedence_feature_size
        ], dim=1) # We want to add them along the dimension 0.

        shared_features = self.shared_head(combined_embedding) # Get the shared features
        action_logits = self.actor_head(shared_features) # Get the actor features
        large_negative = -1e12
        masked_logits = action_logits + (1-action_mask)*large_negative
        action_probs = F.softmax(masked_logits, dim=-1) # Predicting the probabilities of the actions

        state_value = self.critic_head(shared_features)

        return action_probs, state_value
    
    def adjacency_to_edge_index(self, edge_adjacency_matrix):
        "To convert the adjacency matrix of the edges to the edge representation needed for the GCN."

        # You need to get the indices where adjacency_matrix is 1
        # Returns: edge_index: Edge indices [2, num_edges]
        if edge_adjacency_matrix.dim() != 2:
            raise ValueError(f"Expected a 2D adjacency matrix, but got {edge_adjacency_matrix.dim()}D.")

        edge_index = edge_adjacency_matrix.nonzero(as_tuple = False).t().contiguous()
        return edge_index
