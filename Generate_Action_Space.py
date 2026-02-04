import numpy as np
from itertools import product
from itertools import combinations

n_elements = 20
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

resource_1_capacity = 8 
resource_2_capacity = 8
resource_3_capacity = 8

def getActionSpaceLength(n_elements, resource_1_capacity, resource_2_capacity, resource_3_capacity, resource_requirements):
    """
    Generate the action space
    
    Returns : list of binary arrays representing valid actions"""

    action_space = [np.array(action, dtype=int) 
            for action in product([0, 1], repeat=n_elements)]
    
    invalid_actions = []
    
    for i, action in enumerate(action_space):
        total_resource_1_needed = np.sum([resource_requirements[i][0] for i in range(n_elements) if action[i] == 1])
        total_resource_2_needed = np.sum([resource_requirements[i][1] for i in range(n_elements) if action[i] == 1])
        total_resource_3_needed = np.sum([resource_requirements[i][2] for i in range(n_elements) if action[i] == 1])
        if total_resource_1_needed > resource_1_capacity or total_resource_2_needed > resource_2_capacity or total_resource_3_needed > resource_3_capacity:
            invalid_actions.append(i)

    action_space = [action for i, action in enumerate(action_space) if i not in invalid_actions]

    return action_space

action_space = getActionSpaceLength(n_elements, resource_1_capacity, resource_2_capacity, resource_3_capacity, resource_requirements) # This will get the number of valid actions

np.save("Add your path to save the action space location", action_space)
