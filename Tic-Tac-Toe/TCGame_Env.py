from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        #row check
        for i in range(3):
            if (curr_state[(i*3) + 0] + curr_state[(i*3)+1] +curr_state[(i*3)+2]) == 15:
                return True
        #column check
        for i in range(3):
            if (curr_state[(0*3) + i] + curr_state[(1*3)+i] +curr_state[(2*3)+i]) == 15:
                return True
     
        #Diagonal check
        if (curr_state[2] + curr_state[4] +curr_state[6]) == 15:
            return True
        if (curr_state[0] + curr_state[4] +curr_state[8]) == 15:
            return True
        
        return False
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        newState = self.state_transition(curr_state, curr_action)
        isTerminated, result = self.is_terminal(newState)
        
        if(isTerminated == False):
            # Play for Player 2 
            # Select random position from allowed positions
            pos = random.choice(self.allowed_positions(newState))
            # Select random value from allowed possible values for player 2(even numbers)
            val = random.choice(self.allowed_values(newState)[1])
            newState[pos]= val
            isTerminated, result = self.is_terminal(newState)
            if isTerminated == True:
                if result == 'Win':
                    reward=-10
                else:
                    reward=0
            else:
                reward=-1
        else:
            if result == 'Win':
                reward=10
            else:
                reward=0
                
        return newState,reward,isTerminated
            


    def reset(self):
        return self.state
