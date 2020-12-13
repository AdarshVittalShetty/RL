# Import routines

import numpy as np
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # m ranges from 0 to 4,(0,0) is if the Driver takes no action,
        # Hence,Total possibile actions = 21
        self.action_space = [(0, 0)] + list(permutations([i for i in range(0,m)], 2))
        self.state_space = [[x, y, z]for x in range(0,m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # location --> state[0]
        # Time     --> state[1]
        # Day      --> state[2]        
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = [0]+random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request       
        
        actions = [self.action_space[i] for i in possible_actions_index]       
        
        return possible_actions_index,actions   

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        # Initialize various times
        total_time   = 0
        transit_time = 0    # to go from current  location to pickup location
        wait_time    = 0    # in case driver chooses to refuse all requests
        ride_time    = 0    # from Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        curr_time = state[1]
        curr_day = state[2]
        """
         3 Scenarios for calculating reward: 
           a) Refuses all requests
           b) Driver is already at pick up point
           c) Driver is not at pickup point, need to travel to pick up point from cur location
        """    
        if ((pickup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 hr,
            wait_time = 1
        elif (curr_loc == pickup_loc):
            # driver is already at pickup point, wait and transit are both 0 then.
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
        else:
            # Driver is not at the pickup point, he needs to travel to pickup point first
            # time take to reach pickup point from his current location
            transit_time      = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)
            
            # The driver is now at the pickup point
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]

        # Calculate total time as sum of all durations, battery consumption time
        total_time = (wait_time + transit_time + ride_time)
       
        
        reward = (R * ride_time) - (C * total_time)   
        
        return reward,total_time

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        next_state = []
        
        # Initialize various times
        total_time   = 0
        transit_time = 0    # to go from current  location to pickup location
        wait_time    = 0    # in case driver chooses to refuse all requests
        ride_time    = 0    # from Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        curr_time = state[1]
        curr_day = state[2]
        """
         3 Scenarios for calculating next state: 
           a) Refuses all requests
           b) Driver is already at pick up point
           c) Driver is not at pickup point, need to travel to pick up point from cur location
        """    
        if ((pickup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time = 1
            next_loc = curr_loc
        elif (curr_loc == pickup_loc):
            # means driver is already at pickup point, wait and transit are both 0 then.
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            
            # next location is the drop location
            next_loc = drop_loc
        else:
            # Driver is not at the pickup point, he needs to travel to pickup point first
            # time take to reach pickup point from his current location
            transit_time      = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)
            
            # The driver is now at the pickup point
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)
        
        # Construct next_state using the next_loc and the new time states.
        next_state = [next_loc, next_time, next_day]
        
        return next_state

    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        # Get the next state 
        next_state = self.next_state_func(state, action, Time_matrix)

        # Calculate the reward based
        rewards, total_time = self.reward_func(state, action, Time_matrix)
        
        return next_state,rewards,total_time


    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    
    def update_time_day(self, time, day, duration):
        """
        Takes in current time, day, duration
        returns updated time and day, this handy if the ride prolongs to next day
        """
        duration = int(duration)
        new_time = time

        if (time + duration) < 24:
            new_time = time + duration
            # day is unchanged
        else:
            
            # Get the number of days
            num_days = (time + duration) // 24
            
            # New time on next day
            # convert the time to 0-23 range
            new_time = (time + duration) % 24 
            
            # New day Convert the day to 0-6 range
            day = (day + num_days ) % 7

        return new_time, day
