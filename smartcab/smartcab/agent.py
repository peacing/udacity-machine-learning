import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        # q_matrix = (action, light, waypoint, oncoming, right, left)
        self.q_matrix = np.zeros([4,2,3,4,4,4], dtype=float) #initialize with zeroes
        self.actions = (None, 'left', 'right', 'forward')
        self.waypoints = ('left', 'right', 'forward')
        self.lights = ('red','green')
        self.alpha = 0.9    # learning rate
        self.gamma = 0.15    # future rewards rate
        self.epsilon = .20
        self.consecutive_nones = 0
        
        self.successes = 0
        self.penalties = 0
        self.num_trials = 0
        
     
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.epsilon += -0.05
        self.epsilon = max(0.00, self.epsilon)
        self.consecutive_nones = 0
        print "New epsilon: {}".format(self.epsilon)
        self.num_trials += 1
        print "---------TRIAL NUMBER {}------------".format((self.num_trials-1))
        
        
    def findIndex(self, input, input_list):
        for i, v in enumerate(input_list):
            if(input == v):
                return i
    
    def findIndices(self, state):
        """For each input, get it's numeric location in the Q matrix"""
        light = self.findIndex(state[0], self.lights)
        waypoint = self.findIndex(state[1], self.waypoints)
        oncoming = self.findIndex(state[2], self.actions)
        right = self.findIndex(state[3], self.actions)
        left = self.findIndex(state[4], self.actions)
        
        return (light, waypoint, oncoming, right, left)
    
    def findMaxQ(self, state):
        """Return max Q value
        """        
        
        # (1, 2, 0, 0, 0) = ('green', 'forward', None, None, None)
        m_index = self.findIndices(state)
            
        q_values = []
        # For each action, add it's corresponding q-value according to the current state
        for i, a in enumerate(self.actions):
            q_values.append(self.q_matrix[i, m_index[0], m_index[1], m_index[2], m_index[3], m_index[4]]) 
            
            
        max_q = max(q_values)
            
        return max_q
    
    def findMaxQandAction(self, state):
        """Return max Q value and associated action with some randomness
        """
        
        m_index = self.findIndices(state)
        
        rn = random.random() #random number beween 0-1
        #print "rn: {}".format(rn)
        #print "consec nones: {}".format(self.consecutive_nones)
        
        
	    # Force a random action based on epsilon, or if 3 consecutive non-Actions
        if ((rn < self.epsilon) | (self.consecutive_nones > 3)):
            self.consecutive_nones = 0 # reset the consecutive none counter
            action = random.choice(self.actions)
            action_index = self.actions.index(action)
            
            max_q = self.q_matrix[action_index, m_index[0], m_index[1], m_index[2], m_index[3], m_index[4]]
        
        else:
            q_values = []
            # For each action, add it's corresponding q-value according to the current state
            for i, a in enumerate(self.actions):
                q_values.append(self.q_matrix[i, m_index[0], m_index[1], m_index[2], m_index[3], m_index[4]])
	        
            max_q = max(q_values)
            print "q_values: {}".format(q_values)	    
            max_index = np.argmax(q_values)
            action = self.actions[max_index]
	    
        #If action is None, increment the counter. Floor at 0
        if self.num_trials < 20:
            if action is None:
                self.consecutive_nones += 1
            else:
                self.consecutive_nones += -1
                self.consecutive_nones = max(0, self.consecutive_nones)
        
        return max_q, action
	    
        
    def update(self, t):
        """Determine best action, execute, get reward, 
        count successes/penalties and update the Q matrix
        """
        # find the waypoint, inputs, and deadline
        
        # left, right or forward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        # {'light': 'green', 'oncoming': None, 'right': None, 'left': None}
        inputs = self.env.sense(self)  
        
        # e.g. 25
        deadline = self.env.get_deadline(self)
        
        #print "next waypoint: {}, inputs: {}, deadline: {}".format(self.next_waypoint, inputs, deadline)

        # TODO: Update state
        current_state = (inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['right'], inputs['left'])
        
        #print "current state: {}".format(current_state)
        
        self.state = current_state
        
        # TODO: Select action according to your policy
        old_q, action = self.findMaxQandAction(current_state)
        
        #print "old q: {}, action: {}".format(old_q, action)
        
        #possible_directions = [None, "left", "right", "forward"]
        #action = possible_directions[random.randint(0,3)]

        # Get reward
        reward = self.env.act(self, action)
        
        # Count number of successes
        if reward > 9:
            self.successes += 1
            
        # Count number of penalties
        if reward < 0:
            self.penalties += 1
            
        print "No. of successes {}".format(self.successes)
        print "No. of penalties {}".format(self.penalties)

        # TODO: Learn policy based on state, action, reward
        # Get new state and inputs
        new_inputs = self.env.sense(self)
        
        self.new_waypoint = self.planner.next_waypoint()
        
        new_state = (inputs['light'], self.new_waypoint, inputs['oncoming'], inputs['right'], inputs['left'])
        
        #print "new state: {}".format(new_state)
        #print "new_inputs: {}".format(new_inputs)
        
        new_q = self.learn_q(reward, new_state, old_q)
        
        action_index = self.findIndex(action, self.actions)
        state_indices = self.findIndices(current_state)
        
        # Update Q Matrix with new learned value
        self.q_matrix[action_index, state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4]] = new_q


        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    def learn_q(self, reward, new_state, old_q):
        """Update q value with reward received and expected future reward
        """    
        
        # calculate an updated q value
        new_q = old_q + self.alpha * (reward + self.gamma * self.findMaxQ(new_state) - old_q)
        
        return new_q
        
        
        
class PolicyAgent(Agent):
    """An agent that knows how to drive in the smartcab world."""

    def __init__(self, env, policy):
        super(PolicyAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        # q_matrix = (action, light, waypoint, oncoming, right, left)
        self.policy = policy
        self.actions = (None, 'left', 'right', 'forward')
        self.waypoints = ('left', 'right', 'forward')
        self.lights = ('red','green')
        
        # Keep track of the agent's performance
        self.successes = 0
        self.penalties = 0
        self.num_trials = 0
        
        # keep track of the efficiency of the trained agent
        self.maximum_steps = 0
        self.taken_steps = 0
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.num_trials += 1
        print "---------TRAINED TRIAL NUMBER {}------------".format((self.num_trials-1))
        
        print "taken steps/max steps: {}/{}".format(self.taken_steps, self.maximum_steps)
        
        self.maximum_steps += self.env.get_deadline(self)
        
        
    def update(self, t):
        """Determine best action, execute, get reward, and count successes/penalties
        """
        
        # Get inputs for current state
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        state = (inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['right'], inputs['left'])
        
        self.state = state
        
        old_q, action = self.findMaxQandAction(state)
        
        # Get reward
        reward = self.env.act(self, action)
        
        # Count number of steps
        self.taken_steps += 1
        
        # Count number of successes
        if reward > 9:
            self.successes += 1
            
        # Count number of penalties
        if reward < 0:
            self.penalties += 1
            
        print "No. of successes {}".format(self.successes)
        print "No. of penalties {}".format(self.penalties)
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        
        
    def findMaxQandAction(self, state):
    
        m_index = self.findIndices(state)
        
        q_values = []
            
        for i, a in enumerate(self.actions):
            q_values.append(self.policy[i, m_index[0], m_index[1], m_index[2], m_index[3], m_index[4]])
	        
        max_q = max(q_values)
        print "q_values: {}".format(q_values)	    
        max_index = np.argmax(q_values)
        action = self.actions[max_index]
	    
        return max_q, action
        
    def findIndex(self, input, input_list):
        for i, v in enumerate(input_list):
            if(input == v):
                return i
    
    def findIndices(self, state):
    
        light = self.findIndex(state[0], self.lights)
        waypoint = self.findIndex(state[1], self.waypoints)
        oncoming = self.findIndex(state[2], self.actions)
        right = self.findIndex(state[3], self.actions)
        left = self.findIndex(state[4], self.actions)
        
        return (light, waypoint, oncoming, right, left)
        
        
        

def run():
    """Run the agent for a finite number of trials."""

    # Set up learning environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    print "---------------Training Agent-----------------"
    # Now simulate it
    sim = Simulator(e, update_delay=0.00)  # reduce update_delay to speed up simulation
    sim.run(n_trials=75)  # press Esc or close pygame window to quit
    
    policy = a.q_matrix
    
    # Set up trained environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(PolicyAgent, policy)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    print "---------------Trained Agent-----------------"
    # Now simulate it
    sim = Simulator(e, update_delay=0.00)  # reduce update_delay to speed up simulation
    sim.run(n_trials=25)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
