import pygame
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
%matplotlib inline
from IPython.display import Video
from IPython.display import display
from screeninfo import get_monitors
from typing import Optional
from render_util import render
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class BasketballEnv(Env):
    # Modes for rendering:
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 4,
    }
    
    def __init__(self, min_score_prob=0.0, max_score_prob=0.9, line_position = 3, field_length = 10, render_mode=None):
        """ Initializes the environment and defines dynamics.
        
        Please DON'T change the names of the variables that are already defined in this method.
        """
        self.render_mode = render_mode
        self.render_time = 1 # one image per second
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization. "
            )
        if self.render_mode != "text" and self.render_mode != None:
            self.render_width = 130
            self.window_size = ((field_length+1)*self.render_width, self.render_width)
            self.cell_size = (self.render_width, self.render_width)
            self.window_surface = None
        self.state = 0 # starting state
        self.laststate = None
        self.field_length = field_length # max length of the field
        self.line_position = line_position # position of the three point line
        self.action_space = spaces.Discrete(2) # 0 = move, 1 = throw
        self.observation_space = spaces.Discrete(self.field_length+2)
        self.P = {}
        for state in range(self.field_length+2): #n+1까지
            self.P[state]={} # self.P = dict

            # action = 0(move) or 1(throw)
            for action in range(2):
                self.P[state][action] = []
                if action == 0: # 에이전트가 move
                    # 에이전트가 field length전에 있다면,
                    if state <= self.field_length-2:
                        # 튜플 형식으로 저장 (그 행동을 할 확률 100%, 다음상태, 보상, 마침유무)
                        self.P[state][action].append((1.0, state+1, 0, False))
                    if state == self.field_length - 1:
                        # 그 행동을 할 확률 100%, 다음 상태 n+1, 보상, 종료유무
                        self.P[state][action].append((1.0, state+2, 0, True))
                else: # 만약에 공을 던짐
                    # 일단 선형적으로 증가하는 분포를 만들자
                    success_prob = min_score_prob + (max_score_prob - min_score_prob) * (state/self.field_length)
                    if self.line_position <= state: # 선이 뒤에 있음, behind the line
                        # 공 던질때, 성공시 보상 = 2
                        # 튜플 형식으로 저장 (성공 확률, 다음 상태 n, 보상, 마침 유무)
                        self.P[state][action].append((success_prob, self.field_length, 2, True))
                        # 공 던질때 실패시 보상 = 0
                        self.P[state][action].append((1-success_prob, self.field_length+1, 0, True))
                    else: # 선이 앞에 있음, in front of line
                        # 공 던질때 보상 =1, 실패시 보상 =0
                        self.P[state][action].append((success_prob, self.field_length, 3, True))
                        self.P[state][action].append((1-success_prob, self.field_length, 0, True))
                    

# Test environment for performing sanity checks
gym.logger.set_level(40)
field_length = 5
line_position = 3
test_env = BasketballEnv(field_length = field_length, line_position = line_position)
# Is the successor state correct?
s = 0
a = 0
assert test_env.P[s][a][0][1] == s+1

# Do we reach the right states if we try to score?
s = 3
a = 1
assert test_env.P[s][a][0][1] == 5 or test_env.P[s][a][0][1] == 6
# Are the probabilities correct?
s = 0
a = 1
assert test_env.P[s][a][0][0] == 0

def reset(self):
    """ Reset the environment.
    
    Args: 
        None
    Returns: 
        Initial state
    """
    self.state = 0
    self.laststate = None
    
    # Return the initial state
    return self.state
        
def step(self, action):
    """ Take a step in the environment.
    
    Args:
        action
    Returns:
        next state, reward, termination
    """
    # YOUR CODE HERE
    # 현재 상태를 저장
    self.laststate = self.state
    
    # Retrieve the transitions for the current state and action
    transitions = self.P[self.state][action]
    
    # Unpack the transition probabilities, next states, rewards, and done flags
    probabilities, next_states, rewards, dones = zip(*transitions)
    
    # Select a new state based on the transition probabilities
    new_state = np.random.choice(next_states, p=probabilities)
    
    # Retrieve the reward and done flag for the selected next state
    reward = rewards[next_states.index(new_state)]
    done = dones[next_states.index(new_state)]
    
    # Update the current state
    self.state = new_state
    
    return new_state, reward, done
    
# Adding the methods to the class of the environment
setattr(BasketballEnv, 'reset', reset)
setattr(BasketballEnv, 'step', step)
setattr(BasketballEnv, 'render', render)

# checkpoint
field_length = 10
line_position = 5
test_env = BasketballEnv(field_length = field_length, line_position = line_position)
# Does the reset work?
test_env.state = 8
new_state = test_env.reset()
assert test_env.state == 0
# Does the step work?
test_env.state = 5
test_env.step(0) # moving forward
assert test_env.state == 6

# Debug: testing the environment
def evaluate(env, policy, file, num_runs=5):
    """ Evaluates the environment based on a policy.

    Please use this method to debug your code for the environment.

    Args:
        env: Environment we want to use. 
        policy: Numpy array of shape (num_states, num_actions), for each state the array contains
            the probabilities of entering the successor state based on the associated action. 
        file: File used for storing the video.
        num_runs: Number of runs displayed.
    """
    
    frames = []  # collect rgb_image of agent env interaction
    video_created = False
    for _ in range(num_runs):
        done = False
        obs = env.reset()
        while not done:
            action =  np.random.choice(np.flatnonzero(np.isclose(policy[obs], max(policy[obs]), rtol=0.0001)))
            out = env.render()
            frames.append(out)
            obs, reward, done = env.step(action)
            if done:
                out = env.render()
                frames.append(out)
                
    # create animation out of saved frames
    if all(frame is not None for frame in frames):
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        img = plt.imshow(frames[0][0])
        def animate(index):
            img.set_data(frames[index][0])
            return [img]
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
        plt.close()
        anim.save(file, writer="ffmpeg", fps=2)
        video_created = True

# For debug consider varying the parameters and changing the policy. Restarting the cell leads to video output
env = BasketballEnv(min_score_prob = 0.0, max_score_prob = 0.8, line_position = 2, field_length = 10, render_mode = "rgb_array")
policy = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]]) # probabilies for actions per state
video_file_1 = "basketball.mp4"
evaluate(env, policy, video_file_1)
Video(video_file_1, html_attributes="loop autoplay")

# MDP -> policy iteration algorithm
class Agent():
    
    def __init__(self, env, gamma=0.9, update_threshold=1e-6):
        """ Initializes the Agent.
        
        The agent takes properties of the environment and stores them for training.

        Args:
            env: Environment used for training.
            gamma: Discount factor.
            update_threshold: Stopping distance for updates of the value function.
        """
        
        self.mdp = (env.unwrapped.P, env.observation_space.n, env.action_space.n)
        self.update_threshold = update_threshold # stopping distance as criteria for stopping policy evaluation
        self.state_value_fn = np.zeros(self.mdp[1]) # a table leading from state to value expectations
        # Create random policy
        self.policy = []
        for state in range(self.mdp[1]):
            random_entry = np.random.randint(0, 1)
            self.policy.append([0 for _ in range(self.mdp[2])])
            self.policy[state][random_entry] = 1
        self.gamma = gamma # discount factor
        self.iteration = 0
        
    def reset(self):
        """ Resets the agent. """
        self.state_value_fn = np.zeros(self.mdp[1])
        self.policy = []
        for state in range(self.mdp[1]):
            random_entry = np.random.randint(0, 1)
            self.policy.append([0 for _ in range(self.mdp[2])])
            self.policy[state][random_entry] = 1
        self.iteration = 0

    def get_greedy_action(self, state):
        """ Choose an action based on the policy. """
        action = np.random.choice(np.flatnonzero(np.isclose(self.policy[state], max(self.policy[state]), rtol=0.01)))
        return action
    
    def visualize(self):
        """ Visualize the Q-function. """
        x_axis = 1
        y_axis = self.mdp[1]-2 
        vmin = min(self.state_value_fn)
        vmax = max(self.state_value_fn)
        X1 = np.reshape(self.state_value_fn[:-2], (x_axis, y_axis))
        fig, ax = plt.subplots(1, 1)
        cmap = plt.colormaps["Blues_r"]
        cmap.set_under("black")
        img = ax.imshow(X1, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis('off')
        ax.set_title("Values of the state value function on the field")
        for i in range(x_axis):
            for j in range(y_axis):
                ax.text(j, i, str(X1[i][j])[:4], fontsize=12, color='black', ha='center', va='center')
        plt.show()
        
    def render_policy(self):
        """ Print the current policy. """
        print('Policy of the agent:')
        out = ' | '
        render = out
        for i in range(self.mdp[1]-2):
            token = ""
            if self.policy[i][0] > 0:   # move
                token += "Move"
            if self.policy[i][1] > 0:   # up
                token += "Throw"
            if len(token) > 5:
                token = 'Move or Throw'
            render += token + out
        print(render) 

    # Below, the code seems to be flawed. We transferred the code into a comment, so you can copy it into your answer below
    
    
    def train(self):
        policy_stable = False
        total_sweeps = 0
        for i in range(100):
            # Policy Evaluation
            total_sweeps += self.policy_evaluation()
            # Policy Improvement
            policy_stable = self.policy_improvement()
            self.iteration += 1
        print('Sweeps required for convergence ', str(total_sweeps))
        print('Iterations required for convergence ', str(self.iteration))

    def policy_evaluation(self): 
            # in place version
            sweeps = 0
            stable = False
            while True:
            # while delta >= self.update_threshold:
                delta = 0
                sweeps += 1
                for state in range(self.mdp[1]):
                    old_state_value = self.state_value_fn[state]
                    new_state_value = 0
                    # sum over potential actions
                    for action in range(self.mdp[2]):
                        new_state_value += self.get_policy_value(state, action)
                    self.state_value_fn[state] = new_state_value
                    delta = max(delta, np.abs(old_state_value - self.state_value_fn[state]))
                if delta < self.update_threshold:
                    stable = True
                    break
            return sweeps

    def get_policy_value(self, state, action): # v_pi(s)
        # Value expectation considering the policy
        policy_value = 0
        for transition in self.mdp[0][state][action]:
            transition_prob = transition[0] # prob of next state p(ss', a) 상태 변환 확률
            successor_state = transition[1] # value/name of next state s' 다음 상태
            reward = transition[2] # reward of next state R 보상
            # action_value = self.get_action_value(state, action)
            policy_value += transition_prob * (reward + self.gamma * self.state_value_fn[successor_state])
            # policy_value += self.policy[state][action] * transition_prob * (reward + gamma * self.state_value_fn[successor_state])
        return policy_value
    
    def get_action_value(self, state, action): #큐함수 
        # Value expectation without considering the policy
        action_value = 0
        for transition in self.mdp[0][state][action]:
            transition_prob = transition[0] # prob of next state
            successor_state = transition[1] # value/name of next state
            reward = transition[2] # reward of next state
            action_value += transition_prob * (reward + self.gamma * self.state_value_fn[successor_state]) 
        return action_value
        
    def policy_improvement(self):
        policy_stable = True
        current_policy = self.policy # Cache current policy
        best_policy = []
        for state in range(self.mdp[1]):
            best_policy.append([0 for _ in range(self.mdp[2])])
            # Calculate best possible policy based on current value function
            action_values = []
            for action in range(self.mdp[2]):
                action_values.append(self.get_action_value(state, action))
            best_actions = np.where(action_values == max(action_values))[0]
            for index in best_actions:
                best_policy[state][index] = 1
            best_policy[state] = [best_policy[state][action] / len(best_actions)
                                  for action in range(self.mdp[2])]
            # If the current policy is not the best policy, update it
            if not np.array_equal(current_policy[state], best_policy[state]):
                policy_stable = False
                self.policy[state] = best_policy[state]
        return policy_stable
    
# checkpoint
field_length = 12
line_position = 6
test_env = BasketballEnv(field_length = field_length, line_position = line_position)
test_agent = Agent(test_env, gamma=0.9, update_threshold=1e-6)
# Does the visualization work?
test_agent.state_value_fn = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
test_agent.visualize()
test_agent.reset()
# Can we evaluate a policy?
test_agent.policy = [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
test_agent.policy_evaluation()
test_agent.visualize()
test_agent.reset()
# Does the training work?
test_agent.train()
test_agent.visualize()
test_agent.render_policy()

# finally, training the agent
env = BasketballEnv(min_score_prob = 0.1, max_score_prob = 0.95, line_position = 2, field_length = 8, render_mode = "rgb_array")
test_agent = Agent(env, gamma = 0.99)
test_agent.train()

video_file_2 = "basketball_training.mp4"
evaluate(env, test_agent.policy, video_file_2)
Video(video_file_2, html_attributes="loop autoplay")