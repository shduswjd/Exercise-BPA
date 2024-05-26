import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import numpy as np
import custom_envs
from render_util import plot_action_value
from matplotlib.animation import FuncAnimation
%matplotlib inline
from IPython.display import Video
from IPython.display import display
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Agent():
    def __init__(self, env, gamma=0.9):
        """ Initializes the environment and defines dynamics.
        
        Please DON'T change the names of the variables that are already defined in this method.
        """
        self.env = env
        self.action_value_fn = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.C = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # Cumulative sum of ratios
        self.gamma = gamma

        # Generate Policies
        # 행위 정책: 에이전트가 따르는 정책
        # 행위 정책은 타겟 정책과 무관
        # 모든 가능한 상태와 행동을 탐험한다고 해서 소프트 정책이라고도 불림
        def make_random_policy():
            def policy(obs):
                b = np.ones(self.env.action_space.n, dtype=float) / self.env.action_space.n
                return b
            return policy
        self.behavior_policy = make_random_policy()

        # 타겟 정책: 에이전트가 평가하고 개선하는 정책
        # 최대 가치를 갖는 정책을 선택하기 때문에 탐욕 정책이라고 불린다 
        def make_target_policy():
            # Updating continously based on the action value function
            def policy(obs):
                pi = np.zeros(self.env.action_space.n, dtype=float) #[0, 0, 0, 0]
                best_action = np.argmax(self.action_value_fn[obs]) 
                pi[best_action] = 1.0
                return pi
            return policy
        self.target_policy = make_target_policy()

# check point
# Do the policies return values for all the actions?
map = ["SFFH", "FFFH", "HFFH", "HFFG"]
test_env = gym.make('CustomFrozenLake-v1', render_mode=None, desc=map, is_slippery=False) 
test_agent = Agent(test_env)
assert len(test_agent.target_policy(0)) == 4

def get_action(self, policy, obs):
    return np.random.choice(np.arange(len(policy(obs))), p=policy(obs))

setattr(Agent, 'get_action', get_action)
# checkpoint
# Can we pick an action?
map = ["SFFH", "FFFH", "HFFH", "HFFG"]
test_env = gym.make('CustomFrozenLake-v1', render_mode=None, desc=map, is_slippery=False) 
test_agent = Agent(test_env)
test_agent.action_value_fn[0] = [1, 0, 0, 0]
assert test_agent.get_action(test_agent.target_policy, 0) == 0

def train(self, num_episodes, episode_max_duration=100):
    """ Trains the Agent using the given algorithm.

    Inputs:
        num_episodes: Number of episodes for which the training lasts.
        episdoe_max_duration: Maximal duration of an episode. Once the number of steps reaches the threshold,
            the episode is terminated.
    """
    
    # Run through episodes sampled to improve policy incrementally
    for i_episode in range(1, num_episodes + 1):
        # Generate an episode using the behavior policy [(obs, action, reward), (...), ...]
        episode = []
        obs, info = env.reset()
        for t in range(episode_max_duration):
            action = self.get_action(self.behavior_policy, obs)
            next_obs, reward, done, truncated, info = env.step(action)
            episode.append((obs, action, reward))
            if done:
                break
            obs = next_obs
        episode = np.array(episode)
        episode_duration = len(episode[:,:1])
        # Calculate returns and update the policy using weighted importance sampling from the back to save resources
        G = 0.0                                           # Sum of discounted returns
        W = 1.0                                           # Ratios
        for i in range(episode_duration - 1, -1, -1):
            obs = int(episode[i][0])
            action = int(episode[i][1])
            reward = episode[i][2]
            # Update the return
            G = update_return(self.gamma, G, reward)
            # Sum up all the sampling ratios
            self.C[obs][action] += W 
            # Update the action value function (implicitly updates the target policy as well)
            self.action_value_fn[obs][action] += (W / self.C[obs][action]) * (G - self.action_value_fn[obs][action])
            # Update the current sampling ratio
            W = update_W(W, self.target_policy(obs)[action], self.behavior_policy(obs)[action]) 
            if W == 0:
                break


def update_return(gamma, G, reward):
    return gamma * G + reward

def update_W(W, target_policy, behavior_policy):
    return W * target_policy/behavior_policy

setattr(Agent, 'update_return', update_return)
setattr(Agent, 'update_W', update_W)
setattr(Agent, 'train', train)

# checkpoint
assert update_return(1.0, 1.0, 1.0) != 0.0
assert update_W(1.0, 0.75, 0.75) == 1.0

# finally: trining the agent
def evaluate(self, env, file, num_runs=5):
    """ Evaluates the agent in the environment.

    Args:
        env: Environment we want to use. 
        file: File used for storing the video.
        num_runs: Number of runs displayed
    """
    frames = []  # collect rgb_image of agent env interaction
    video_created = False
    for _ in range(num_runs):
        done = False
        obs, info = env.reset()
        out = env.render()
        frames.append(out)
        while not done:
            action = self.get_action(self.target_policy, obs)
            obs, reward, done, truncated, info = env.step(action)
            out = env.render()
            frames.append(out)
    # create animation out of saved frames
    if all(frame is not None for frame in frames):
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        img = plt.imshow(frames[0])
        def animate(index):
            img.set_data(frames[index])
            return [img]
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
        plt.close()
        anim.save(file, writer="ffmpeg", fps=5) 
    return

setattr(Agent, 'evaluate', evaluate)
setattr(Agent, 'plot_action_value', plot_action_value)

map = ["SFFH", "FFFH", "HFFH", "HFFG"]
env = gym.make('CustomFrozenLake-v1', render_mode='rgb_array', desc=map, is_slippery=False) 
env.reset()
agent = Agent(env, gamma=0.9)
agent.train(num_episodes=5000)
agent.plot_action_value()
video = "final_run.mp4"
agent.evaluate(env, video, num_runs=5)
Video(video, html_attributes="loop autoplay")