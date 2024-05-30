# 큐 함수랑 비슷하지만 수식에 max_q가 아니고
# 엡실론 그리디 폴리시로 action 을 취한 q을 가져간다

import gym
import random
env = gym.make("Taxi-v3", render_mode = "human")

alpha = 0.85
gamma = 0.90
epsilon = 0.8

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0

# epsilon greedy policy
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x : q[(state, x)])
    

# iteration
for _ in range(4000):
    prev_state, info = env.reset()
    total_reward = 0
    action = epsilon_greedy_policy(prev_state, epsilon)
    while True:
        next_state, reward, done, truncated, info = env.step(action)

        # sarsa algorithm
        next_action = epsilon_greedy_policy(next_state, epsilon)
        q[(prev_state, action)] += alpha * (reward + gamma * q[(next_action, next_action)] - q[(prev_state, action)])

        action = next_action
        prev_state = next_state
        total_reward += reward

        if done:
            break
env.close()
        
