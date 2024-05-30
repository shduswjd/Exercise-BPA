import gym
import random
import pygame
import numpy as np

env = gym.make("Taxi-v3", render_mode="human")
# observation space: 500개
# action space: 6개
alpha = 0.4
gamma = 0.999
epsilon = 0.017

# 큐테이블 초기화. 큐테이블: 상태-행동 가치 저장
q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0

# 위의 방법으로 했다가 자꾸 unhashable type: dict 에러가 나서 바꿈
# q = np.zeros((env.observation_space.n, env.action_space.n)) #(500, 6)

# 큐러닝 업데이트 함수
def update_q_table(prev_state, action, reward, next_state, alpha, gamma):
    qa = 0
    for i in range(env.action_space.n):
        qa = np.max(q[next_state, i])
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])


def epsilon_greedy_policy(q, state):
    if random.uniform(0,1) < epsilon: # 무작위 행동 선택
        return env.action_space.sample()
    else: 
        return max(list(range(env.action_space.n)), key = lambda x: q[(state, x)])
        # return np.argmax(q[state])
        
# 실행
for i in range(8000):
    prev_state, info = env.reset()
    total_reward = 0
    while True:

        action = epsilon_greedy_policy(q, prev_state)
        
        next_state, reward, done, truncated, info = env.step(action)
        
        # update q table (return값이 없어서 굳이 값 저장 안해도 됨)
        update_q_table(prev_state, action, reward, next_state, alpha, gamma)

        prev_state = next_state

        total_reward += reward
        if done:
            break
    print("total reward: ", total_reward)

env.close()


