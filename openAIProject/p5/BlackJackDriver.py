# Team Members: Yuzhang Guo, Zigeng Zhu

import gym
import numpy as np
from p5.DeepRLAgent import DeepRLAgent
import tensorflow as tf
import sys


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    env.reset()

    if sys.argv[1] == '-train':
        resume = True
        batch_size = 1
        num_batch = 100000

        agent = DeepRLAgent(len(env.observation_space.sample()), env.action_space, learning_rate=0.001, exploration_rate=1, exploration_decay_rate=1-1e-5, discount= 0.6)
        if resume:
            agent.load("model/model.tf")

        for batch in range(num_batch):
            agent.reset()
            obs, acts, rewards = [], [], []

            for i in range(batch_size):
                observation = env.reset()
                done = False
                t=0
                total_reward = 0
                while not done:
                    obs.append(observation)
                    # print(observation)
                    action = agent.act(observation)
                    # print(action)
                    observation, reward, done, _ = env.step(action)
                    # print(reward)
                    acts.append(action)
                    rewards.append(reward*200 if reward == 1 else reward*50)
                    total_reward += reward

                    # env.render()  #comment out for faster training!
                    # print(observation)
                    # action = env.action_space.sample() #random action, use your own action policy here
                    # observation, reward, done, info = env.step(action)
                    t += 1

                print("Episode finished after {} timesteps with reward {} ".format(t, total_reward) )

            agent.update(obs, acts, rewards)

            if (batch+1) % 1000 == 0:
                agent.save("model/model.tf")

        agent._sess.close()
        env.close()

    elif sys.argv[1] == '-test':

        agent = DeepRLAgent(len(env.observation_space.sample()), env.action_space, exploration_rate=0.0)
        agent.load("model/model.tf")

        for i in range(10):
            observation = env.reset()
            total_reward = 0
            while True:
                action = agent.act(observation)
                new_observation, reward, done, info = env.step(action)
                observation = new_observation
                total_reward += reward
                if done:
                    break
            print("Episode {} total reward: {}".format(i+1, total_reward))

        agent._sess.close()
        env.close()
