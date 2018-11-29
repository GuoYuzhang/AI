import gym
import numpy as np
from p5.DeepRLAgent import DeepRLAgent
import tensorflow as tf
from p5.CNNAgent import CNNAgent

resume = False
batch_size = 10
num_batch = 100


def preprocess(prev_image, image):
    image = image - prev_image
    print(image)
    return image


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    env.reset()

    agent = CNNAgent(env.observation_space.shape, env.action_space)
    if resume:
        agent.load("model/model.tf")

    for batch in range(num_batch):
        agent.reset()
        obs, acts, rewards = [], [], []

        for i in range(batch_size):
            observation = env.reset()
            prev = np.zeros_like(observation)
            done = False
            t=0
            total_reward = 0
            while not done:
                image = preprocess(prev, np.array(observation))
                obs.append(image)
                prev = observation
                action = agent.act(observation)
                observation, reward, done, _ = env.step(action)
                acts.append(action)
                rewards.append(reward)
                total_reward += reward

                # env.render()  #comment out for faster training!
                # print(observation)
                # action = env.action_space.sample() #random action, use your own action policy here
                # observation, reward, done, info = env.step(action)
    #             t += 1
    #
    #         print("Episode finished after {} timesteps with reward {} ".format(t, total_reward) )
    #
    #     agent.update(obs, acts, rewards)
    #
    #     if (batch+1) % 1000 == 0:
    #         agent.save("model/model.tf")
    #     print
    #
    #
    # for i in range(10):
    #     observation = env.reset()
    #     agent._exploration_rate = 0.0
    #     total_reward = 0
    #     for j in range(500):
    #         env.render()
    #         action = agent.act(observation)
    #         new_observation, reward, done, info = env.step(action)
    #         observation = new_observation
    #         total_reward += reward
    #         if done:
    #             break
    #     print(total_reward)
    #
    # agent._sess.close()

    env.close()
