"""
Based on:
https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/main/test_hover.py
"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from ppo import PPO

from gym_pybullet_drones.utils.Logger import Logger
from RandomFlyThruGateAviary import RandomFlyThruGateAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

def failure_function(env):
    dist = env.collision_distance()
    return dist < .05
    # print(f"dist = {dist}")
    # return False



#################################### Testing ###################################
def collect_data(num_samples, save_period, resume=False):
    print("============================================================================================")

    ################## hyperparameters ##################

    max_ep_len = 8 * 30         # training is 8 seconds * 30 steps per second

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    # just here to re-initialize model before loading weights
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    #####################################################

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

    env = RandomFlyThruGateAviary(obs=DEFAULT_OBS,act=DEFAULT_ACT)

    # state space dimension
    state_dim = 12

    # action space dimension
    action_dim = 4

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.)

    # load checkpoint
    checkpoint_path = "log_dir/thrugate/875_ppo_drone.pth"
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    if resume:

        npz = np.load("gfd.npz")
        old_net_rewards=npz["net_rewards"]
        old_failures=npz["failures"]
        old_init_obs=npz["init_obs"]
        npz.close()

        start_sample = len(old_failures)
        num_samples = start_sample + num_samples

        net_rewards = np.zeros(num_samples)
        failures = np.zeros(num_samples, dtype=bool)
        init_obs = np.empty((num_samples, state_dim))

        net_rewards[:start_sample] = old_net_rewards
        failures[:start_sample] = old_failures
        init_obs[:start_sample] = old_init_obs

    else:

        net_rewards = np.zeros(num_samples)
        failures = np.zeros(num_samples, dtype=bool)
        init_obs = np.empty((num_samples, state_dim))

        start_sample = 0

    for episode in range(start_sample, num_samples):

        obs, info = env.reset(seed=42, options={})
        init_obs[episode] = obs

        start_time = datetime.now().replace(microsecond=0)
        start = time.time()
        # for i in range((env.EPISODE_LEN_SEC+20)*env.CTRL_FREQ):
        for i in range(env.EPISODE_LEN_SEC*env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            # print(action)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            net_rewards[episode] += reward
            # env.render()
            # sync(i, start, env.CTRL_TIMESTEP)

            failures[episode] = failure_function(env)
            if failures[episode]: break

        print(f"episode {episode} of {num_samples}: reward {net_rewards[episode]}, failure={failures[episode]}")

        # clear buffer
        ppo_agent.buffer.clear()

        # save
        if (episode % save_period == 0) or (episode + 1 == num_samples):
            np.savez_compressed("gfd.npz",
                net_rewards=net_rewards[:episode+1],
                failures=failures[:episode+1],
                init_obs=init_obs[:episode+1])

    env.close()

    print("rewards:")
    print(net_rewards)
    print(f"{net_rewards.mean()} +/- {net_rewards.std()}")
    print(f"failure rate = {failures.mean()}")


if __name__ == '__main__':

    do_rollouts = False
    resume = False
    num_samples = 340_000
    save_period = 1000

    if do_rollouts:
        start_time = time.perf_counter()
        collect_data(num_samples, save_period, resume)
        total_time = time.perf_counter() - start_time
        print(f"Total time = {total_time:.1f}, {total_time/num_samples:.2f} per sample")

    npz = np.load("gfd.npz")
    net_rewards=npz["net_rewards"]
    failures=npz["failures"]
    init_obs=npz["init_obs"]
    npz.close()

    # print(init_obs)

    import matplotlib.pyplot as pt
    pt.figure(figsize=(4.5,4))
    pt.hist(net_rewards, ec='k', fc=(.5,.5,1), bins=50)
    pt.xlabel("Net Reward")
    pt.ylabel("Frequency")
    pt.title(f"Reward distribution (failure rate = {100*failures.mean():.3f}%)")
    pt.tight_layout()
    pt.savefig("gfd.eps")
    pt.savefig("gfd.png")
    pt.show()


