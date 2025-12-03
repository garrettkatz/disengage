import pickle as pk
import numpy as np
import torch as tr
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
from ppo import PPO
import matplotlib.pyplot as pt

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

if __name__ == "__main__":

    # these numbers specific to 345k sample (3% blackbox failrate) leaving 338k for calibration tests
    num_train = 1600 # number of samples for fitting only
    num_valid = 400 # number of samples for testing only, leave some for calibration
    num_calib = 10_000
    num_deploys = 100

    delta = 0.0025 # acceptable failure probability

    npz = np.load("gfd_345k.npz")
    labels = npz["failures"]
    features = npz["init_obs"]
    npz.close()

    print(f"{num_train} training examples ({len(labels)}) total, {features.shape[-1]} features")

    # MLP - same architecture as critic
    # num_hidden = 64 # same as critic
    num_hidden = 16
    regressor = tr.nn.Sequential(
        tr.nn.Linear(features.shape[-1], num_hidden),
        tr.nn.Tanh(),
        tr.nn.Linear(num_hidden, num_hidden),
        tr.nn.Tanh(),
        tr.nn.Linear(num_hidden, 1),
    )

    print(f"MLP: {sum([p.numel()for p in regressor.parameters()])} parameters")

    x = tr.tensor(features).float()
    regressor.load_state_dict(tr.load("ttr_early.pt", weights_only=True))
    with tr.no_grad():
        predictions = regressor(x)

    # apply sigmoids and flatten
    predictions = tr.sigmoid(predictions).flatten().numpy()

    # extract the calibration set
    calib_predictions = predictions[num_train+num_valid:num_train+num_valid+num_calib]
    calib_failures = labels[num_train+num_valid:num_train+num_valid+num_calib]

    # do conformal thing
    statistics = calib_failures * (1. - calib_predictions)
    sort_idx = np.argsort(statistics)
    statistics = statistics[sort_idx]
    calib_failures = calib_failures[sort_idx]
    interval = statistics[int(np.ceil((1-delta)*(num_calib+1)))]
    tau = 1 - interval

    input(f"tau = {tau}...")

    # start episodes

    ################## hyperparameters ##################

    max_ep_len = 8 * 30         # training is 8 seconds * 30 steps per second

    render = True             # render environment on screen
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

    env = RandomFlyThruGateAviary(obs=DEFAULT_OBS,act=DEFAULT_ACT,gui=True)

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

    pt.ion()
    pt.figure()
    pt.show()

    failures = np.zeros(num_deploys, dtype=bool)
    disengages = np.zeros(num_deploys, dtype=bool)
    preds = np.zeros(num_deploys)
    for episode in range(num_deploys):

        obs, info = env.reset(seed=42, options={})
        init_obs = obs

        # go or no-go
        with tr.no_grad():
            x = tr.tensor(init_obs).float().unsqueeze(0)
            pred = tr.sigmoid(regressor(x)[0]).numpy()
        preds[episode] = pred

        if pred > tau:
            print("no-go, too risky")
            disengages[episode] = True
        else:
            net_reward = 0
            start_time = datetime.now().replace(microsecond=0)
            start = time.time()
            # for i in range((env.EPISODE_LEN_SEC+20)*env.CTRL_FREQ):
            for i in range(env.EPISODE_LEN_SEC*env.CTRL_FREQ):
                action = ppo_agent.select_action(obs)
                # print(action)
                action = np.expand_dims(action, axis=0)
                obs, reward, terminated, truncated, info = env.step(action)
                net_reward += reward
                env.render()
                # sync(i, start, env.CTRL_TIMESTEP)
    
                failure = failure_function(env)
                if failure:
                    failures[episode] = True
                    break
    
            print(f"episode {episode}: reward {net_reward}, failure={failure}")

            # clear buffer
            ppo_agent.buffer.clear()

        pt.cla()
        pt.plot(preds[:episode+1], 'bo', label="failure prediction")
        pt.plot([0,episode+1], [tau,tau], 'k:', label="tau")
        pt.legend()
        pt.xlabel("Episode")
        pt.ylabel("Value")
        pt.title(f"{disengages[:episode+1].sum():d} disengaged ({100*disengages[:episode+1].mean():.2f}%), {failures[:episode+1].sum():d} engaged and failed ({100*failures[:episode+1].mean():.2f}%)")
        pt.pause(0.01)

        if episode == 0: input("Enter and record...")

    env.close()

