"""
Adapted from 
https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/da52ed17e0bc1923a1f0eb7d7d2cecdf01aec4f9/test_thrugate.py
"""
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as pt
import torch as tr

from gym_pybullet_drones.utils.Logger import Logger
from FlyThruRuinsAviary import FlyThruRuinsAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from ppo import PPO

np.set_printoptions(linewidth=100)

def load(checkpoint_name, render=False):

    ################## hyperparameters ##################

    action_std = 0.1            # set same std for action distribution which was used while saving
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    DEFAULT_GUI = render
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        print(filename)
        os.makedirs(filename+'/')

    env = FlyThruRuinsAviary(gui=DEFAULT_GUI,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           # record=DEFAULT_RECORD_VIDEO,
                           )

    # state space dimension
    state_dim = 12

    # action space dimension
    action_dim = 4

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # load pretrained weights
    checkpoint_path = f"ruins_log_dir/ruins/{checkpoint_name}.pth"
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    return env, ppo_agent

# @profile
def run(env, ppo_agent, perturb_obs=None, render=False):
    # if failure, last state in rollout is failure state

    # obs, info = env.reset(seed=42, options={})
    obs, info = env.reset()

    if perturb_obs is not None: obs = perturb_obs(obs)

    observations = [obs]
    actions = []
    dep_imgs = []

    ep_rew = 0
    ep_len = 1
    start_time = datetime.now().replace(microsecond=0)
    start = time.time()
    failed = False

    for i in range(1, env.EPISODE_LEN_SEC*env.CTRL_FREQ):
        ep_len += 1

        with tr.no_grad():
            action = ppo_agent.select_action(obs)

        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        failed = env.failure_predicate() or (truncated and not terminated)
        ep_rew += reward

        # perturb after checking failure
        if perturb_obs is not None: obs = perturb_obs(obs)

        observations.append(obs)
        actions.append(action)

        if render:
            print(f"timestep {i}: net reward = {ep_rew:.3f}, {terminated=}, {truncated=}, {failed=}")
            env.render()
            deps = env.depth_sense()
            dep_imgs.append(deps)
            sync(i, start, env.CTRL_TIMESTEP)

        if failed:
            # print("Failed!")
            break

        if terminated or truncated:
            # if truncated: print("Trunk!")
            break

    # clear buffer
    ppo_agent.buffer.clear()

    return failed, ep_rew, ep_len, (observations, dep_imgs), actions

if __name__ == "__main__":
    # model, _ = load_model("a2c", "LunarLander-v3")

    render = True

    # env, model = load("23087_ppo_drone", render)
    env, model = load("41652_ppo_drone", render)
    print(env)
    print(model.policy)
    # input('.')

    def perturb_obs(o):
        return o * np.random.uniform(1 - .05, 1 + .05, size=o.shape)
        # return o * np.random.uniform(1 - 10., 1 + 10., size=o.shape)

    for rep in range(100):
        failure, ep_rew, ep_len, observations, actions = run(env, model, perturb_obs, render)
        observations, dep_imgs = observations
        print(f"{failure=:b} (dur={ep_len}={len(observations)})")
        input('.')
        
        pt.figure(figsize=(4,4))
        pt.ion()
        pt.show()

        print(f"depth shape = {dep_imgs[0].shape}")
        
        for t, deps in enumerate(dep_imgs):
            for d, dep in enumerate(deps): # one channel per drone
                pt.subplot(1,len(deps),d+1)
                pt.cla()
                pt.imshow(dep)
            pt.pause(0.01)

        input('.')

