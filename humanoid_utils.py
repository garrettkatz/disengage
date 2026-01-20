"""
Adapted from 
https://github.com/carlosferrazza/humanoid-bench/blob/main/humanoid_bench/mjx/mjx_test.py
"""
import os
import mujoco.viewer
import numpy as np
from humanoid_bench.mjx.flax_to_torch import TorchModel, TorchPolicy
from humanoid_bench.mjx.envs.cpu_env import HumanoidNumpyEnv
import tqdm
from humanoid_bench.mjx.video_utils import save_numpy_as_video, make_grid_video_from_numpy

RELATIVE_NOISE = 0 #.5 # maximum percent change in perturbed action magnitudes

def load():

    task = "reach"
    data_path = os.environ["HOME"] + '/humanoid-bench/data/reach_one_hand/'
    env_path = os.environ["HOME"] + \
        '/humanoid-bench/humanoid_bench/assets/mjx/scene_test_mesh_collisions_hands_two_targets_pos.xml'

    env = HumanoidNumpyEnv(env_path, task)

    torch_model = TorchModel(55, 19)
    torch_policy = TorchPolicy(torch_model)

    model_name = "torch_model.pt"
    mean_name = "mean.npy"
    var_name = "var.npy"

    torch_policy.load(
        os.path.join(data_path, model_name),
        mean=os.path.join(data_path, mean_name),
        var=os.path.join(data_path, var_name))

    return env, torch_policy

def run(env, torch_policy, num_timesteps, failure_predicate, perturb_obs=None, render=False):

    data_path = os.environ["HOME"] + '/humanoid-bench/data/reach_one_hand/'
    
    state = env.reset()

    if perturb_obs is not None: state = perturb_obs(state)
    
    if render:

        m, d = env.model, env.data
        # viewer = mujoco.viewer.launch_passive(m, d) # to view live
        renderer = mujoco.Renderer(m, height=480, width=480) # to save video

        state = env.reset()
        observations = [state]
        actions = []

        i = 1
        ep_rew = 0
        video = []
        while True:
            action = torch_policy(state)
            action *= np.random.uniform(1. - RELATIVE_NOISE, 1. + RELATIVE_NOISE, action.shape)
            state, reward, done, info = env.step(action)
            ep_rew += reward
            i += 1
            failure = failure_predicate(env, state, reward, done, info)

            # perturb after checking failure
            if perturb_obs is not None: state = perturb_obs(state)

            observations.append(state)
            actions.append(action)

            renderer.update_scene(d, camera='cam_default') # saved video
            # viewer.sync() # live view
            frame = renderer.render()
            video.append(frame)

            if done or failure or i >= num_timesteps: break

        all_videos = [np.array(video)]
        make_grid_video_from_numpy(all_videos, 1, output_name="evaluation.mp4", **{'fps': 24})

        # viewer.close() # live video

    else:

        state = env.reset()
        observations = [state]
        actions = []

        i = 1
        ep_rew = 0
        while True:
            action = torch_policy(state)
            action *= np.random.uniform(1. - RELATIVE_NOISE, 1. + RELATIVE_NOISE, action.shape)
            # print(action.shape, type(action), action)
            # input('.')
            state, reward, done, info = env.step(action)
            ep_rew += reward
            i += 1

            failure = failure_predicate(env, state, reward, done, info)

            # perturb after checking failure
            if perturb_obs is not None: state = perturb_obs(state)

            observations.append(state)
            actions.append(action)

            if done or failure or i >= num_timesteps: break

    ep_len = i

    # unsqueeze observations for consistency with other environments
    observations = [obs[None,:] for obs in observations]

    return failure, ep_rew, ep_len, observations, actions

if __name__ == '__main__':

    env, model = load()
    render = False
    obs_noise = .3

    def failure_predicate(env, obs, reward, done, info): return done

    def perturb_obs(o):
        return o * np.random.uniform(1 - obs_noise, 1 + obs_noise, size=o.shape)

    failures = []
    for r in range(20):
        failure, ep_rew, ep_len, observations, actions = run(
            env, model, num_timesteps=500, failure_predicate=failure_predicate, perturb_obs=perturb_obs, render=render)
        failures.append(failure)
        print(f"{r=}: {failure=:b} ({ep_rew} reward, {ep_len}={len(observations)} duration)")
        # if failure: break

    print(observations[-1])
    print(observations[-1].shape)
    print(f"failure rate = {np.mean(failures)}")
