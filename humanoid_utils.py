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

RELATIVE_NOISE = .5 # maximum percent change in perturbed action magnitudes

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

def run(env, torch_policy, num_timesteps=1000, render=False):

    data_path = os.environ["HOME"] + '/humanoid-bench/data/reach_one_hand/'
    
    state = env.reset()
    # print("State:", state)
    
    if render:

        m, d = env.model, env.data
        # viewer = mujoco.viewer.launch_passive(m, d) # to view live
        renderer = mujoco.Renderer(m, height=480, width=480) # to save video

        state = env.reset()
        i = 0
        reward = 0
        video = []
        while True:
            action = torch_policy(state)
            action *= np.random.uniform(1. - RELATIVE_NOISE, 1. + RELATIVE_NOISE, action.shape)
            state, r, done, _ = env.step(action)
            reward += r
            i += 1
            renderer.update_scene(d, camera='cam_default') # saved video
            # viewer.sync() # live view
            frame = renderer.render()
            video.append(frame)
            if done or i > num_timesteps:
                break
        all_videos = [np.array(video)]
        make_grid_video_from_numpy(all_videos, 1, output_name="evaluation.mp4", **{'fps': 24})
        print("Net reward:", reward)

        # viewer.close() # live video

    else:

        state = env.reset()
        i = 0
        reward = 0
        while True:
            action = torch_policy(state)
            action *= np.random.uniform(1. - RELATIVE_NOISE, 1. + RELATIVE_NOISE, action.shape)
            # print(action.shape, type(action), action)
            # input('.')
            state, r, done, _ = env.step(action)
            reward += r
            i += 1
            if done or i > num_timesteps:
                break
            print(i, reward)


if __name__ == '__main__':

    env, model = load()
    run(env, model, num_timesteps=500, render=True)
    
