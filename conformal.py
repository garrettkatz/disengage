"""
First stab at conformal regression approach with linear per-t value functions
Try to verify formal results
"""
import numpy as np
from time import perf_counter
from planar import PlanarEnv
from planar_features import FeatureExtractor
from handcode_planar import Policy
import pickle as pk

if __name__ == "__main__":

    do_fit = False
    do_show = True

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_samples = 512 # number of samples for fitting each value function
    num_rollouts = 64 # number of rollouts used to empirically estimate value function
    num_timesteps = 128

    # setup environment, policy, feature extractor
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )
    policy = Policy(env)
    get_features = FeatureExtractor(env, kernel="quadratic")
    input(f"{get_features.get_feature_dim()} features <<? {num_samples} samples...")

    # fit per-timestep linear value approximators
    if do_fit:
        approximators = {}
        value_means, value_stds, error_means, error_stds = [], [], [], []
        for t in range(num_timesteps):
    
            # get samples of x(t)
            # print(f"{t=} getting samples...")
            observation, _ = env.reset(batch_size=num_samples)
            for _ in range(t):
                action = policy(observation)
                observation, _, _, _, _ = env.step(action)
            features, _ = get_features(observation)
    
            # estimate value at each x(t)
            # print(f"{t=} estimating values...")
            value_labels = np.zeros((num_rollouts, num_samples))
            env.state = env.state.broadcast_to((num_rollouts,))
            observation = env.get_observation()
            for _ in range(t, num_timesteps):
                action = policy(observation)
                observation, reward, _, _, _ = env.step(action)
                value_labels += reward
            value_labels = value_labels.mean(axis=0)
    
            # fit the value regressor
            approximators[t] = np.linalg.lstsq(features, value_labels, rcond=None)[0]
            errors = np.fabs(features @ approximators[t] - value_labels)
    
            # print value distribution and residual error
            print(f"{t=} of {num_timesteps}: value = {value_labels.mean()} +- {value_labels.std()}, error = {errors.mean()} +- {errors.std()}")
    
            value_means.append(value_labels.mean())
            value_stds.append(value_labels.std())
            error_means.append(errors.mean())
            error_stds.append(errors.std())

        with open("conformal.pkl", "wb") as f:
            pk.dump((approximators, value_means, value_stds, error_means, error_stds), f)

    with open("conformal.pkl", "rb") as f:
        (approximators, value_means, value_stds, error_means, error_stds) = pk.load(f)

    if do_show:
        import matplotlib.pyplot as pt
        pt.fill_between(np.arange(num_timesteps), np.array(value_means) - np.array(value_stds), np.array(value_means) + np.array(value_stds), alpha=.1, color='g')
        pt.plot(value_means, 'g-', label="empirical rewards-to-go")
    
        pt.fill_between(np.arange(num_timesteps), np.array(error_means) - np.array(error_stds), np.array(error_means) + np.array(error_stds), alpha=.1, color='r')
        pt.plot(error_means, 'r-', label="regression absolute error")
        pt.ylabel("Metric")
        pt.xlabel("Time-step")
        # pt.yscale("log")
        pt.legend()
        pt.show()
