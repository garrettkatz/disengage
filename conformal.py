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

    do_fit = True
    do_calibrate = True
    do_test = True
    do_show = True

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_samples = 512 # number of samples for fitting each value function
    num_rollouts = 64 # number of rollouts used to empirically estimate value function
    num_timesteps = 32#128
    sig_level = 0.05 # overall significance level

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

    if do_calibrate:

        # get per-timestep confidence intervals
        width = {}
        for t in range(num_timesteps):
    
            # get samples of x(t)
            print(f"{t=} getting samples...")
            observation, _ = env.reset(batch_size=num_samples)
            for _ in range(t):
                action = policy(observation)
                observation, _, _, _, _ = env.step(action)
            features, _ = get_features(observation)
    
            # estimate value at each x(t)
            print(f"{t=} estimating values...")
            value_labels = np.zeros((num_rollouts, num_samples))
            env.state = env.state.broadcast_to((num_rollouts,))
            observation = env.get_observation()
            for _ in range(t, num_timesteps):
                action = policy(observation)
                observation, reward, _, _, _ = env.step(action)
                value_labels += reward
            value_labels = value_labels.mean(axis=0)
    
            # get the value regressor's absolute errors on each sample
            errors = np.fabs(features @ approximators[t] - value_labels)

            # find the sig_level quantile as the confidence interval width
            idx = int(np.ceil((num_samples+1)*(1 - sig_level)))
            width[t] = np.inf if idx == num_samples else np.sort(errors)[idx]
    
            # print confidence interval width
            print(f"{t=} of {num_timesteps}: value = {value_labels.mean()} +- {value_labels.std()}, error = {errors.mean()} +- {errors.std()}, width = {width[t]}")

        with open("conformal_width.pkl", "wb") as f:
            pk.dump((approximators, value_means, value_stds, error_means, error_stds, width), f)

    with open("conformal_width.pkl", "rb") as f:
        (approximators, value_means, value_stds, error_means, error_stds, width) = pk.load(f)

    if do_test:

        # test empirically if formal result is correct

        # set thresholds at 1 stdev below mean value
        thresholds = np.array(value_means) - np.array(value_stds)

        # sample some rollouts
        decision_points = np.empty((num_samples, num_timesteps))
        value_estimates = np.empty((num_samples, num_timesteps))
        undetected = np.empty((num_samples, num_timesteps), dtype=bool)
        disengaged = np.empty((num_samples, num_timesteps), dtype=bool)
        observation, _ = env.reset(batch_size=num_samples)
        for t in range(num_timesteps):
            print(f"Engaged to step {t} of {num_timesteps}")

            # get the decision point from the value regressor
            features, _ = get_features(observation)
            decision_points[:,t] = features @ approximators[t] - width[t]

            # save state before value estimate rollouts
            save_state = env.state

            # estimate value at each x(t)
            print(f"{t=} estimating values...")
            rewards_to_go = np.zeros((num_rollouts, num_samples))
            env.state = env.state.broadcast_to((num_rollouts,))
            observation = env.get_observation()
            for _ in range(t, num_timesteps):
                action = policy(observation)
                observation, reward, _, _, _ = env.step(action)
                rewards_to_go += reward
            value_estimates[:,t] = rewards_to_go.mean(axis=0)

            # print undetected errors
            disengaged[:,t] = (decision_points[:,t] < thresholds[t])
            undetected[:,t] = (value_estimates[:,t] < thresholds[t]) & ~ disengaged[:,t]
            print(f"{t=} of {num_timesteps}: {disengaged[:,t].mean()} of samples disengaged, {undetected[:,t].mean()} of samples had undetected errors, vs per-timestep significance {sig_level/num_timesteps}")

            # restore state and take another step
            env.state = save_state
            action = policy(env.get_observation())
            observation, _, _, _, _ = env.step(action)

        print(f"existential disengaged rate = {disengaged.any(axis=1).mean()}")
        print(f"existential undetected rate = {undetected.any(axis=1).mean()}")

        with open("conformal_test.pkl", "wb") as f:
            pk.dump((thresholds, decision_points, value_estimates, undetected, disengaged), f)

    with open("conformal_test.pkl", "rb") as f:
        (thresholds, decision_points, value_estimates, undetected, disengaged) = pk.load(f)


    if do_show:

        print("per-timestep undetected rates:")
        print(undetected.mean(axis=0))
        input(f"Existential undetected rate: {undetected.any(axis=1).mean()}...")

        import matplotlib.pyplot as pt

        # pt.subplot(1,2,1)
        pt.fill_between(np.arange(num_timesteps), np.array(value_means) - np.array(value_stds), np.array(value_means) + np.array(value_stds), alpha=.1, color='g')
        pt.plot(value_means, 'g-', label="empirical rewards-to-go")
    
        pt.fill_between(np.arange(num_timesteps), np.array(error_means) - np.array(error_stds), np.array(error_means) + np.array(error_stds), alpha=.1, color='r')
        pt.plot(error_means, 'r-', label="regression absolute error")

        pt.plot([width[t] for t in range(num_timesteps)], 'k-', label="confidence interval width")

        pt.ylabel("Metric")
        pt.xlabel("Time-step")
        # pt.yscale("log")
        pt.legend()

        # pt.subplot(1,2,2)
        # pt.plot(disengaged.mean(axis=0), 'b-', label='disengaged rate')
        # pt.plot(undetected.mean(axis=0), 'r-', label='undetected rate')
        # pt.plot([0, num_timesteps], [disengaged.any(axis=0).mean()]*2, 'b:')
        # pt.plot([0, num_timesteps], [undetected.any(axis=0).mean()]*2, 'r:')

        # pt.plot([0, num_timesteps], [sig_level]*2, 'g-', label='alpha')
        # pt.plot([0, num_timesteps], [sig_level/num_timesteps]*2, 'g:')

        # pt.legend()
        # pt.ylabel("Rate")
        # pt.xlabel("Time-step")
        # # pt.yscale("log")

        pt.savefig("conformal.png")
        pt.show()
