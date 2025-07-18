"""
Conformal regression on probability of failure
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
    do_calibrate = False
    do_test = True
    do_show = True

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_timesteps = 64
    num_samples = 512 # number of samples for fitting regressor
    num_rollouts = 32 # number of rollouts used to empirically estimate regression target
    sig_level = 0.4 #0.05 # overall significance level

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

    # fit regressors
    if do_fit:
        regressors = {}
        thresholds = {}
        target_means, target_stds, error_means, error_stds = [], [], [], []
        for t in range(num_timesteps):
        # for t in range(1): # just start with initial state regressor
    
            # get samples of x(t)
            print(f"{t=} getting samples...")
            observation, _ = env.reset(batch_size=num_samples)
            for _ in range(t):
                action = policy(observation)
                observation, _, _, _, _ = env.step(action)
            features, _ = get_features(observation)
    
            # estimate failure probability at each x(t)
            print(f"{t=} getting labels...")
            reward_to_go = np.zeros((num_rollouts, num_samples))
            env.state = env.state.broadcast_to((num_rollouts,))
            observation = env.get_observation()
            for _ in range(t, num_timesteps):
                action = policy(observation)
                observation, reward, _, _, _ = env.step(action)
                reward_to_go += reward

            # # get 2 stdev below estimated value as unacceptable
            # value_estimates = reward_to_go.mean(axis=0)
            # thresholds[t] = value_estimates.mean() - 2*value_estimates.std()

            # # negative reward-to-go is unacceptable (viewed more often than viewing)
            # thresholds[t] = 0

            # on average, should be unviewed and view at least one each
            thresholds[t] = 2*(num_timesteps - t)

            # fit the regressor
            target_labels = (reward_to_go < thresholds[t]).mean(axis=0)
            regressors[t] = np.linalg.lstsq(features, target_labels, rcond=None)[0]
            errors = np.fabs(features @ regressors[t] - target_labels)
    
            # print value distribution and residual error
            print(f"{t=} of {num_timesteps}: Pr(fail) ~ {target_labels.mean()} +- {target_labels.std()}, error ~ {errors.mean()} +- {errors.std()}")

            target_means.append(target_labels.mean())
            target_stds.append(target_labels.std())
            error_means.append(errors.mean())
            error_stds.append(errors.std())

        with open("cp.pkl", "wb") as f:
            pk.dump((regressors, thresholds, target_means, target_stds, error_means, error_stds), f)

    with open("cp.pkl", "rb") as f:
        (regressors, thresholds, target_means, target_stds, error_means, error_stds) = pk.load(f)

    if do_calibrate:

        # get per-timestep confidence intervals
        pad = {}
        for t in range(num_timesteps):
        # for t in range(1): # just initial state to start
    
            # get samples of x(t)
            print(f"{t=} getting samples...")
            observation, _ = env.reset(batch_size=num_samples)
            for _ in range(t):
                action = policy(observation)
                observation, _, _, _, _ = env.step(action)
            features, _ = get_features(observation)
    
            # get labels at each x(t)
            print(f"{t=} getting labels...")
            reward_to_go = np.zeros((num_rollouts, num_samples))
            env.state = env.state.broadcast_to((num_rollouts,))
            observation = env.get_observation()
            for _ in range(t, num_timesteps):
                action = policy(observation)
                observation, reward, _, _, _ = env.step(action)
                reward_to_go += reward

            # one-sided confidence bands
            # you do not want to under-estimate the chance of failure
            # so you target labels to be upper-bounded near regressor with high probability
            # (1-sig) of data satisfies target < regressor + pad
            target_labels = (reward_to_go < thresholds[t]).mean(axis=0)
            one_sided_errors = target_labels - features @ regressors[t]

            # find the sig_level quantile as the confidence interval padding
            # idx = int(np.ceil((num_samples+1)*(1 - sig_level)))
            idx = int(np.ceil((num_samples+1)*(1 - sig_level / num_timesteps)))
            pad[t] = np.inf if idx >= num_samples else np.sort(one_sided_errors)[idx]
    
            # print confidence interval pad
            print(f"{t=} of {num_timesteps}: label ~ {target_labels.mean()} +- {target_labels.std()}, one-sided error ~ {one_sided_errors.mean()} +- {one_sided_errors.std()}, padding = {pad[t]}")

        with open("cp_pad.pkl", "wb") as f:
            pk.dump((regressors, thresholds, target_means, target_stds, error_means, error_stds, pad), f)

    with open("cp_pad.pkl", "rb") as f:
        (regressors, thresholds, target_means, target_stds, error_means, error_stds, pad) = pk.load(f)

    thresholds = np.array([thresholds[t] for t in range(num_timesteps)])

    if do_test:

        # # test empirically if formal result is correct

        risk_tolerance = 0.1

        # get some rollouts with rewards-to-go
        observation, _ = env.reset(batch_size=num_samples)
        disengaged = np.empty((num_samples, num_timesteps), dtype=bool)
        rewards = np.empty((num_samples, num_timesteps))
        for t in range(num_timesteps):

            features, _ = get_features(observation)
            action = policy(observation)
            observation, rewards[:,t], _, _, _ = env.step(action)
            disengaged[:,t] = (features @ regressors[t] + pad[t] >= risk_tolerance)

        rewards_to_go = np.cumsum(rewards, axis=1)
        undetected = ((rewards_to_go < thresholds) & ~disengaged).any(axis=1).mean()

        print(f"disengage rate = {disengaged.any(axis=1).mean()}, undetected rate = {undetected} vs risk tolerance {risk_tolerance}")

        # # for t in range(num_timesteps):
        # for t in range(1): # just initial state to start
    
        #     # get samples of x(t)
        #     print(f"{t=} getting samples...")
        #     observation, _ = env.reset(batch_size=num_samples)
        #     for _ in range(t):
        #         action = policy(observation)
        #         observation, _, _, _, _ = env.step(action)
        #     features, _ = get_features(observation)
    
        #     # get labels at each x(t)
        #     print(f"{t=} getting labels...")
        #     reward_to_go = np.zeros((num_rollouts, num_samples))
        #     env.state = env.state.broadcast_to((num_rollouts,))
        #     observation = env.get_observation()
        #     for _ in range(t, num_timesteps):
        #         action = policy(observation)
        #         observation, reward, _, _, _ = env.step(action)
        #         reward_to_go += reward

        #     # how often is reward-to-go outside conformal band?
        #     target_labels = (reward_to_go < thresholds[t]).mean(axis=0)
        #     outside = (target_labels > features @ regressors[t] + pad[t])

        #     print(f"{t=} of {num_timesteps}: {outside.mean()} of samples outside conformal range vs significance level {sig_level}")

        # # sample some rollouts
        # decision_points = np.empty((num_samples, num_timesteps))
        # target_estimates = np.empty((num_samples, num_timesteps))
        # undetected = np.empty((num_samples, num_timesteps), dtype=bool)
        # disengaged = np.empty((num_samples, num_timesteps), dtype=bool)
        # observation, _ = env.reset(batch_size=num_samples)
        # for t in range(num_timesteps):
        #     print(f"Engaged to step {t} of {num_timesteps}")

        #     # get the decision point from the value regressor
        #     features, _ = get_features(observation)
        #     decision_points[:,t] = features @ regressors[t] - pad[t]

        #     # save state before value estimate rollouts
        #     save_state = env.state

        #     # estimate value at each x(t)
        #     print(f"{t=} estimating values...")
        #     rewards_to_go = np.zeros((num_rollouts, num_samples))
        #     env.state = env.state.broadcast_to((num_rollouts,))
        #     observation = env.get_observation()
        #     for _ in range(t, num_timesteps):
        #         action = policy(observation)
        #         observation, reward, _, _, _ = env.step(action)
        #         rewards_to_go += reward
        #     target_estimates[:,t] = rewards_to_go.mean(axis=0)

        #     # print undetected errors
        #     disengaged[:,t] = (decision_points[:,t] < thresholds[t])
        #     undetected[:,t] = (target_estimates[:,t] < thresholds[t]) & ~ disengaged[:,t]
        #     print(f"{t=} of {num_timesteps}: {disengaged[:,t].mean()} of samples disengaged, {undetected[:,t].mean()} of samples had undetected errors, vs per-timestep significance {sig_level/num_timesteps}")

        #     # restore state and take another step
        #     env.state = save_state
        #     action = policy(env.get_observation())
        #     observation, _, _, _, _ = env.step(action)

        # print(f"existential disengaged rate = {disengaged.any(axis=1).mean()}")
        # print(f"existential undetected rate = {undetected.any(axis=1).mean()}")

    #     with open("cp_test.pkl", "wb") as f:
    #         pk.dump((thresholds, decision_points, target_estimates, undetected, disengaged), f)

    # with open("cp_test.pkl", "rb") as f:
    #     (thresholds, decision_points, target_estimates, undetected, disengaged) = pk.load(f)


    if do_show:

    #     print("per-timestep undetected rates:")
    #     print(undetected.mean(axis=0))
    #     input(f"Existential undetected rate: {undetected.any(axis=1).mean()}...")

        import matplotlib.pyplot as pt

        # pt.subplot(1,2,1)
        pt.fill_between(np.arange(num_timesteps), np.array(target_means) - np.array(target_stds), np.array(target_means) + np.array(target_stds), alpha=.1, color='g')
        pt.plot(target_means, 'g-', label="failure rate")
    
        pt.fill_between(np.arange(num_timesteps), np.array(error_means) - np.array(error_stds), np.array(error_means) + np.array(error_stds), alpha=.1, color='r')
        pt.plot(error_means, 'r-', label="regression absolute error")

        pt.plot([pad[t] for t in range(num_timesteps)], 'k-', label="confidence interval pad")

        pt.ylabel("Metric")
        pt.xlabel("Time-step")
        # pt.yscale("log")
        pt.legend()

    #     # pt.subplot(1,2,2)
    #     # pt.plot(disengaged.mean(axis=0), 'b-', label='disengaged rate')
    #     # pt.plot(undetected.mean(axis=0), 'r-', label='undetected rate')
    #     # pt.plot([0, num_timesteps], [disengaged.any(axis=0).mean()]*2, 'b:')
    #     # pt.plot([0, num_timesteps], [undetected.any(axis=0).mean()]*2, 'r:')

    #     # pt.plot([0, num_timesteps], [sig_level]*2, 'g-', label='alpha')
    #     # pt.plot([0, num_timesteps], [sig_level/num_timesteps]*2, 'g:')

    #     # pt.legend()
    #     # pt.ylabel("Rate")
    #     # pt.xlabel("Time-step")
    #     # # pt.yscale("log")

        pt.savefig("cp.png")
        pt.show()

