"""
Estimate failure rate
failure means any ally is viewed by enemy
"""
import numpy as np
from time import perf_counter
from planar import PlanarEnv
from planar_features import FeatureExtractor
from handcode_planar import Policy
import pickle as pk

if __name__ == "__main__":

    do_estimate = True
    do_show = True

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_samples = 1024 # number of samples for fitting each value function
    num_timesteps = 4096

    # setup environment, policy, feature extractor
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )
    policy = Policy(env)
    # get_features = FeatureExtractor(env, kernel="quadratic")

    # estimate per-timestep failure rates
    if do_estimate:

        failure = np.empty((num_samples, num_timesteps), dtype=bool)
        observation, _ = env.reset(batch_size=num_samples, upperhand=True)
        for t in range(num_timesteps):
            _, viz = env._team_reward(env.state.adversaries, env.state.allies)
            failure[:,t] = viz.any(axis=(-2,-1))

            action = policy(observation)
            observation, _, _, _, _ = env.step(action)

        with open("failrate.pkl", "wb") as f:
            pk.dump((failure,), f)

    with open("failrate.pkl", "rb") as f:
        (failure,) = pk.load(f)

    if do_show:

        over = np.zeros(num_samples, dtype=bool)
        failrate = np.full(num_timesteps, np.nan)
        overrate = np.empty(num_timesteps)
        for t in range(num_timesteps):
            failrate[t] = failure[~over, t].mean()
            over = over | failure[:,t]
            overrate[t] = over.mean()

        print(failrate)

        import matplotlib.pyplot as pt

        pt.figure(figsize=(5,3), constrained_layout=True)
        pt.plot(overrate, 'k-', label='failed by $t$')
        # pt.plot(failrate, 'b--', label='fail at t')
        pt.xlabel("Timestep $t$")
        pt.ylabel("Estimate of $\\text{Pr}(F_{\\leq t})$")
        # pt.legend()

        pt.savefig("failrate.eps")
        pt.show()
