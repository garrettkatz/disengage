"""
Monte-Carlo estimate failure rate as training data for conformal regression
"""
import numpy as np
from planar import PlanarEnv
from handcode_planar import Policy
import pickle as pk
import matplotlib.pyplot as pt

def do_rollouts(env, policy, init_states, num_rollouts, num_timesteps, failure_function, verbose=False):
    failures = np.zeros((num_samples, num_rollouts), dtype=bool)
    observation = env.get_observation(init_states)
    for r in range(num_rollouts):
        env.state = init_states
        for t in range(num_timesteps):
            failures[:,r] = failures[:,r] | failure_function(env)
            action = policy(observation)
            observation, _, _, _, _ = env.step(action)
        if verbose and (r % (num_rollouts//10)) == 0:
            fail_rates = failures[:,:r+1].mean(axis=1)
            print(f"rollout {r} of {num_rollouts}, fail_rates: {fail_rates.min()} <= ~{fail_rates.mean()} +/- {fail_rates.std()} <= {fail_rates.max()}")

    fail_rates = failures.mean(axis=1)
    return fail_rates

if __name__ == "__main__":

    collect_data = True
    do_show = True

    # experiment parameters
    num_timesteps = 100
    num_samples = 10_000 # number of state samples
    num_rollouts = 512 # number of rollouts per state to estimate state-conditioned failure rate
    confidence = 0.05 # confidence for Hoeffding interval

    # setup environment and policy
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )
    policy = Policy(env)

    def failure_function(env):
        """
        Assumes env.state is set and batched
        """
        _, viz = env._team_reward(env.state.adversaries, env.state.allies)
        return viz.any(axis=(-2,-1))    

    if collect_data:

        observation, _ = env.reset(batch_size=num_samples, upperhand=True)
        init_states = env.state    
        fail_rates = do_rollouts(env, policy, init_states, num_rollouts, num_timesteps, failure_function, verbose=True)
    
        print(f"marginal failure rate stats: {fail_rates.mean()} +/- {fail_rates.std()}")    
        with open(f"failrate_data.pkl", "wb") as f:
            pk.dump((env, init_states, fail_rates), f)

    if do_show:

        with open(f"failrate_data.pkl", "rb") as f:
            (env, init_states, fail_rates) = pk.load(f)

        # setup half-width of hoeffding confidence interval
        hoeffding = np.sqrt(- np.log(confidence/2) / (2*num_rollouts))

        # setup clopper-pearson interval, much tighter than hoeffding
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
        from scipy.stats import beta
        k = num_rollouts * fail_rates
        n = num_rollouts
        p_u = beta.ppf(confidence / 2, k, n - k + 1)
        p_o = beta.ppf(1 - confidence / 2, k + 1, n - k)
        p_o[np.isnan(p_o)]= 1
        p_u[np.isnan(p_u)]= 0

        idx = np.argsort(-fail_rates)
        # ylo = fail_rates - hoeffding
        # yhi = fail_rates + hoeffding
        ylo = p_u
        yhi = p_o
        pt.figure(figsize=(5,3))
        pt.fill_between(1+np.arange(len(idx)), ylo[idx], yhi[idx], color=(.75,)*3, label="Confidence interval")
        pt.plot(1+np.arange(len(idx)), fail_rates[idx], label="Fail rate")
        pt.xlabel("Initial states (sorted by estimated fail rate)")
        pt.ylabel("Estimated fail rate")
        pt.xscale("log")
        pt.tight_layout()
        pt.savefig("get_failrate_data.eps")
        pt.show()
    
