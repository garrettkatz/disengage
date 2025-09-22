"""
Monte-Carlo estimate failure rate as training data for conformal regression
"""
import numpy as np
from planar import PlanarEnv
from handcode_planar import Policy
import pickle as pk
import matplotlib.pyplot as pt

def do_rollouts(env, policy, init_states, num_rollouts, num_timesteps, failure_function, verbose=False):
    failures = np.zeros((init_states.batch_size(), num_rollouts), dtype=bool)
    observation = env.get_observation(init_states)
    for r in range(num_rollouts):
        env.state = init_states
        for t in range(num_timesteps):
            failures[:,r] = failures[:,r] | failure_function(env)
            action = policy(observation)
            observation, _, _, _, _ = env.step(action)
        if verbose and (num_rollouts < 10 or (r % (num_rollouts//10)) == 0):
            fail_rates = failures[:,:r+1].mean(axis=1)
            print(f"rollout {r} of {num_rollouts}, fail_rates: {fail_rates.min()} <= ~{fail_rates.mean()} +/- {fail_rates.std()} <= {fail_rates.max()}")

    fail_rates = failures.mean(axis=1)
    return fail_rates

def failure_function(env):
    """
    Assumes env.state is set and batched
    """
    _, viz = env._team_reward(env.state.adversaries, env.state.allies)
    return viz.any(axis=(-2,-1))    

if __name__ == "__main__":

    collect_data = False
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
        pt.figure(figsize=(10,4))

        for i, (ylo, yhi, name) in enumerate([(fail_rates-hoeffding, fail_rates + hoeffding, "Hoeffding"), (p_u, p_o, "Clopper-Pearson")]):

            pt.subplot(1,2,i+1)
            pt.fill_between(1+np.arange(len(idx)), ylo[idx], yhi[idx], color=(.75,)*3, label="Confidence interval")
            pt.plot(1+np.arange(len(idx)), fail_rates[idx], label="Fail rate")
            pt.xscale("log")
            pt.ylim([fail_rates.min()-hoeffding - .05, fail_rates.max()+hoeffding + .05])
            pt.title(name)

        pt.gcf().supxlabel("Initial states (sorted by estimated fail rate)")
        pt.gcf().supylabel("Estimated fail rate")
        pt.tight_layout()
        pt.savefig("get_failrate_data.eps")
        pt.show()
    
