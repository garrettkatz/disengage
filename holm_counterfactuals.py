import numpy as np
import matplotlib.pyplot as pt
from planar import PlanarEnv
import planar_features as pf
import handcode_planar as hp
from scipy.stats import ttest_ind

if __name__ == "__main__":

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_rollouts = 10
    num_timesteps = 200
    num_alternatives = 10
    num_inits = 100

    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )

    # helper to evaluate any policy
    def do_rollouts(init_state, policy):
        env.state = init_state.expand_to(num_rollouts)
        observation = env.get_observation()
        
        actions = np.empty((num_timesteps, num_rollouts, env.num_allies, 3))
        net_rewards = np.zeros(num_rollouts)
        for t in range(num_timesteps):
            actions[t] = policy(observation)
            observation, reward, _, _, _ = env.step(actions[t])
            net_rewards += reward

        return net_rewards, actions

    # instantiate blackbox policy
    blackbox = hp.Policy(env)

    pvals = np.empty((num_inits, num_alternatives))

    for init in range(num_inits):

        # set up initial state and observation for rollouts
        init_state = env.random_state()
        init_obs = env.get_observation(init_state)

        # helper to sample alternative policies
        # weighed average between blackbox and alternative action
        # weights based on distance from initial observation
        def alternative_policy_factory():
            alternative_action = env.action_space.sample()
            scale = np.random.uniform()
            def policy(obs):
                alpha = - scale * np.mean((obs - init_obs)**2, axis=-1)
                alpha = np.exp(alpha)[...,None,None]
                return alpha * alternative_action + (1-alpha) * blackbox(obs)
            return policy
    
        blackbox_rewards, blackbox_actions = do_rollouts(init_state, blackbox)
        bb_avg = blackbox_rewards.mean()
        print(f"init {init} of {num_inits}: {bb_avg=}")

        # variance in actions
        variance = np.sum((blackbox_actions - blackbox_actions.mean(axis=1, keepdims=True))**2, axis=(1,2,3))
        
        pt.plot(variance)
        pt.show()
    
        alternative_rewards = []
        for alt in range(num_alternatives):
            alternative = alternative_policy_factory()
            alt_rewards, alt_actions = do_rollouts(init_state, alternative)
            alternative_rewards.append(alt_rewards)

            result = ttest_ind(blackbox_rewards, alternative_rewards[-1], equal_var=False, alternative="less")
            pvals[init, alt] = result.pvalue
            avg = alternative_rewards[-1].mean()
            if avg > bb_avg: print( f" found better: {avg} (pval={pvals[init,alt]})")

