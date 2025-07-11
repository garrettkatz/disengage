import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from planar import PlanarEnv
import planar_features as pf
import handcode_planar as hp
from scipy.stats import ttest_ind

if __name__ == "__main__":

    do_run = False

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_rollouts = 30
    num_timesteps = 200
    num_alternatives = 100
    num_inits = 5000
    reward_cutoff = 300

    sig_level = 0.05

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

    bb_avgs = np.empty(num_inits)
    pvals = np.empty((num_inits, num_alternatives))
    alt_avgs = np.empty((num_inits, num_alternatives))

    if do_run:
        for init in range(num_inits):
    
            # set up initial state and observation for rollouts
            init_state = env.random_state()
            init_obs = env.get_observation(init_state)
    
            # helper to sample alternative policies
            # weighed average between blackbox and alternative action
            # weights based on distance from initial observation
            def alternative_policy_factory():
                alternative_action = env.action_space.sample()
                # scale = np.random.uniform()
                scale = np.random.exponential()
                def policy(obs):
                    alpha = - scale * np.mean((obs - init_obs)**2, axis=-1)
                    alpha = np.exp(alpha)[...,None,None]
                    return alpha * alternative_action + (1-alpha) * blackbox(obs)
                return policy
        
            blackbox_rewards, blackbox_actions = do_rollouts(init_state, blackbox)
            bb_avgs[init] = blackbox_rewards.mean()
            print(f"init {init} of {num_inits}: blackbox avg = {bb_avgs[init]}")
    
            # only look for counterfactuals in lowest-performing states
            if bb_avgs[init] > reward_cutoff: continue
    
            # # variance in actions
            # variance = np.sum((blackbox_actions - blackbox_actions.mean(axis=1, keepdims=True))**2, axis=(1,2,3))        
            # pt.plot(variance)
            # pt.show()
        
            alternative_rewards = []
            for alt in range(num_alternatives):
                alternative = alternative_policy_factory()
                alt_rewards, alt_actions = do_rollouts(init_state, alternative)
                alternative_rewards.append(alt_rewards)
                avg = alt_rewards.mean()

                result = ttest_ind(blackbox_rewards, alt_rewards, equal_var=False, alternative="less")
                pvals[init, alt] = result.pvalue
                alt_avgs[init,alt] = avg
                if avg > bb_avgs[init]:
                    print( f" found better: {avg} (pval={pvals[init,alt]})")

        with open("holm.pkl", "wb") as f:
            pk.dump((bb_avgs, alt_avgs, pvals), f)

    with open("holm.pkl", "rb") as f:
        (bb_avgs, alt_avgs, pvals) = pk.load(f)

    baddies = (bb_avgs <= reward_cutoff)
    print(f"{100*baddies.mean():.3f}% episodes <= {reward_cutoff} reward")

    pt.figure(figsize=(16,6))

    pt.subplot(1,2,1)
    pt.hist(bb_avgs, bins=30)
    pt.plot([reward_cutoff], [0], 'ro')
    pt.xlabel(f"average net reward ({num_rollouts} rollouts per initial state)")
    pt.ylabel("frequency")
    pt.title(f"Black box performance ({num_inits} initial states)")
    # pt.show()

    if baddies.any():
        bb_avgs = bb_avgs[baddies]
        pvals = pvals[baddies]
        alt_avgs = alt_avgs[baddies]

        # do holm correction on each baddy init
        significant = np.zeros(pvals.shape, dtype=bool)
        adjustments = np.zeros(pvals.shape)
        for init in range(len(pvals)):
            idx = np.argsort(pvals[init])
            for k, i in enumerate(idx):

                # # no correction
                # significant[init, i] = (pvals[init, i] <= sig_level)

                # # holm/-sidak correction
                # if pvals[init, i] <= sig_level / (len(idx)-k):
                if pvals[init, i] <= 1 - (1 - sig_level) ** (1/(len(idx)-k)):
                    significant[init, i] = True
                else: break

        print(f"num significant = {significant.sum()}")

        pt.subplot(1,2,2)

        # best_alts = pvals.argmin(axis=-1, keepdims=True)
        best_alts = alt_avgs.argmax(axis=-1, keepdims=True)
        performance = np.take_along_axis(alt_avgs, best_alts, axis=-1)
        sigs = np.take_along_axis(significant, best_alts, axis=-1)

        colors = np.zeros((baddies.sum(), 4))
        colors[:,3:] = np.where(sigs, 1., 0.25)
        pt.scatter(performance, bb_avgs, c=colors)

        mn = min(bb_avgs.min(), performance.min())*.99
        mx = max(bb_avgs.max(), performance.max())*1.01
        pt.plot([mn, mx], [mn, mx], 'k--')

        pt.xlim([mn, mx])
        pt.ylim([mn, mx])
        pt.xlabel(f"Best counterfactual performance (out of {num_alternatives} sampled alternatives)")
        pt.ylabel(f"Blackbox performance")
        pt.title("Black dots significant at p=0.05 (Holm-Sidak correction)")

        pt.savefig("holm.png")
        pt.show()

