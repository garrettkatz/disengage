from line_profiler import profile # python -m kernprof -lvr random_search.py
from collections import namedtuple
import numpy as np
from scipy.stats import ttest_ind_from_stats

TrainConfig = namedtuple("TrainConfig", [
    "stdev", "samples", "episodes", "timesteps",
])

@profile
def run_episodes(env, actor, config):

    # collect batched episodes
    rewards = np.empty((config.timesteps, config.episodes))
    observation, _ = env.reset(batch_size=config.episodes)
    for t in range(config.timesteps):

        # # need to batch actor
        # action = np.stack([actor(obs) for obs in observation])
        # batched actor
        action = actor(observation)

        observation, reward, _, _, _ = env.step(action)
        rewards[t] = reward

    net_rewards = rewards.mean(axis=0)

    # sample stats
    avg = np.mean(net_rewards)
    se = np.std(net_rewards, ddof=1)
    nobs = len(net_rewards)
    return avg, se, nobs

    # # performance metric
    # avg_net_reward = np.mean(net_rewards)
    # std_net_reward = np.std(net_rewards)
    # # perf = avg_net_reward - std_net_reward # maximize typical "worst" case
    # perf = avg_net_reward
    # margin = std_net_reward / config.episodes**.5
    # return perf, margin

@profile
def train(env, actor, config):
    """
    modifies actor params in-place
    """

    best_params = actor.params
    # best_perf, best_margin = run_episodes(env, actor, config)
    best_avg, best_se, nobs = run_episodes(env, actor, config)

    # save best parameters and rewards
    all_perfs, best_perfs = [], []

    for sample in range(config.samples):

        # perturb actor
        params = np.random.normal(best_params, config.stdev)
        actor.set_parameters(params)

        # collect episode rewards
        # perf, margin = run_episodes(env, actor, config)
        avg, se, _ = run_episodes(env, actor, config)

        # ttest
        ttest_res = ttest_ind_from_stats(
            mean1=avg, std1=se, nobs1=nobs,
            mean2=best_avg, std2=best_se, nobs2=nobs,
            equal_var=False, alternative="greater")

        # print(avg, se, best_avg, best_se, ttest_res.statistic, ttest_res.pvalue)

        # statistically significant boost in mean
        if ttest_res.statistic > 0 and ttest_res.pvalue < 0.05:
        # # conservative, require high likelihood of improvement
        # if perf - margin > best_perf + best_margin:
        # # liberal, change at small chance of improvement
        # if perf > best_perf:
            best_avg, best_se, best_params = avg, se, params
            # best_perf, best_margin, best_params = perf, margin, params

        # update curves
        all_perfs.append(avg)
        best_perfs.append(best_avg)

        # progress update
        print(f"sample {sample} of {config.samples}: {avg:.3f} +/- {se:.3f} vs [best = {best_avg:.3f} +/- {best_se:.3f}] ||w|| ~ {np.linalg.norm(best_params):.3f}")

    # wrap up
    actor.set_parameters(best_params)
    return all_perfs, best_perfs

if __name__ == "__main__":

    # step_size = 0.05
    num_allies = 1
    num_adversaries = 1
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion
    do_train = True
    resume = True

    @profile
    def trig_encode(agents):
        # helper for better heading features
        # period invariant; 0 similar to 2pi
        xy, angles = agents[...,:2], agents[...,2:]
        c, s = np.cos(angles), np.sin(angles)
        return np.concatenate((xy, c, s), axis=-1)

    class Actor:

        def __init__(self, env):
            self.env = env
            self.obs_dim = 4*(env.num_allies + env.num_adversaries) + 1 # 4* for trig encoding, +1 for bias
            self.act_dim = 3*env.num_allies
            # # linear model
            # self.params = np.zeros((self.act_dim, self.obs_dim))
            # quadratic model
            self.params = np.zeros((self.act_dim, self.obs_dim**2))

        def set_parameters(self, params):
            assert self.params.shape == params.shape
            self.params = params

        def __call__(self, observation):
            # get allies and adversaries coordinates
            split = 3*self.env.num_allies
            allies, adversaries = observation[:split], observation[split:]
            allies = allies.reshape(-1, 3)
            adversaries = adversaries.reshape(-1, 3)

            # encode angles
            allies = trig_encode(allies)
            adversaries = trig_encode(adversaries)

            # sort for permutation invariance
            allies_idx = np.lexsort(allies.T)
            adversaries_idx = np.lexsort(adversaries.T)
            sorted_obs = np.concatenate([
                allies[allies_idx].flatten(),
                adversaries[adversaries_idx].flatten(),
                np.ones(1)]) # for bias

            # # linear model
            # features = sorted_obs.flatten()
            # # quadratic model
            # features = (sorted_obs[:,None] * sorted_obs).flatten()
            # cubic model
            features = (sorted_obs[:,None,None] * sorted_obs[None,:,None] * sorted_obs).flatten()

            # constrain output to valid action space
            out = np.tanh(self.params @ features)

            # reshape and unsort action
            sorted_action = step_size * out.reshape(env.num_allies, 3)
            action = np.empty(sorted_action.shape)
            for k, idx in enumerate(allies_idx):
                action[idx] = sorted_action[k]

            return action

    class WeightSharedActor:

        def __init__(self, env):
            self.env = env
            self.obs_dim = 4*(env.num_allies + env.num_adversaries) + 1 # 4* for trig encoding, +1 for bias
            self.act_dim = 3 # 3 dims per agent, all use same params
            # linear model
            self.params = np.zeros((self.act_dim, self.obs_dim))
            # # quadratic model
            # self.params = np.zeros((self.act_dim, self.obs_dim**2))
            # # cubic model
            # self.params = np.zeros((self.act_dim, self.obs_dim**3))

        def set_parameters(self, params):
            assert self.params.shape == params.shape
            self.params = params

        @profile
        def __call__(self, observation):

            batched = len(observation.shape) > 1

            # get allies and adversaries coordinates
            split = 3*self.env.num_allies
            allies, adversaries = observation[...,:split], observation[...,split:]
            allies = allies.reshape(-1, self.env.num_allies, 3)
            adversaries = adversaries.reshape(-1, self.env.num_adversaries, 3)

            # encode angles
            allies = trig_encode(allies)
            adversaries = trig_encode(adversaries)

            # do allies one at a time
            action = []
            for a in range(self.env.num_allies):

                # sort other agents for permutation invariance
                other_allies = allies[:, np.arange(self.env.num_allies) != a, :]
                allies_idx = np.lexsort(np.transpose(other_allies, (2, 0, 1)), axis=1)
                adversaries_idx = np.lexsort(np.transpose(adversaries, (2, 0, 1)), axis=1)
                sorted_obs = np.concatenate([
                    allies[:,a],
                    np.take_along_axis(other_allies, allies_idx[:,:,None], axis=1).reshape(len(allies), (self.env.num_allies-1)*4),
                    np.take_along_axis(adversaries, adversaries_idx[:,:,None], axis=1).reshape(-1, self.env.num_adversaries*4),
                    np.ones((len(allies), 1)) # for bias
                    ], axis=1)

                # # don't bother sorting right now
                # other_allies = allies[:, np.arange(self.env.num_allies) != a, :]
                # sorted_obs = np.concatenate([
                #     allies[:,a],
                #     other_allies.reshape(-1, (self.env.num_allies-1)*4),
                #     adversaries.reshape(-1, self.env.num_adversaries*4),
                #     np.ones((len(allies), 1)) # for bias
                #     ], axis=1)
    
                # linear model
                features = sorted_obs.reshape(len(allies), -1)
                # # quadratic model
                # features = (sorted_obs[:,:,None] * sorted_obs[:,None,:]).reshape(len(allies),-1)
                # # cubic model
                # features = (sorted_obs[:,None,None] * sorted_obs[None,:,None] * sorted_obs).flatten()
    
                # constrain output to valid action space
                out = step_size * np.tanh(self.params[None,:,:] @ features[:,:,None])[:,:,0]
                action.append(out)

            action = np.stack(action, axis=1)
            if not batched: action = action[0]

            return action

    from planar import PlanarEnv
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )

    # actor = Actor(env)
    actor = WeightSharedActor(env)

    config = TrainConfig(
        stdev=0.1,
        samples=5000,
        episodes=1000,
        timesteps=100,
    )

    import pickle as pk

    if do_train:

        if resume:
            with open("rs.pkl","rb") as f: results = pk.load(f)
            (all_perfs, best_perfs), params = results
            actor.set_parameters(params)

        curves = train(env, actor, config)

        if resume:
            all_new, best_new = curves
            all_perfs += all_new
            best_perfs += best_new
            curves = (all_perfs, best_perfs)

        with open("rs.pkl","wb") as f: pk.dump((curves, actor.params), f)

    with open("rs.pkl","rb") as f: results = pk.load(f)
    (all_perfs, best_perfs), params = results
    actor.set_parameters(params)

    import matplotlib.pyplot as pt

    pt.ion()

    buck = len(all_perfs)//10
    # pt.subplot(1,2,1)
    pt.plot(all_perfs, 'b-')
    pt.plot(np.arange(0, len(all_perfs), buck), np.array(all_perfs).reshape(-1,buck).mean(axis=1), 'ro-')
    pt.plot(best_perfs, 'g+-')
    # pt.subplot(1,2,2)
    # pt.plot(grad_curve)
    pt.show()

    input("[Enter] to run agent")
    pt.figure()

    observation, info = env.reset()
    states = [env.state] # precompute intermediate states along plan
    for t in range(200):
        env.render(pt.gca(), hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")
        action = actor(observation)
        observation, reward, _, _, _ = env.step(action)
        states.append(env.state)
        pt.pause(0.01)
    input('[Enter] to animate...')

    ## animation
    import matplotlib.animation as animation

    fig, axs = pt.subplots(1,1)
    def drawframe(n):
        states[n].render(axs)

    print("animating...")
    anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=50, blit=False)
    print("saving...")
    anim.save("rs.mp4")



