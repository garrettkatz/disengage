"""
Vanilla Policy Gradient implementation
"""
from collections import namedtuple
import numpy as np
import torch as tr

TrainConfig = namedtuple("TrainConfig", [
    "lr", "updates", "episodes", "timesteps",
])

def train(env, actor, config):

    # training setup
    opt = tr.optim.SGD(actor.parameters(), lr=config.lr)

    # training loop
    reward_curve = []
    grad_curve = []
    # prob_sat_curve = []
    for update in range(config.updates):

        # collect batch
        net_rewards = []
        # mean_prob_sat = []
        for episode in range(config.episodes):
            rewards, log_probs = [], []

            # run episode
            observation, info = env.reset()
            for t in range(config.timesteps):
                action, log_prob = actor(observation)
                observation, reward, _, _, _ = env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)

            # rewards to go
            returns = tr.cumsum(tr.tensor(rewards, dtype=tr.float32), 0)
            net_reward = returns[-1]
            net_rewards.append(net_reward.item())

            # policy gradient
            log_probs = tr.stack(log_probs)
            loss = - net_reward * tr.sum(log_probs) / (config.timesteps * config.episodes)
            loss.backward()    

            # probs = tr.exp(log_probs)
            # prob_sat = tr.minimum(probs, 1 - probs)            
            # mean_prob_sat.append(prob_sat.mean().item())

        # prob_sat_curve.append(np.mean(mean_prob_sat))

        grad_norm = sum([tr.sum((p.grad)**2) for p in actor.parameters()])**.5
        grad_curve.append(grad_norm)

        # gradient update
        opt.step()
        opt.zero_grad()

        reward_curve.append(np.mean(net_rewards))
        print(f"update {update} of {config.updates}: |grad| = {grad_norm}, mean net reward {reward_curve[-1]}")

    return reward_curve, grad_curve
    # return reward_curve, grad_curve, prob_sat_curve

if __name__ == "__main__":

    step_size = 0.05
    do_train = True

    class Actor(tr.nn.Module):

        def __init__(self, env):
            super().__init__()
            self.env = env
            # linear model
            obs_dim = 3*(env.num_allies + env.num_adversaries) + 1 # +1 for bias
            act_dim = 2*3*env.num_allies # 2* for beta distribution
            # self.lin = tr.nn.Linear(obs_dim, act_dim, bias=False)
            self.quad = tr.nn.Linear(obs_dim**2, act_dim, bias=False)

        def forward(self, observation):
            # get allies and adversaries coordinates
            split = 3*self.env.num_allies
            allies, adversaries = observation[:split], observation[split:]
            allies = allies.reshape(-1, 3)
            adversaries = adversaries.reshape(-1, 3)
            
            # sort for permutation invariance
            allies_idx = np.lexsort(allies.T)
            adversaries_idx = np.lexsort(adversaries.T)
            sorted_obs = np.concatenate([
                allies[allies_idx].flatten(),
                adversaries[adversaries_idx].flatten(),
                np.ones(1)]) # for bias

            # # linear model
            # out = .001 + 5 * tr.sigmoid(self.lin(tr.tensor(sorted_obs, dtype=tr.float32)))

            # quadratic model
            features = (sorted_obs[:,None] * sorted_obs).flatten()
            out = .001 + 5 * tr.sigmoid(self.quad(tr.tensor(features, dtype=tr.float32)))

            out = out.reshape(2, -1, 3)
            alpha, beta = out

            # stochastic policy
            dist = tr.distributions.beta.Beta(alpha, beta)
            sorted_action = dist.sample()
            log_prob = dist.log_prob(sorted_action).sum()

            # unsort action
            action = np.empty(sorted_action.shape)
            for k, idx in enumerate(allies_idx):
                action[idx] = sorted_action[k]

            # scale action to environment's action space
            action = step_size * (2 * action - 1)            

            return action, log_prob

    config = TrainConfig(
        lr=0.0001,
        updates=100,
        episodes=100,
        timesteps=100,
    )

    from planar import PlanarEnv
    env = PlanarEnv(
        num_allies=1,
        num_adversaries=1,
        step_size=step_size,
        view_angle=np.pi/8,
    )

    actor = Actor(env)

    import pickle as pk

    if do_train:

        results = train(env, actor, config)
        with open("vpg.pkl","wb") as f: pk.dump(results, f)

    with open("vpg.pkl","rb") as f: results = pk.load(f)
    reward_curve, grad_curve = results

    import matplotlib.pyplot as pt

    buck = 100
    pt.subplot(1,2,1)
    pt.plot(reward_curve)
    pt.plot(np.arange(0, len(reward_curve), buck), np.array(reward_curve).reshape(-1,buck).mean(axis=1), 'ro-')
    pt.subplot(1,2,2)
    pt.plot(grad_curve)
    pt.show()

