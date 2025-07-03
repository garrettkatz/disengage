"""
Directly fit the optimal Q Bellman equation (no iteration) with state-dependent quadratic approximations
"""
from line_profiler import profile # python -m kernprof -lvr direct_quad.py
import numpy as np
import torch as tr
from planar_features import FeatureExtractor

tr.set_default_dtype(tr.float64)

if __name__ == "__main__":

    do_show = True
    do_train = True
    resume = False

    num_state_samples = 64
    num_transit_samples = 32
    num_updates = 1000
    gamma = 0.9

    # setup environment
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    from planar import PlanarEnv
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )

    get_features = FeatureExtractor(env)

    policy_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim(), 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, env.num_allies*3),
        tr.nn.Tanh())

    value_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim(), 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 1),
        tr.nn.Tanh())

    quad_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim(), 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, (env.num_allies*3)**2))

    parameters = list(policy_net.parameters()) + list(value_net.parameters()) + list(quad_net.parameters())
    # opt = tr.optim.SGD(parameters, lr=0.01)
    opt = tr.optim.Adam(parameters, lr=0.005)

    loss_curve = []
    for update in range(num_updates if do_train else 0):

        # sample some states and actions
        state = env.random_state(batch_size=num_state_samples)
        observation = env.get_observation(state)
        features, allies_idx = get_features(observation)
        reward = env.reward_function(state)
        action = env.action_space.sample(batch_size=num_state_samples)

        # accumulate transit samples
        next_features = []
        for transit in range(num_transit_samples):
            next_state = env.transition(state, action)
            next_observation = env.get_observation(state)
            next_features.append(get_features(next_observation)[0])
        next_features = np.stack(next_features)

        # sort original actions to match policy output order
        action = action[allies_idx]...?

        # pass features through function approximators
        features = tr.tensor(features)
        action = tr.tensor(action)
        next_features = tr.tensor(next_features)
        reward = tr.tensor(reward)

        M = quad_net(features)
        U = value_net(features)
        A = policy_net(features)
        next_U = value_net(next_features)

        # put actions and values in correct ranges
        A = (A.reshape(-1, env.num_allies, 3) * tr.tensor(step_size)).reshape(-1, env.num_allies*3)
        U = U * env.get_max_reward()
        next_U = next_U * env.get_max_reward()

        # put quad coefficient in correct shape and batch multiply
        M = M.reshape(-1, env.num_allies*3, env.num_allies*3)
        MA = (M @ (action.reshape(A.shape) - A)[...,None]).squeeze()

        # calculate optimal bellman error
        Q = U - tr.sum(MA**2, dim=-1, keepdim=True)
        bellman = reward.reshape(-1, 1) + gamma * next_U.sum(dim=0)
        loss = tr.mean((Q - bellman)**2)
        loss_curve.append(loss.item())

        # gradient descent
        loss.backward()
        opt.step()
        opt.zero_grad()

        # progress update
        print(f"{update=} of {num_updates}, {loss=:.5f}")

    import matplotlib.pyplot as pt
    pt.plot(loss_curve)
    pt.yscale('log')
    pt.show()

