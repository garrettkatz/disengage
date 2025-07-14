"""
Approximate the non-discounted value function of handcode planar
"""
import numpy as np
import torch as tr
from time import perf_counter
from planar import PlanarEnv
from planar_features import FeatureExtractor
from handcode_planar import Policy

tr.set_default_dtype(tr.float64)

if __name__ == "__main__":

    do_train = True
    do_show = True
    resume = False

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_rollouts = 64
    num_timesteps = 256
    num_updates = 100_000

    # setup environment, policy, feature extractor
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )
    policy = Policy(env)
    get_features = FeatureExtractor(env, kernel="quadratic")

    # setup approximator and optimizer
    value_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim(), 256),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(256, 256),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(256, 256),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(256, 256),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(256, 1),
    )
    # opt = tr.optim.SGD(value_net.parameters(), lr=0.01)
    opt = tr.optim.Adam(value_net.parameters(), lr=0.00001)

    if resume:
        checkpoint = tr.load('hpv.pt', weights_only=True)
        value_net.load_state_dict(checkpoint["value_net"])
        opt.load_state_dict(checkpoint["opt"])
        loss_curve = checkpoint["loss_curve"]
    else:
        loss_curve = []

    start_time = perf_counter()

    for update in range(num_updates if do_train else 0):

        # collect some rollouts
        if True:
        # if update == 0:

            observation, _ = env.reset(batch_size=num_rollouts)
            features, _ = get_features(observation)

            net_rewards = np.zeros(num_rollouts)
            for t in range(num_timesteps):
                action = policy(observation)
                observation, reward, _, _, _ = env.step(action)
                net_rewards += reward

            # value network labels: average per-timestep reward
            value_labels = net_rewards / num_timesteps

            # tensor conversion
            value_labels = tr.tensor(value_labels[:,None]) # (batch_size, 1) to match approximator output shape
            features = tr.tensor(features)

        # pass features through value function approximator
        value_estimates = value_net(features)

        # backprop loss
        loss = tr.mean((value_estimates - value_labels)**2)
        loss_curve.append(loss.item())

        # gradient descent
        loss.backward()
        opt.step()
        opt.zero_grad()

        # progress update
        print(f"{update=} of {num_updates}, {loss=:.5f}")

    run_time = perf_counter() - start_time

    if do_show:
        import matplotlib.pyplot as pt
        bucket = len(loss_curve)//100
        trend = np.array(loss_curve).reshape(-1, bucket).mean(axis=-1)
        pt.plot(loss_curve, 'b-')
        pt.plot(np.arange(0, len(loss_curve), bucket), trend, 'k-')
        pt.yscale('log')
        pt.show()

    # save results
    if do_train:
        input(f'{num_updates} took {run_time}s. enter to save, ctrl-c to abort')
        # print(f'{num_updates} took {run_time}s')
        checkpoint = {
            "value_net": value_net.state_dict(),
            "opt": opt.state_dict(),
            "loss_curve": loss_curve,
        }
        tr.save(checkpoint, "hpv.pt")

