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

    num_state_samples = 128
    num_transit_samples = 128
    num_updates = 2000
    gamma = 0.9

    # setup environment
    num_allies = 1
    num_adversaries = 1
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

    def_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim() + (env.num_allies*3), 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 1),
        tr.nn.Sigmoid())

    parameters = list(policy_net.parameters()) + list(value_net.parameters()) + list(quad_net.parameters()) + list(def_net.parameters())
    # opt = tr.optim.SGD(parameters, lr=0.01)
    opt = tr.optim.Adam(parameters, lr=0.0001)

    if resume:
        checkpoint = tr.load('dq.pt', weights_only=True)
        policy_net.load_state_dict(checkpoint["policy_net"])
        value_net.load_state_dict(checkpoint["value_net"])
        quad_net.load_state_dict(checkpoint["quad_net"])
        def_net.load_state_dict(checkpoint["def_net"])
        opt.load_state_dict(checkpoint["opt"])
        loss_curve = checkpoint["loss_curve"]
    else:
        loss_curve = []

    for update in range(num_updates if do_train else 0):

        if True:
        # if update == 0:
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
                next_observation = env.get_observation(next_state)
                next_features.append(get_features(next_observation)[0])
            next_features = np.stack(next_features)
    
            # pass features through function approximators
            features = tr.tensor(features)
            action = tr.tensor(action)
            next_features = tr.tensor(next_features)
            reward = tr.tensor(reward)
            sorted_action = tr.take_along_dim(action, tr.tensor(allies_idx[...,None]), -2)

        M = quad_net(features)
        U = value_net(features)
        A = policy_net(features)
        next_U = value_net(next_features)
        D = def_net(tr.cat([features, sorted_action.reshape(num_state_samples, -1)], dim=-1))

        # put actions and values in correct ranges
        A = A.reshape(-1, env.num_allies, 3) * tr.tensor(step_size)
        U = U * env.get_max_reward() / (1 - gamma)
        next_U = next_U * env.get_max_reward() / (1 - gamma)

        # alternative deficit calculation
        action_gate = tr.sum((sorted_action.reshape(A.shape) - A)**2, dim=-1, keepdim=True)
        deficit = action_gate * D * 2 * env.get_max_reward() / (1 - gamma)

        # # sort policy output to match original ally order
        # inverse_idx = tr.tensor(np.argsort(allies_idx, axis=-1))
        # A = tr.take_along_dim(A, inverse_idx[...,None], -2)

        # # flatten policy output again for matrix multiply
        # A = A.reshape(-1, env.num_allies*3)

        # # put quad coefficient in correct shape and batch multiply
        # M = M.reshape(-1, env.num_allies*3, env.num_allies*3)
        # MA = (M @ (action.reshape(A.shape) - A)[...,None]).squeeze()

        # # put quad deficit term into correct range
        # deficit = tr.sum(MA**2, dim=-1, keepdim=True)
        # deficit = tr.tanh(deficit) * 2 * env.get_max_reward() / (1 - gamma)

        # calculate optimal bellman error
        Q = U - deficit
        # bellman = reward.reshape(-1, 1) + gamma * next_U.sum(dim=0)
        bellman = reward.reshape(-1, 1) + gamma * next_U.mean(dim=0)
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

    # render policy behavior
    if do_show:
        observation, info = env.reset()
        states = [env.state] # precompute intermediate states along plan
        for t in range(100):
            env.render(pt.gca(), hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")

            # run policy
            features, allies_idx = get_features(observation)
    
            # pass features through function approximator
            features = tr.tensor(features)
            A = policy_net(features[None]) # batch
    
            # put actions and values in correct ranges
            A = A.reshape(-1, env.num_allies, 3) * tr.tensor(step_size)
    
            # sort policy output to match original ally order
            inverse_idx = tr.tensor(np.argsort(allies_idx, axis=-1))
            A = tr.take_along_dim(A, inverse_idx[...,None], -2)

            action = A.detach().numpy()[0] # unbatch
            print("action:", action)
            observation, reward, _, _, _ = env.step(action)
            states.append(env.state)
            pt.pause(0.01)

    # save results
    input('enter to save, ctrl-c to abort')
    checkpoint = {
        "policy_net": policy_net.state_dict(),
        "value_net": value_net.state_dict(),
        "quad_net": quad_net.state_dict(),
        "def_net": def_net.state_dict(),
        "opt": opt.state_dict(),
        "loss_curve": loss_curve,
    }
    tr.save(checkpoint, "dq.pt")

    if do_show:
        input('[Enter] to animate...')

        import matplotlib.animation as animation
    
        fig, axs = pt.subplots(1,1)
        def drawframe(n):
            states[n].render(axs)
    
        print("animating...")
        anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=50, blit=False)
        print("saving...")
        anim.save("dq.mp4")

