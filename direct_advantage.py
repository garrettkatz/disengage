"""
Directly fit the optimal Q Bellman equation (no iteration) with utility and non-negative advantage approximations
"""
from line_profiler import profile # python -m kernprof -lvr direct_advantage.py
import numpy as np
import torch as tr
from planar_features import FeatureExtractor
from time import perf_counter

tr.set_default_dtype(tr.float64)

@profile
def main():

    do_show = True
    do_train = True
    do_render = False
    resume = True

    num_state_samples = 32
    num_transit_samples = 8
    num_updates = 100_000
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

    value_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim(), 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 1),
        tr.nn.Tanh())

    advantage_net = tr.nn.Sequential(
        tr.nn.Linear(get_features.get_feature_dim() + env.num_allies*3, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 1),
        tr.nn.Sigmoid())

    parameters = list(value_net.parameters()) + list(advantage_net.parameters())
    # opt = tr.optim.SGD(parameters, lr=0.01)
    opt = tr.optim.Adam(parameters, lr=0.0001)

    if resume:
        checkpoint = tr.load('da.pt', weights_only=True)
        value_net.load_state_dict(checkpoint["value_net"])
        advantage_net.load_state_dict(checkpoint["advantage_net"])
        opt.load_state_dict(checkpoint["opt"])
        loss_curve = checkpoint["loss_curve"]
    else:
        loss_curve = []

    start_time = perf_counter()

    for update in range(num_updates if do_train else 0):

        if True:
        # if update == 0:

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

        U = value_net(features)
        next_U = value_net(next_features)
        deficit = advantage_net(tr.cat([features, sorted_action.reshape(num_state_samples, -1)], dim=-1))

        # put values in correct ranges
        U = U * env.get_max_reward() / (1 - gamma)
        next_U = next_U * env.get_max_reward() / (1 - gamma)
        deficit = deficit * 2 * env.get_max_reward() / (1 - gamma)

        # calculate optimal bellman error
        Q = U - deficit
        bellman = reward.reshape(-1, 1) + gamma * next_U.mean(dim=0)
        loss = tr.mean((Q - bellman)**2)
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
            "advantage_net": advantage_net.state_dict(),
            "opt": opt.state_dict(),
            "loss_curve": loss_curve,
        }
        tr.save(checkpoint, "da.pt")

    # render policy behavior
    if do_render:
        advantage_net.eval()
        observation, info = env.reset()
        states = [env.state] # precompute intermediate states along plan
        
        _, render_ax = pt.subplots(1,1)
        _, deficit_ax = pt.subplots(1,1)
        
        for t in range(100):
            env.render(render_ax, hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")

            # optimize action through advantage function
            features, allies_idx = get_features(observation[None])
            features = tr.tensor(features) # batched for policy
            sorted_action = tr.zeros(env.num_allies, 3, requires_grad=True)
            action_opt = tr.optim.Adam([sorted_action], lr=.1)

            deficit_curve = []
            for action_update in range(100):
                deficit = advantage_net(tr.cat([features, sorted_action.reshape(1, -1)], dim=-1))
                deficit_curve.append(deficit.item())

                deficit[0,0].backward() # unbatched
                action_opt.step()
                action_opt.zero_grad()
                sorted_action.data = sorted_action.data.clamp(-tr.tensor(step_size), tr.tensor(step_size))
                print(sorted_action.data.numpy())

                if (tr.abs(sorted_action.data) == tr.tensor(step_size)).all(): break

            # deficit_ax.clear()
            # deficit_ax.plot(deficit_curve)

            # unsort action to match original ally order
            inverse_idx = tr.tensor(np.argsort(allies_idx[0], axis=-1)) # unbatched
            action = tr.take_along_dim(sorted_action, inverse_idx[...,None], -2)

            # execute action
            action = action.detach().numpy()
            observation, reward, _, _, _ = env.step(action)
            states.append(env.state)
            pt.pause(0.01)

            # input(f"{action_update} action updates: deficit, action = {deficit.item()}, {action}")

    #     input('[Enter] to animate...')

    #     import matplotlib.animation as animation
    
    #     fig, axs = pt.subplots(1,1)
    #     def drawframe(n):
    #         states[n].render(axs)
    
    #     print("animating...")
    #     anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=50, blit=False)
    #     print("saving...")
    #     anim.save("da.mp4")


if __name__ == "__main__": main()

