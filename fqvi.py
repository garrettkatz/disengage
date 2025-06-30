"""
Fitted value iteration with discrete action space, both Q and V approximation
"""
from line_profiler import profile # python -m kernprof -lvr fqvi.py
import itertools as it
import numpy as np
import pickle as pk

@profile
def main():

    do_show = True
    do_train = False
    resume = True

    num_state_samples = 800
    num_action_samples = 100
    num_fits = 5
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

    def get_actions():
        # return np.array(list(it.product((-1,0,1), repeat=3*num_allies))).reshape(-1, num_allies, 3) * step_size
        moves = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
        ally_actions = list(m + (t,) for (m,t) in it.product(moves, (-1,0,1)))
        # ally_actions = list(m + (t,) for (m,t) in it.product(moves, (-1,1)))
        actions = np.stack([np.array(a) for a in it.product(ally_actions, repeat=num_allies)]) * step_size
        return actions

    # feature extraction
    def trig_encode(agents):
        # helper for better heading features
        # period invariant; 0 similar to 2pi
        xy, angles = agents[...,:2], agents[...,2:]
        c, s = np.cos(angles), np.sin(angles)
        return np.concatenate((xy, c, s), axis=-1)

    @profile
    def get_features(observation):

        # get allies and adversaries coordinates
        split = 3*env.num_allies
        allies, adversaries = observation[...,:split], observation[...,split:]
        allies = allies.reshape(-1, env.num_allies, 3)
        adversaries = adversaries.reshape(-1, env.num_adversaries, 3)

        # encode angles
        allies = trig_encode(allies)
        adversaries = trig_encode(adversaries)

        # sort allies and adversaries for permutation in/equivariance
        allies_idx = np.lexsort(np.transpose(allies, (2, 0, 1)), axis=1)
        adversaries_idx = np.lexsort(np.transpose(adversaries, (2, 0, 1)), axis=1)
        features = np.concatenate([
            np.take_along_axis(allies, allies_idx[:,:,None], axis=1).reshape(len(allies), env.num_allies*4),
            np.take_along_axis(adversaries, adversaries_idx[:,:,None], axis=1).reshape(len(adversaries), env.num_adversaries*4),
            np.ones((len(allies), 1)) # for bias
            ], axis=1)

        # apply kernel

        # quadratic model
        features = (features[:,:,None] * features[:,None,:]).reshape(len(allies),-1)
        # # cubic model
        # features = (features[:,None,None] * features[None,:,None] * sorted_obs).flatten()

        return features, allies_idx

    # discretized actions
    actions = get_actions()
    data_size = num_state_samples * len(actions) * num_action_samples
    # input(f"{num_state_samples} * {len(actions)} * {num_action_samples} = {data_size}...")

    # large sample for value iteration
    state = env.random_state(batch_size=num_state_samples)
    observation = env.get_observation(state)
    features, allies_idx = get_features(observation)
    reward = env.reward_function(state)

    # initialize linear critic weights
    if resume:

        with open("fqvi.pkl","rb") as f: results = pk.load(f)
        current_errors, new_errors, critic_changes, critic_V, critic_Q = results

    else:

        current_errors = []
        new_errors = []
        critic_changes = []
        critic_V = np.zeros(features.shape[1])
        critic_Q = np.zeros((features.shape[1], len(actions)))

    # input(f"(batch, feature dim) = {features.shape}...")
    print(f"(batch, feature dim) = {features.shape}...")

    # outer fit loop
    if do_train:
        for fit in range(num_fits):

            # new samples each iteration to avoid overfit
            state = env.random_state(batch_size=num_state_samples)
            observation = env.get_observation(state)
            features, allies_idx = get_features(observation)
            reward = env.reward_function(state)

            # action loop
            next_values = np.zeros((len(reward), len(actions)))
            for a, action in enumerate(actions):
                # if a % (len(actions) // 10) == 0: print(f" action {a} of {len(actions)}")

                # reorder action based on ally sort order
                reordered_action = np.empty(allies_idx.shape + (3,))
                np.put_along_axis(reordered_action, allies_idx[:,:,None], action[None], axis=1)

                # accumulate value estimates over next state samples
                for n in range(num_action_samples):

                    # get next state features
                    next_state = env.transition(state, reordered_action)
                    next_observation = env.get_observation(next_state)
                    next_features, _ = get_features(next_observation)

                    # get max next-state Q value
                    # input(f"{next_features.shape} @ {critic_V.shape}")
                    val = (next_features @ critic_V)

                    # update average
                    next_values[:,a] += val / num_action_samples

            # compute error of current critic
            bellman_V = reward + gamma * next_values.max(axis=1)
            bellman_Q = reward[:,None] + gamma * next_values
            
            # current_error = np.fabs(features @ critic_V - bellman_V).mean()
            current_error = ((features @ critic_V - bellman_V)**2).mean()
            current_errors.append(current_error)

            # fit new critics
            new_critic_V = np.linalg.lstsq(features, bellman_V, rcond=None)[0]
            new_critic_Q = np.linalg.lstsq(features, bellman_Q, rcond=None)[0]
            # new_error = np.fabs(features @ new_critic_V - bellman_V).mean()
            new_error = ((features @ new_critic_V - bellman_V)**2).mean()
            new_errors.append(new_error)

            # critic change
            critic_change = np.fabs(new_critic_V - critic_V).mean()
            critic_changes.append(critic_change)

            # update critic
            critic_V = new_critic_V
            critic_Q = new_critic_Q

            print(f"fit {fit} of {num_fits}, {current_error=:.5f}, {new_error=:.5f}, {critic_change=:.5f}")

        results = (current_errors, new_errors, critic_changes, critic_V, critic_Q)
        with open("fqvi.pkl","wb") as f: pk.dump(results, f)

    if do_show:

        print(critic_Q)
        print(np.fabs(critic_Q).max())
        print(critic_V)
        print(np.fabs(critic_V).max())

        import matplotlib.pyplot as pt
        pt.subplot(1,4,1)
        pt.plot(current_errors)
        pt.subplot(1,4,2)
        pt.plot(new_errors)
        pt.subplot(1,4,3)
        pt.plot(critic_changes)
        pt.subplot(1,4,4)
        pt.imshow(critic_Q)
        pt.show()
    
        # animate agent
        input("[Enter] to run agent")
        pt.ion()
        pt.figure()
    
        observation, info = env.reset()
        states = [env.state] # precompute intermediate states along plan
        for t in range(500):
            env.render(pt.gca(), hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")
    
            features, allies_idx = get_features(observation)
            action = actions[(features @ critic_Q).argmax()]
            reordered_action = np.empty(allies_idx.shape + (3,))
            np.put_along_axis(reordered_action, allies_idx[:,:,None], action[None], axis=1)
    
            observation, reward, _, _, _ = env.step(reordered_action[0])
            states.append(env.state)
            pt.pause(0.01)

        input('[Enter] to animate...')

        import matplotlib.animation as animation
    
        fig, axs = pt.subplots(1,1)
        def drawframe(n):
            states[n].render(axs)
    
        print("animating...")
        anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=50, blit=False)
        print("saving...")
        anim.save("fqvi.mp4")

if __name__ == "__main__": main()


