"""
Fitted value iteration with discrete Q action space
"""
from line_profiler import profile # python -m kernprof -lvr fvi.py
import itertools as it
import numpy as np
import pickle as pk

@profile
def main():

    do_train = True
    resume = False

    num_state_samples = 500
    num_action_samples = 500
    num_fits = 50
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

    # feature extraction
    def trig_encode(agents):
        # helper for better heading features
        # period invariant; 0 similar to 2pi
        xy, angles = agents[...,:2], agents[...,2:]
        c, s = np.cos(angles), np.sin(angles)
        return np.concatenate((xy, c, s), axis=-1)

    def get_features(observation):

        # get allies and adversaries coordinates
        split = 3*env.num_allies
        allies, adversaries = observation[...,:split], observation[...,split:]
        allies = allies.reshape(-1, env.num_allies, 3)
        adversaries = adversaries.reshape(-1, env.num_adversaries, 3)

        # encode angles
        allies = trig_encode(allies)
        adversaries = trig_encode(adversaries)

        # reconcatenate
        features = np.concatenate([
            allies.reshape(-1, env.num_allies*4),
            adversaries.reshape(-1, env.num_adversaries*4),
            np.ones((len(allies), 1)), # bias
        ], axis=1)

        # apply kernel

        # quadratic model
        features = (features[:,:,None] * features[:,None,:]).reshape(len(allies),-1)
        # # cubic model
        # features = (features[:,None,None] * features[None,:,None] * sorted_obs).flatten()

        return features

    # discretized actions
    actions = np.array(list(it.product((-1,0,1), repeat=3*num_allies))).reshape(-1, num_allies, 3) * step_size
    data_size = num_state_samples * len(actions) * num_action_samples
    # input(f"{num_state_samples} * {len(actions)} * {num_action_samples} = {data_size}...")

    # large sample for value iteration
    state = env.random_state(batch_size=num_state_samples)
    observation = env.get_observation(state)
    features = get_features(observation)
    reward = env.reward_function(state)

    # initialize linear critic weights
    if resume:

        with open("fvi.pkl","rb") as f: results = pk.load(f)
        current_errors, new_errors, critic_changes, critic = results

    else:

        critic = np.zeros((features.shape[1], len(actions)))
        current_errors = []
        new_errors = []
        critic_changes = []

    input(f"(batch, feature dim) = {features.shape}...")

    # outer fit loop
    if do_train:
        for fit in range(num_fits):
    
            # action loop
            next_values = np.zeros((len(reward), len(actions)))
            for a, action in enumerate(actions): 
    
                # accumulate value estimates over next state samples
                for n in range(num_action_samples):
    
                    # get next state features
                    next_state = env.transition(state, action)
                    next_observation = env.get_observation(next_state)
                    next_features = get_features(next_observation)
    
                    # get max next-state Q value
                    val = (next_features @ critic).max(axis=1)
    
                    # update average
                    next_values[:,a] += val / num_action_samples
    
            # compute error of current critic
            bellman = reward[:,None] + gamma * next_values
            # current_error = np.fabs(features @ critic - bellman).mean()
            current_error = ((features @ critic - bellman)**2).mean()
            current_errors.append(current_error)
    
            # fit new critic
            new_critic = np.linalg.lstsq(features, bellman, rcond=None)[0]
            # new_error = np.fabs(features @ new_critic - bellman).mean()
            new_error = ((features @ new_critic - bellman)**2).mean()
            new_errors.append(new_error)
    
            # critic change
            critic_change = np.fabs(new_critic - critic).mean()
            critic_changes.append(critic_change)
    
            # update critic
            critic = new_critic
    
            print(f"fit {fit} of {num_fits}, {current_error=:.5f}, {new_error=:.5f}, {critic_change=:.5f}")

        results = (current_errors, new_errors, critic_changes, critic)
        with open("fvi.pkl","wb") as f: pk.dump(results, f)

    import matplotlib.pyplot as pt
    pt.subplot(1,3,1)
    pt.plot(current_errors)
    pt.subplot(1,3,2)
    pt.plot(new_errors)
    pt.subplot(1,3,3)
    pt.plot(critic_changes)
    pt.show()

    # animate agent
    input("[Enter] to run agent")
    pt.ion()
    pt.figure()

    observation, info = env.reset()
    states = [env.state] # precompute intermediate states along plan
    for t in range(100):
        env.render(pt.gca(), hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")

        features = get_features(observation)
        action = actions[(features @ critic).argmax()]

        observation, reward, _, _, _ = env.step(action)
        states.append(env.state)
        pt.pause(0.01)
    input('[Enter] to animate...')

if __name__ == "__main__": main()
