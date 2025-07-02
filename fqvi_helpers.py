from line_profiler import profile
import numpy as np

class Policy:
    def __init__(self, get_features, actions, critic_Q):
        self.get_features = get_features
        self.actions = actions
        self.critic_Q = critic_Q

    @profile
    def __call__(self, observation):
        features, allies_idx = self.get_features(observation)
        values = (features @ self.critic_Q) # [..., num_actions]
        action = self.actions[values.argmax(axis=-1)]
        # reordered_action = np.empty(allies_idx.shape + (3,))
        reordered_action = np.empty(action.shape)
        # np.put_along_axis(reordered_action, allies_idx[:,:,None], action[None], axis=1)
        # return reordered_action[0]
        np.put_along_axis(reordered_action, allies_idx[...,None], action, axis=-2)
        return reordered_action

class FeatureExtractor:
    def __init__(self, env):
        self.env = env

    def trig_encode(self, agents):
        # helper for better heading features
        # period invariant; 0 similar to 2pi
        xy, angles = agents[...,:2], agents[...,2:]
        c, s = np.cos(angles), np.sin(angles)
        return np.concatenate((xy, c, s), axis=-1)

    @profile
    def __call__(self, observation):
        env = self.env

        # get allies and adversaries coordinates
        split = 3*env.num_allies
        allies, adversaries = observation[...,:split], observation[...,split:]
        allies = allies.reshape(-1, env.num_allies, 3)
        adversaries = adversaries.reshape(-1, env.num_adversaries, 3)

        # encode angles
        allies = self.trig_encode(allies)
        adversaries = self.trig_encode(adversaries)

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


