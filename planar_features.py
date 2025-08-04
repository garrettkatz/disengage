"""
Feature extraction for planar environment
"""
from line_profiler import profile
import numpy as np

class FeatureExtractor:
    def __init__(self, env, kernel=None):
        assert kernel in (None, "quadratic", "cubic"), f"invalid kernel {kernel}"
        self.env = env
        self.kernel = kernel

    def get_feature_dim(self):
        obs_dim = 4*(self.env.num_allies + self.env.num_adversaries) + 1
        if self.kernel == "quadratic":
            obs_dim = obs_dim**2
        elif self.kernel == "cubic":
            obs_dim = obs_dim**3
        return obs_dim

    def trig_encode(self, agents):
        # helper for better heading features
        # period invariant; 0 similar to 2pi
        xy, angles = agents[...,:2], agents[...,2:]
        c, s = np.cos(angles), np.sin(angles)
        return np.concatenate((xy, c, s), axis=-1)

    @profile
    def __call__(self, observation):
        # assumes batched observation is (batch_size, obs_dim)
        env = self.env

        # get allies and adversaries coordinates
        split = 3*env.num_allies
        allies, adversaries = observation[..., :split], observation[..., split:]
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
        if self.kernel == "quadratic":
            features = (features[:,:,None] * features[:,None,:]).reshape(len(allies),-1)
        elif self.kernel == "cubic":
            features = (features[:,:,None,None] * features[:,None,:,None] * features[:,None,None,:]).reshape(len(allies),-1)
            # features = (features[:,None,None] * features[None,:,None] * sorted_obs).flatten()

        return features, allies_idx




