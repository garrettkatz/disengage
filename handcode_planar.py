"""
Hand-coded heuristic policy for planar env
- Each agent points towards the adversary nearest to its line of sight
- Each agent moves away from the closest adversary line of sight
"""
from planar import PlanarEnv
import planar_features as pf
import numpy as np

class Policy:
    def __init__(self, env):
        self.env = env

    def __call__(self, observation):
        split = 3*self.env.num_allies
        allies, adversaries = observation[..., :split], observation[..., split:]
        allies = allies.reshape(-1, env.num_allies, 3)
        adversaries = adversaries.reshape(-1, env.num_adversaries, 3)

        ally_xy = allies[...,:2]
        ally_cs = np.stack([np.cos(allies[...,2]), np.sin(allies[...,2])], axis=-1)
        ally_90 = np.stack([-np.sin(allies[...,2]), np.cos(allies[...,2])], axis=-1)
        advr_xy = adversaries[...,:2]
        advr_cs = np.stack([np.cos(adversaries[...,2]), np.sin(adversaries[...,2])], axis=-1)

        # get adversary closest (rotationally) to line of sight
        print(advr_xy.shape)
        diffs = advr_xy[...,None,:,:] - ally_xy[...,None,:]
        diffs /= np.linalg.norm(diffs, axis=-1)
        coss = (diffs * ally_cs).sum(axis=-1)
        closest = np.argmax(coss, axis=-1, keepdims=True)

        # get its angle relative to ally
        coss = np.take_along_axis(coss, closest, axis=-2)
        coss = coss[...,0,:]
        diffs = np.take_along_axis(diffs, closest[...,None], axis=-2)
        diffs = diffs[...,0,:]
        print(diffs.shape)
        sins = (diffs * ally_90).sum(axis=-1)
        angle = np.arctan2(sins, coss)

        action = np.zeros((env.num_allies, 3))
        action[...,2] = angle
        action = np.clip(action, -env.step_size, env.step_size)

        return action        


if __name__ == "__main__":

    # setup environment
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )

    policy = Policy(env)

    observation, info = env.reset()
    states = [env.state] # precompute intermediate states along plan

    import matplotlib.pyplot as pt
    _, render_ax = pt.subplots(1,1)

    for t in range(100):
        env.render(render_ax, hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")
        action = policy(observation)
        observation, reward, _, _, _ = env.step(action)
        states.append(env.state)
        pt.pause(0.01)

