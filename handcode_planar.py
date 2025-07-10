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
        allies = allies.reshape(-1, self.env.num_allies, 3)
        adversaries = adversaries.reshape(-1, self.env.num_adversaries, 3)
        action = np.zeros((observation.shape[0], self.env.num_allies, 3))

        ## keep closest adversary in field of view
        ally_xy = allies[...,:2]
        ally_cs = np.stack([np.cos(allies[...,2]), np.sin(allies[...,2])], axis=-1)
        ally_90 = np.stack([-np.sin(allies[...,2]), np.cos(allies[...,2])], axis=-1)
        advr_xy = adversaries[...,:2]
        advr_cs = np.stack([np.cos(adversaries[...,2]), np.sin(adversaries[...,2])], axis=-1)
        advr_90 = np.stack([-np.sin(adversaries[...,2]), np.cos(adversaries[...,2])], axis=-1)

        diffs = advr_xy[...,None,:,:] - ally_xy[...,None,:]
        diffs /= np.maximum(np.linalg.norm(diffs, axis=-1, keepdims=True), 1e-6)
        coss = (diffs * ally_cs[...,None,:]).sum(axis=-1)
        sins = (diffs * ally_90[...,None,:]).sum(axis=-1)
        angles = np.arctan2(sins, coss)
        closest = np.argmin(np.fabs(angles), axis=-1, keepdims=True)
        angle = np.take_along_axis(angles, closest, axis=-1)

        action[...,2:] = angle

        ## move in the opposite direction of adversary's field of view
        ## HACK! find closest here too
        # action[...,:2] = -advr_cs.mean(axis=-2,keepdims=True)

        # move away from closest adverary line of sight
        diffs = ally_xy[...,None,:,:] - advr_xy[...,None,:]
        diffs /= np.maximum(np.linalg.norm(diffs, axis=-1, keepdims=True), 1e-6)
        coss = (diffs * advr_cs[...,None,:]).sum(axis=-1)
        sins = (diffs * advr_90[...,None,:]).sum(axis=-1)
        angles = np.arctan2(sins, coss)
        closest = np.argmin(np.fabs(angles), axis=-2)
        angle = np.take_along_axis(angles, closest[...,None,:], axis=-2)
        delta = np.take_along_axis(advr_90, closest[...,None], axis=-2)
        delta *= np.sign(angle)
        action[...,:2] = delta

        action = np.clip(action, -self.env.step_size, self.env.step_size)
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

    rewards = []
    for t in range(100):
        env.render(render_ax, hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")
        action = policy(observation)[0]
        observation, reward, _, _, _ = env.step(action)
        rewards.append(reward)
        states.append(env.state)
        pt.pause(0.01)

    print(f"mean reward = {np.mean(reward)} vs {env.get_max_reward()}")

    pt.figure()
    pt.plot(np.arange(100), np.cumsum(rewards), 'b-')
    pt.plot(np.arange(100), np.arange(100)*env.get_max_reward(), 'g--')
    pt.show()

    # repeat 30 times for envelope comparison
    num_timesteps = 200
    num_episodes = 30
    observation, info = env.reset(batch_size=num_episodes)
    rewards = np.empty((num_episodes, num_timesteps))
    for t in range(num_timesteps):
        action = policy(observation)
        observation, reward, _, _, _ = env.step(action)
        rewards[:,t] = reward

    import pickle as pk
    with open("envelope_best.pkl","rb") as f:
        (envelope_rewards, better_rewards, _, _, _, _) = pk.load(f)

    # render envelope
    # pt.subplot(1,2,1)
    pt.plot(np.arange(num_timesteps), np.arange(1, num_timesteps+1)*env.get_max_reward(), 'k--', label="theoretically optimal")

    cumulative_rewards = envelope_rewards.cumsum(axis=1)
    pt.fill_between(np.arange(num_timesteps), cumulative_rewards.min(axis=0), cumulative_rewards.max(axis=0), alpha=.1, color='r')
    pt.plot(cumulative_rewards.mean(axis=0), 'r-', label="black-box-old")

    better_cumulative_rewards = better_rewards.cumsum(axis=1)
    pt.fill_between(np.arange(num_timesteps), better_cumulative_rewards.min(axis=0), better_cumulative_rewards.max(axis=0), alpha=.1, color='b')

    cumulative_rewards = rewards.cumsum(axis=1)
    pt.fill_between(np.arange(num_timesteps), cumulative_rewards.min(axis=0), cumulative_rewards.max(axis=0), alpha=.1, color='g')
    pt.plot(cumulative_rewards.mean(axis=0), 'g-', label="black-box-new")

    pt.plot(better_cumulative_rewards.mean(axis=0), 'b-', label="counterfactual-old")
    pt.xlabel("Time-step")
    pt.ylabel("Cumulative Reward")
    pt.title(f"Performance mean and envelope ({num_episodes} episodes)")
    pt.legend()
    pt.savefig("blackboxnew.png")
    pt.show()

