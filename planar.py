from line_profiler import profile # python -m kernprof -lvr planar.py
import numpy as np
import matplotlib.patches as mp

_TPI = 2*np.pi

class PlanarActionSpace:
    def __init__(self, num_allies, step_size, rng):
        self.num_allies = num_allies
        self.step_size = step_size
        self.rng = rng
    def sample(self):
        return self.step_size * self.rng.uniform(-1, +1, size=(self.num_allies, 3))

class PlanarState:
    def __init__(self, allies, adversaries):
        self.allies = allies
        self.adversaries = adversaries

    def render(self, ax):
        """
        render current state on matplotlib Axes ax
        """
        ax.clear()

        for agents, color in ((self.allies, 'g'), (self.adversaries, 'r')):

            # headings
            U, V = .1*np.cos(agents[:,2]), .1*np.sin(agents[:,2])
            ax.quiver(agents[:,0], agents[:,1], U, V, angles='xy', scale_units='xy', scale=1)

            # coordinates
            ax.scatter(*agents.T[:2], marker='o', color=color)
            for i, (x, y) in enumerate(agents[:,:2]): ax.text(x, y, str(i))

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal', 'box')

class PlanarEnv:
    def __init__(self, num_allies, num_adversaries, step_size=0.05, view_angle=np.pi/2):
        self.num_allies = num_allies
        self.num_adversaries = num_adversaries
        self.step_size = step_size
        self.view_angle = view_angle
        self.rng = np.random.default_rng()
        self.action_space = PlanarActionSpace(num_allies, self.step_size, self.rng)
        self.state = None

    def transition(self, state, agent_motion, opponent_motion=None):
        # assumes |agent_motion| <= self.step_size
        # should support batching in the state

        if opponent_motion is None:
            opponent_motion = self.step_size * self.rng.uniform(-1, 1, state.adversaries.shape)

        # take step
        allies = state.allies + agent_motion
        adversaries = state.adversaries + opponent_motion

        allies[:,:2] = np.minimum(1, np.maximum(0, allies[:,:2]))
        adversaries[:,:2] = np.minimum(1, np.maximum(0, adversaries[:,:2]))

        return PlanarState(allies, adversaries)

    def reward_function(self, state):
        # adversaries are visible

        def team_reward(team, opponents):
            diffs = opponents[:,None,:2] - team[None,:,:2]
            angles = np.arctan2(diffs[:,:,1], diffs[:,:,0]) % _TPI
            headings = team[:,2] % _TPI
            deltas = np.minimum((angles - headings) % _TPI, (headings - angles) % _TPI)
            viz = deltas < self.view_angle
            return viz.sum()

        allies_reward = team_reward(state.allies, state.adversaries)
        adversaries_reward = team_reward(state.adversaries, state.allies)
        return allies_reward - adversaries_reward

    def random_state(self):
        allies = self.rng.uniform(size=(self.num_allies, 3))
        allies[:,2] *= _TPI
        adversaries = self.rng.uniform(size=(self.num_adversaries, 3))
        adversaries[:,2] *= _TPI
        return PlanarState(allies, adversaries)

    def get_observation(self, state):
        return np.concatenate((state.allies, state.adversaries), axis=None)

    def reset(self, seed=None):

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # initialize state
        self.state = self.random_state()

        # return observation and info
        observation = self.get_observation(self.state)
        info = None
        return observation, info

    @profile
    def step(self, action, opponent_motion=None):

        self.state = self.transition(self.state, action, opponent_motion)
        observation = self.get_observation(self.state)
        reward = self.reward_function(self.state)
        terminated = False
        truncated = False
        info = None
        return observation, reward, terminated, truncated, info

    def close(self):
        pass

    def render(self, ax, hang=False, msg=None):
        """
        render current state on matplotlib Axes ax
        if msg is not None, it is printed (should be type str)
        if hang is True, waits until user hits enter after message is printed 
        """
        self.state.render(ax)
        if hang:
            if msg is None: input('.')
            else: input(msg)
        elif msg is not None: print(msg)


if __name__ == "__main__":

    import matplotlib.pyplot as pt

    env = PlanarEnv(3, 4, .01)
    env.reset()

    pt.ion()
    pt.show()
    for t in range(1000):
        env.render(pt.gca(), hang=False, msg = f"t={t}, r={env.reward_function(env.state)}")
        action = env.action_space.sample()
        action = np.full((3,3), .05)
        env.step(action)
        print(env.state.allies)
        pt.pause(0.01)
    input('.')

