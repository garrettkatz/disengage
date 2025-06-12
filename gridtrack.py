from line_profiler import profile # python -m kernprof -lvr gridtrack.py
import itertools as it
import numpy as np
import matplotlib.patches as mp

class GridTrackActionSpace:
    def __init__(self, num_trackers, rng):
        self.num_trackers = num_trackers
        self.rng = rng
    def sample(self):
        return self.rng.integers(-1, +1, size=(self.num_trackers, 2), endpoint=True)

class GridTrackState:

    @profile
    def __init__(self, env, trackers=None, targets=None, visible_data=None):
        """
        Randomly samples trackers and targets if not provided
        Recomputes visibility data (vis_coords, is_visible) if not provided
        """
        self.env = env

        if targets is None:
            targets = env.rng.integers(env.shape, size=(env.num_targets, 2))
        self.targets = targets

        if trackers is None:
            # sample tracker coordinates outside of walls
            nonwall = np.argwhere(~env.wall_mask)
            trackers = env.rng.choice(nonwall, size=env.num_trackers)
        self.trackers = trackers

        if visible_data is None:

            # initialize visible coordinates
            self.vis_coords = self.trackers[:, None, :] + env.vrc[None, :, :] # (T, V, 2)
    
            # mark target visibility
            self.is_visible = (self.targets[:,None,:] == self.vis_coords.reshape(-1, 2)[None,:,:]).all(axis=2).any(axis=1)

        else:
            self.vis_coords, self.is_visible = visible_data

    @profile
    def _transit_agents(self, action, target_motion=None, clip_targets=True):
        env = self.env

        step_trackers = self.trackers + action
        step_trackers = np.minimum(np.maximum(step_trackers, 0), env.shape-1)
        no_wall = ~env.wall_mask[*step_trackers.T]
        trackers = np.where(no_wall[:,None], step_trackers, self.trackers)

        if target_motion is None:
            target_motion = env.rng.integers(-1, 1, size=(env.num_targets, 2), endpoint=True)

        targets = self.targets + target_motion
        if clip_targets:
            targets = np.maximum(np.minimum(targets, env.shape-1), 0)

        return trackers, targets

    def all_transits(self, action):
        """
        Calculate all possible transitions from given action
        """
        env = self.env
        
        step_trackers = self.trackers + action
        step_trackers = np.minimum(np.maximum(step_trackers, 0), env.shape-1)
        no_wall = ~env.wall_mask[*step_trackers.T]
        trackers = np.where(no_wall[:,None], step_trackers, self.trackers)

        all_targets = self.targets + env.all_target_motions
        all_targets = np.maximum(np.minimum(all_targets, env.shape-1), 0)

        return trackers, all_targets

    @profile
    def transition(self, action, target_motion=None, recompute_visibility=True, clip_targets=True):
        """
        Target motion random if not provided
        """
        env = self.env

        trackers, targets = self._transit_agents(action, target_motion, clip_targets)

        if recompute_visibility:
            deltas = trackers - self.trackers
            vis_coords = self.vis_coords + deltas[:, None, :]
            is_visible = (targets[:,None,:] == vis_coords.reshape(-1, 2)[None,:,:]).all(axis=2).any(axis=1)
        else:
            vis_coords = self.vis_coords
            is_visible = self.is_visible

        new_state = GridTrackState(env, trackers, targets, (vis_coords, is_visible))
        return new_state

class GridTrackEnv:
    """
    Each wall is a vertical or horizontal strip that spans the midpoint of the grid
    visibility is the max-norm radius of visibility around each tracker

    observations: many-hot arrays where
        obs[r,c,0] = (target at r,c)
        obs[r,c,1] = (wall at r,c)
        obs[r,c,2] = (tracker at r,c)
        obs[r,c,3] = (flag at r,c)
    if env is not fully observable, only visible targets are indicated

    actions: action[t] = (dr, dc) for tracker t
    dr and dc are each in (-1, 0, +1)
    trackers cannot move off grid or through walls
    targets cannot move off grid, but can move through walls

    rewards:
        base reward is number of targets visible to at least one tracker
        flag_penalty is subtracted from base for each non-visible target that is at a flag

    """
    def __init__(self, wall_mask, flags, num_trackers, num_targets, visibility, flag_penalty, fully_observable):
        self.num_rows = wall_mask.shape[0]
        self.num_cols = wall_mask.shape[1]
        self.shape = np.array(wall_mask.shape)
        self.num_trackers = num_trackers
        self.num_targets = num_targets
        self.visibility = visibility
        self.flag_penalty = flag_penalty
        self.fully_observable = fully_observable
        self.flags = flags

        self.rng = np.random.default_rng()

        # get wall coordinates
        self.wall_mask = wall_mask
        self.walls = []
        if wall_mask.any():
            self.walls = list(map(tuple, np.argwhere(wall_mask)))

        # cache fixed part of observation (walls and flags)
        self.fixed_observation = np.zeros((self.num_rows, self.num_cols, 4))
        for (r, c) in self.walls: self.fixed_observation[r,c,1] = 1
        for (r, c) in self.flags: self.fixed_observation[r,c,3] = 1

        # cache meshgrid of visibility range
        vr, vc = np.mgrid[-visibility:visibility+1, -visibility:visibility+1]
        self.vrc = np.stack([vr.flatten(), vc.flatten()]).T

        # cache powers for state-index conversion
        self.bases = np.tile(self.shape, self.num_trackers + self.num_targets)
        self.powers = np.ones(len(self.bases), dtype=int)
        self.powers[1:] = np.cumprod(self.bases)[:-1]

        # cache all possible target motions
        self.all_target_motions = np.array(list(it.product((-1,0,+1), repeat=2*num_targets))).reshape((-1, num_targets, 2))

    def randomized(num_rows, num_cols, num_walls, num_trackers, num_targets, num_flags, visibility, flag_penalty, fully_observable):

        shape = np.array([num_rows, num_cols])

        rng = np.random.default_rng()

        # build random walls
        walls = []
        for w in range(num_walls):
            dim = rng.choice(2)
            coord = rng.integers(shape[dim])
            lo = rng.integers(shape[1-dim]//2)
            hi = rng.integers(shape[1-dim]//2, shape[1-dim])
            cols = np.arange(lo, hi+1)
            rows = np.full(cols.shape, coord)
            if dim == 1: rows, cols = cols, rows
            walls.extend(zip(rows, cols))

        wall_mask = np.zeros(tuple(shape), dtype=bool)
        if len(walls) > 0:
            wall_mask[tuple(zip(*walls))] = True

        # sample flag coordinates outside of walls
        nonwall = np.argwhere(~wall_mask)
        flags = rng.choice(nonwall, size=num_flags)

        return GridTrackEnv(wall_mask, flags, num_trackers, num_targets, visibility, flag_penalty, fully_observable)

    def reset(self, seed=None):

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.action_space = GridTrackActionSpace(self.num_trackers, self.rng)

        # set randomly sampled state
        self.current_state = GridTrackState(self)

        # return observation and info
        observation = self._observe_state()
        info = None
        return observation, info

    @profile
    def _observe_state(self):
        # returns manyhot array obs where
        # obs[r,c,0] = (target at r,c)
        # obs[r,c,1] = (wall at r,c)
        # obs[r,c,2] = (tracker at r,c)
        # if env is not fully observable, only visible targets are included
        # assumes self.is_visible is up-to-date
        
        observation = self.fixed_observation.copy()

        rs, cs = self.current_state.trackers.T
        observation[rs, cs, 2] = 1

        if self.fully_observable:
            rs, cs = self.current_state.targets.T
            observation[rs, cs, 0] = 1
        elif self.is_visible.any():
            rs, cs = self.current_state.targets[self.is_visible, :].T
            observation[rs, cs, 0] = 1

        return observation

    def reward_function(self, state=None):
        """
        assumes is_visible is already up-to-date
        """
        if state is None: state = self.current_state

        reward = state.is_visible.sum()
        on_flag = (state.targets[:,None,:] == self.flags[None,:,:]).all(axis=2).any(axis=1)
        reward -= self.flag_penalty * (on_flag & ~state.is_visible).sum()
        return reward

    @profile
    def step(self, action, target_motion=None):

        self.current_state = self.current_state.transition(action, target_motion)
        observation = self._observe_state()
        reward = self.reward_function()
        terminated = None
        truncated = None
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
        ax.clear()

        # grid lines
        for r in range(self.num_rows+1): ax.plot([0, self.num_cols], [r, r], 'k-')
        for c in range(self.num_cols+1): ax.plot([c, c], [0, self.num_rows], 'k-')

        # walls
        for (r,c) in self.walls:
            ax.add_patch(mp.Rectangle((c,r), 1, 1, color='k', fill=True))

        # flags
        for (r,c) in self.flags:
            ax.add_patch(mp.Rectangle((c,r), 1, 1, color='g', fill=True))

        # targets
        for (r, c) in self.current_state.targets:
            ax.add_patch(mp.Circle((c+.5,r+.5), radius=.4, ec='k', fc='r'))

        # trackers
        for (r, c) in self.current_state.trackers:
            ax.add_patch(mp.CirclePolygon((c+.5,r+.5), radius=.3, resolution=4, ec='k', fc='b'))

        # visibility
        for (r, c) in set(map(tuple, self.current_state.vis_coords.reshape(-1, 2))):
            if (r < 0) or (c < 0) or (self.shape <= (r, c)).any(): continue
            ax.add_patch(mp.Rectangle((c, r), 1, 1, color='b', alpha=.2))

        ax.set_aspect('equal', 'box')

        if hang:
            if msg is None: input('.')
            else: input(msg)
        elif msg is not None: print(msg)

if __name__ == "__main__":

    import matplotlib.pyplot as pt

    viz = True

    env = GridTrackEnv.randomized(
        num_rows = 10,
        num_cols = 12,
        num_walls = 4,
        num_trackers = 2,
        num_targets = 2,
        num_flags = 20,
        visibility = 2,
        flag_penalty = 10,
        fully_observable=True,
    )
    observation, info = env.reset()

    if viz:
        fig, axs = pt.subplots(1,2)
        pt.ion()
        pt.show()

        obs_img = observation[:,:,:3]
        obs_img[:,:,1] += observation[:,:,3]*.5
        axs[0].imshow(obs_img)
        env.render(axs[1], hang=True)

    for t in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if viz:
            obs_img = observation[:,:,:3]
            obs_img[:,:,1] += observation[:,:,3]*.5
            axs[0].imshow(obs_img)
            env.render(axs[1], hang=True, msg=f"t={t}: reward={reward}")

