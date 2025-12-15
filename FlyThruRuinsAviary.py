"""
Adapted from 
https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/da52ed17e0bc1923a1f0eb7d7d2cecdf01aec4f9/gym_pybullet_drones/envs/FlyThruGateAvitary.py
"""
import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

def add_box(position, quaternion, half_extents, rgb, mass=0.1):

    cid = p.createCollisionShape(
        shapeType = p.GEOM_BOX,
        halfExtents = half_extents,
    )
    vid = p.createVisualShape(
        shapeType = p.GEOM_BOX,
        halfExtents = half_extents,
        rgbaColor = rgb+(1,), # alpha is opaque
    )
    mid = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition = position,
        baseOrientation = quaternion,
    )

    return mid

class FlyThruRuinsAviary(BaseRLAviary):
    """Single agent RL problem: Navigate through ruins.
    
    The starting position is default near origin with added uniform xy noise in [-.3, +.3]
    The goal position is at [0, -2, 0.75]
    Scattered walls lie in between
    Collision with walls (closer than .025) is considered failure
    Actions have added noise as a proxy for disturbances like wind
    """

    ################################################################################

    def set_camera(self):
        # good for hand-placed walls
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=-152,
            cameraPitch=-71,
            cameraTargetPosition=[-.2,-.77,.17])


    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int=240,
                 ctrl_freq: int=30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 buggy=False,
                 num_walls=7,
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 8
        self.num_walls = num_walls
        super().__init__(drone_model=drone_model,
                         num_drones = 1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

        self.base_xyzs = self.INIT_XYZS.copy()
        self.base_rpys = self.INIT_RPYS.copy()
        self.buggy = buggy

        self.set_camera()

    def reset(self,
              seed : int = None,
              options : dict = None):

        # make sure random initial points are within truncation range:
        # (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0# Truncate when the drone is too far away
        #      or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        self.INIT_XYZS = self.base_xyzs + np.random.uniform([-.3, -.3, 0], [+.3, +.3, 0], self.base_xyzs.shape)
        self.INIT_RPYS = self.base_rpys + np.random.uniform(-.1, .1, self.base_rpys.shape)
        # print(self.INIT_XYZS)
        # input('.')

        self.set_camera()

        result = super().reset(seed, options)

        # goal position
        p.addUserDebugPoints(
            pointPositions = [(0., -2., 0.75)],
            pointColorsRGB = [(0., 1., 0.)],
            pointSize = 10.,
        )

        return result

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment. Designed to be like walls of ruins.
        """

        half_extents = (.25, .01, .75)
        rgb = (.5, .5, .5)
        self.obstacle_ids = []

        # # random wall placement
        # rng = np.random.default_rng(seed=12345) # keep object locations constant across runs for now
        # for wall in range(self.num_walls):
        #     position = rng.uniform([-1, .5, .75], [1, 1.3, .75])
        #     angle = rng.uniform(-3.14, 3.14)
        #     quaternion = p.getQuaternionFromEuler((0,0,angle))
        #     self.obstacle_ids.append(add_box(position, quaternion, half_extents, rgb, mass=0))

        # hand-placed walls
        positions_angles = [
            ((.2, -.4, .75), .2),
            ((.4, -.3, .75), .1),
            ((-.3, -1, .75), -.4),
            ((-.5, -.8, .75), -.6),
            ((-.7, -.7, .75), -.3),
            ((.8, -1.4, .75), .2),
            ((1, -1.3, .75), .1),
        ]
        for (position, angle) in positions_angles:
            quaternion = p.getQuaternionFromEuler((0,0,angle))
            self.obstacle_ids.append(add_box(position, quaternion, half_extents, rgb, mass=0))

        # sandy ground
        ground_id = add_box((0,0,0), (0,0,0,1), (100, 100, .02), (246/255, 215/255, 176/255), mass=0)
        sand_id = p.loadTexture("noisy_sand.png")
        p.changeVisualShape(ground_id, -1, textureUniqueId = sand_id)
        self.obstacle_ids.append(ground_id)

    def collision_distance(self):
        drone_id = self.DRONE_IDS[0]
        pos, quat = p.getBasePositionAndOrientation(drone_id)
        dist = pos[2]
        for body_id in self.obstacle_ids:
            points = p.getClosestPoints(drone_id, body_id, 1)
            for point in points:
                # dist = min(dist, point.contactDistance)
                dist = min(dist, point[8])
        return dist

    def failure_predicate(self):
        dist = self.collision_distance()
        return dist < .01


    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter/self.PYB_FREQ) / self.EPISODE_LEN_SEC
        reward = max(0, 1 - np.linalg.norm(np.array([0, -2*norm_ep_time, 0.75])-state[0:3]))

        failure = self.failure_predicate()
        if failure: reward -= 1

        return reward

    ################################################################################
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0# Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("out of time")
            return True
        else:
            return False

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _preprocessAction(self, action):
        # add small random noise (relative units) to rpms
        action = action * np.random.uniform(0.99, 1.01, action.shape)
        return super()._preprocessAction(action)

    def _computeObs(self):
        # BaseRLAviary was updated to include action in observation; remove this for compatibility with phuongboi
        ret = super()._computeObs()
        return ret[...,:12]


    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

if __name__ == "__main__":

    env = FlyThruRuinsAviary(gui=True, num_walls=7)
    input('.')

    # env.reset()
    # while True:
    #     action = np.random.randn(1,4)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     print(reward)
    #     input('.')
    #     if done:
    #         print(terminated, truncated)
    #         break



