import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

class RandomHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_xyzs = self.INIT_XYZS.copy()
        self.base_rpys = self.INIT_RPYS.copy()
        self.buggy = kwargs.get("debug", False)

    def reset(self,
              seed : int = None,
              options : dict = None):

        self.INIT_XYZS = self.base_xyzs + .1*np.random.randn(*self.base_xyzs.shape)
        self.INIT_RPYS = self.base_rpys + .05*np.random.randn(*self.base_rpys.shape)
        # print(self.INIT_XYZS)
        # input('.')
        return super().reset(seed, options)

    def _computeObs(self):
        # BaseRLAviary was updated to include action in observation; remove this for compatibility with phuongboi
        ret = super()._computeObs()
        return ret[...,:12]

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)

        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)

        # https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/da52ed17e0bc1923a1f0eb7d7d2cecdf01aec4f9/gym_pybullet_drones/envs/HoverAviary.py#L88
        rpy = state[7:10]
        ang_vel=state[13:16]
        ret = max(0, 1 - np.linalg.norm(self.TARGET_POS-state[0:3])) - 0.001*np.linalg.norm(rpy) - 0.001* np.linalg.norm(ang_vel)
        return ret

