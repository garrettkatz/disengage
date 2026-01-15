import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import sb_utils as su
from lunar_lander_mu import failure_predicate

if __name__ == "__main__":

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    num_reps = 10
    num_rollouts = 30

    env, model = su.load(env_name, alg_name)
    durations = np.empty((num_reps, num_rollouts))
    for rep in range(num_reps):
        init_state = env.reset()
        print(rep, init_state)
        for rollout in range(num_rollouts):
            _, _, durations[rep, rollout],_, _ = su.run(env, model, num_timesteps, failure_predicate, init_state=init_state)

    pt.plot((np.arange(num_reps)[:,None] + np.random.randn(*durations.shape)*.05).flatten(), durations.flatten(), 'k.')
    pt.xlabel("Initial state sample")
    pt.ylabel("Episode durations")
    pt.show()


    # # experimental repetitions to estimate preemption failure rate
    # ep_lens = []
    # failures = []
    # for rep in range(num_rollouts):

    #     env.reset()
    #     env.set_attr("state", init_state, indices=0)

    #     obs = init_state
    #     failure = False
    #     ep_len = 0
    #     ep_rew = 0
    #     lstm_states = None
    #     episode_start = np.ones((env.num_envs,), dtype=bool)

    #     for timestep in range(num_timesteps):
    #         if timestep <= 1: print(rep, obs)
    
    #         with tr.no_grad():
    #             action, lstm_states = model.predict(
    #                 obs,  # type: ignore[arg-type]
    #                 state=lstm_states,
    #                 episode_start=episode_start,
    #                 deterministic=True,
    #             )
    
    #         if perturb_action is not None:
    #             action = perturb_action(action)
    
    #         obs, reward, done, infos = env.step(action)    
    #         episode_start = done
    
    #         ep_rew += reward[0]
    #         ep_len += 1
    
    #         failure = failure_predicate(env, obs, reward, done, infos)
    #         if failure: done = True
    
    #         if done: break

    #     print(infos)
    #     ep_lens.append(ep_len)
    #     failures.append(failure)

    # print(ep_lens)

