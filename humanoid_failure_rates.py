import numpy as np
from humanoid_conditional import EpisodeRunner

if __name__ == "__main__":

    num_rollouts = 100
    max_episode_length = 500
    obs_noises = [.1, .15, .2]

    failures = np.empty((len(obs_noises), num_rollouts), dtype=bool)
    durations = np.empty((len(obs_noises), num_rollouts), dtype=int)

    for n, obs_noise in enumerate(obs_noises):

        run_episode = EpisodeRunner(obs_noise, max_episode_length)
        for r in range(num_rollouts):
            print(n,r)
            failures[n,r], _, _, observations, _ = run_episode()
            durations[n,r] = len(observations)

    print("obs_noises:")
    print(obs_noises)
    print("failure rates:")
    print(failures.mean(axis=1))
    print("fail durations:")
    print([d[f].mean() for (d,f) in zip(durations, failures)])



