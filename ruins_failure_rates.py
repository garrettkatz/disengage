import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from ruins_conditional import EpisodeRunner

pt.rcParams['font.family'] = 'serif'
pt.rcParams['font.size'] = 12

if __name__ == "__main__":

    do_rollouts = True
    do_show = True
    base = "ru"

    num_rollouts = 100
    obs_noises = [.05, .2, .5]

    if do_rollouts:
        failures = np.empty((len(obs_noises), num_rollouts), dtype=bool)
        durations = np.empty((len(obs_noises), num_rollouts), dtype=int)
    
        for n, obs_noise in enumerate(obs_noises):
    
            run_episode = EpisodeRunner(obs_noise)
            for r in range(num_rollouts):
                print(n,r)
                failures[n,r], _, _, observations, _ = run_episode()
                durations[n,r] = len(observations)
            run_episode.close()
    
        print("obs_noises:")
        print(obs_noises)
        print("failure rates:")
        print(failures.mean(axis=1))
        print("fail durations:")
        print([d[f].mean() for (d,f) in zip(durations, failures)])

        with open(f"{base}_data/{base}_fr.pkl", "wb") as f: pk.dump((failures, durations), f)

    if do_show:

        with open(f"{base}_data/{base}_fr.pkl", "rb") as f: (failures, durations) = pk.load(f)

        print("obs_noises:")
        print(obs_noises)
        print("failure rates:")
        print(failures.mean(axis=1))
        print("fail durations:")
        print([d[f].mean() for (d,f) in zip(durations, failures)])

        pt.figure(figsize=(6,3))
        pt.subplot(1,2,1)
        pt.plot(failures.mean(axis=1), 'ko-')
        pt.xticks(range(len(obs_noises)), map(str, obs_noises))
        pt.ylabel("Failure Rate")

        pt.subplot(1,2,2)
        for n in range(len(obs_noises)):
            pt.plot(n + np.random.randn(failures[n].sum())*.05, durations[n, failures[n]], 'x', color=(.3,)*3)
            pt.plot(n + np.random.randn((1-failures[n]).sum())*.05, durations[n, ~failures[n]], 'o', mec=(.3,)*3, mfc='none')
            # pt.plot(n + np.random.randn(num_rollouts)*.05, durations[n], '.', color=(.5,)*3)
        pt.violinplot(durations.T, positions=range(len(obs_noises)), points=15, showextrema=False)
        #     pt.plot([n]*num_rollouts, durations[n, failures[n]], '.', color=(.5,)*3)
        # pt.violinplot([d[f] for (d,f) in zip(durations, failures)], positions=range(len(obs_noises)))
        pt.xticks(range(len(obs_noises)), map(str, obs_noises))
        pt.ylabel("Episode Duration")

        pt.gcf().supxlabel("Observation Noise")
        pt.tight_layout()
        pt.savefig(f"{base}_fr.pdf")
        pt.show()




