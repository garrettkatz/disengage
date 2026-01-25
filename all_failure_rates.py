import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

pt.rcParams['font.family'] = 'serif'
pt.rcParams['font.size'] = 14

pt.figure(figsize=(8,4))

envs = {
    "Cart Pole": ("cp", [.4, .5, .6]),
    "Lunar Lander": ("ll", [.1, .2, .3]),
    "Humanoid Bench": ("hu", [.1, .15, .2]),
    "Ruins": ("ru", [.05, .1, .15]),
}

for sp, (title, (base, obs_noises)) in enumerate(envs.items()):

    with open(f"{base}_failrates.pkl", "rb") as f: (failures, durations) = pk.load(f)

    print("obs_noises:")
    print(obs_noises)
    print("failure rates:")
    print(failures.mean(axis=1))
    print("fail durations:")
    print([d[f].mean() for (d,f) in zip(durations, failures)])

    pt.subplot(2,len(envs),sp+1)
    pt.plot(failures.mean(axis=1), 'ko-')
    pt.xticks(range(len(obs_noises)), map(str, obs_noises))
    pt.title(title, fontsize=14)
    if sp==0: pt.ylabel("Failure Rate")

    pt.subplot(2,len(envs),sp+1+len(envs))
    parts = pt.violinplot(durations[:,np.random.rand(100) < .1].T, positions=range(len(obs_noises)), points=50, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor((.8,)*3)
        pc.set_edgecolor('none')
        pc.set_alpha(1.)

    # subsample points for legibility
    subsample = (np.random.rand(failures.shape[1]) < .3)
    failures = failures[:,subsample]
    durations = durations[:,subsample]
    for n in range(len(obs_noises)):
        pt.plot(n + np.random.randn(failures[n].sum())*.05, durations[n, failures[n]], 'x', color=(.6,)*3)
        pt.plot(n + np.random.randn((1-failures[n]).sum())*.05, durations[n, ~failures[n]], 'o', mec=(.6,)*3, mfc='none')

    pt.xticks(range(len(obs_noises)), map(str, obs_noises))
    if sp==0: pt.ylabel("Episode Length")

pt.gcf().supxlabel("Observation Noise")
pt.tight_layout()
pt.savefig("all_failrates.pdf")
pt.show()


