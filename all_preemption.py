import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import torch as tr

pt.rcParams['font.family'] = 'serif'
pt.rcParams['font.size'] = 12

envs = {
    "Cart Pole": ("cp", [.4, .5, .6], 1e-5, .5, 25, 25000),
    "Lunar Lander": ("ll", [.1, .2, .3], 1e-3, .2, 50, 5000),
    "Humanoid Bench": ("hu", [.1, .15, .2], 1e-4, 10., 25, 10_400),
    # "Ruins": ("ru", [.05, .2, .5], 5e-3, .2, 12, 10_400),
    "Ruins": ("ru", [.05, .1, .15], 5e-3, .2, 12, 10_400),
}
num_train_reps = 5
num_repetitions = 100
delta_ratios = [.1, .25, .5]

pt.figure(figsize=(8,5))

for sp, (title, (base, obs_noises, learning_rate, weight_decay, max_leadtime, num_updates)) in enumerate(envs.items()):
    print(f"plotting {title}...")

    Ls = np.arange(max_leadtime)+1
    unpreempted = np.empty((6, num_repetitions, num_train_reps, 2, len(delta_ratios), max_leadtime))
    false_alarms = np.empty((6, num_repetitions, num_train_reps, 2, len(delta_ratios), max_leadtime))
    combos = np.empty((6,2))

    for n, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):
        combos[n] = (calib_obs_noise, final_obs_noise)

        # load results across each repetition
        for rep in range(num_repetitions):

            results_name = f"{base}_data/{base}_cond_conform_rep{rep}_tn{obs_noises[0]}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
                f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}.pkl"
            with open(results_name, "rb") as f:
                (failrate, delta_ratios, deltas, num_calibration, final_failure, final_duration, taus, alarm_times) = pk.load(f)

            # unpreempted if failed and first alarm came at or later than max(1, fail time - L)
            unpreempted[n, rep] = final_failure & (alarm_times >= np.maximum(1, final_duration - Ls))

            # false alarm if not failed and first alarm came at any time before episode end
            false_alarms[n, rep] = (not final_failure) & (alarm_times < final_duration)

    # rates are average across conformal reps
    rand_rate = unpreempted[:,:,:,0,0,:].mean(axis=1)
    pret_rate = unpreempted[:,:,:,1,0,:].mean(axis=1)
    # mean/stdv across training reps
    rand_mean = rand_rate.mean(axis=1)
    pret_mean = pret_rate.mean(axis=1)
    rand_std = rand_rate.std(axis=1)
    pret_std = pret_rate.std(axis=1)
    # mean/max of mean-stdev across L
    pret_max = (pret_mean-pret_std).max(axis=-1)
    pret_mean = (pret_mean-pret_std).mean(axis=-1)

    pt.subplot(2, len(envs), sp+1)
    pt.bar(np.arange(6), height=pret_mean, yerr=np.stack([np.zeros(6),pret_max-pret_mean]), edgecolor='k', facecolor=(.8,.8,1), align="edge")
    pt.plot([0,6], [deltas[0]]*2, 'k:')
    # pt.xticks(np.arange(6), [str(tuple(c)) for c in combos], rotation=45)
    pt.xticks([],[])
    pt.title(title)
    if sp == 0: pt.ylabel("Unpreempted\nFailure Rate")

    # rates are average across conformal reps
    rand_rate = false_alarms[:,:,:,0,0,:].mean(axis=1)
    pret_rate = false_alarms[:,:,:,1,0,:].mean(axis=1)
    # mean/stdv across training reps
    rand_mean = rand_rate.mean(axis=1)
    pret_mean = pret_rate.mean(axis=1)
    rand_std = rand_rate.std(axis=1)
    pret_std = pret_rate.std(axis=1)
    # first L where pret mean+stdev > rand mean-stdev
    effective_leadtime_mask = ((pret_mean + pret_std) > (rand_mean - rand_std))
    effective_leadtime = np.where(effective_leadtime_mask.any(axis=-1), effective_leadtime_mask.argmax(axis=-1), Ls.max())
    print(effective_leadtime)

    pt.subplot(2, len(envs), len(envs)+sp+1)
    pt.bar(np.arange(6), height=effective_leadtime, edgecolor='k', facecolor=(.8,.8,1), align="edge")
    pt.plot([0,6], [Ls.max()]*2, 'k:')
    pt.xticks(np.arange(6)+.4, [str(tuple(map(float,c))) for c in combos], rotation=90)
    # pt.title(title)
    if sp == 0: pt.ylabel("Effective\nLead Time")

pt.gcf().supxlabel(f"$(\\nu_{{calib}},\\nu_{{deploy}})$")
pt.tight_layout()
pt.savefig("all_pre.pdf")
pt.show()




