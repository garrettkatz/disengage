import numpy as np
import matplotlib.pyplot as pt
import torch as tr

pt.rcParams['font.family'] = 'serif'
pt.rcParams['font.size'] = 12

envs = {
    "Cart Pole": ("cp", .4, 1e-5, .5, 25, 25000),
    "Lunar Lander": ("ll", .1, 1e-3, .5, 50, 5000),
    "Humanoid Bench": ("hu", .1, 1e-4, 10., 25, 10_400),
    "Ruins": ("ru", .05, 5e-3, .2, 12, 10_400),
}
num_reps = 5

pt.figure(figsize=(8,2.5))

for sp, (title, (base, obs_noise, learning_rate, weight_decay, max_leadtime, num_updates)) in enumerate(envs.items()):
    print(f"plotting {title}...")

    valid_loss = np.empty((num_reps, max_leadtime))

    for train_rep in range(num_reps):
        train_basename = f"{base}_data/{base}_cond_train_rep{train_rep}_on{obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}"
        (_, loss_curve, _, _) = tr.load(f"{train_basename}_trained.pt", weights_only=True)
        for L in range(1, max_leadtime):
            valid_loss[train_rep, L-1] = min(loss_curve[L]["valid"])

    pt.subplot(1, len(envs), sp+1)
    pt.plot(np.arange(max_leadtime)+1, valid_loss.T, '.', color=(.8,)*3)
    pt.plot(np.arange(max_leadtime)+1, valid_loss.mean(axis=0), 'k-')
    # pt.yscale("log")
    pt.title(title)
    if sp == 0: pt.ylabel("Best Validation Loss")

pt.gcf().supxlabel("L", fontstyle="italic")
pt.tight_layout()
pt.savefig("all_mu.pdf")
pt.show()


