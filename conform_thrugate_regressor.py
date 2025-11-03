import pickle as pk
import numpy as np
import torch as tr

if __name__ == "__main__":

    # these numbers specific to 345k sample (3% blackbox failrate) leaving 338k for calibration tests
    num_train = 1600 # number of samples for fitting only
    num_valid = 400 # number of samples for testing only, leave some for calibration
    num_repetitions = 500 # number of reps
    num_calib = 685 # number for calibration, 1 more is deployment

    delta = 0.01 # acceptable failure probability

    npz = np.load("gfd_345k.npz")
    labels = npz["failures"]
    features = npz["init_obs"]
    npz.close()

    print(f"{num_train} training examples ({len(labels)}) total, {features.shape[-1]} features")

    # MLP - same architecture as critic
    # num_hidden = 64 # same as critic
    num_hidden = 16
    regressor = tr.nn.Sequential(
        tr.nn.Linear(features.shape[-1], num_hidden),
        tr.nn.Tanh(),
        tr.nn.Linear(num_hidden, num_hidden),
        tr.nn.Tanh(),
        tr.nn.Linear(num_hidden, 1),
    )

    print(f"MLP: {sum([p.numel()for p in regressor.parameters()])} parameters")

    x = tr.tensor(features).float()
    regressor.load_state_dict(tr.load("ttr_early.pt", weights_only=True))
    with tr.no_grad():
        predictions = regressor(x)

    # apply sigmoids and flatten
    predictions = tr.sigmoid(predictions).flatten().numpy()

    # shuffle the calibration set
    idx = np.random.permutation(num_repetitions * (num_calib+1))
    predictions[num_train+num_valid:] = predictions[num_train+num_valid:][idx]
    labels[num_train+num_valid:] = labels[num_train+num_valid:][idx]

    # do the repetitions
    failed = np.empty(num_repetitions, dtype=bool)
    engaged = np.empty(num_repetitions, dtype=bool)
    interval = np.empty(num_repetitions)
    tau = np.empty(num_repetitions)
    for rep in range(num_repetitions):

        # get the current data
        calib_offset = num_train+num_valid+rep*(num_calib+1)
        deploy_offset = calib_offset + num_calib
        calib_predictions = predictions[calib_offset:deploy_offset]
        calib_failures = labels[calib_offset:deploy_offset]

        # do conformal thing
        statistics = calib_failures * (1. - calib_predictions)
        sort_idx = np.argsort(statistics)
        statistics = statistics[sort_idx]
        calib_failures = calib_failures[sort_idx]
        interval[rep] = statistics[int(np.ceil((1-delta)*(num_calib+1)))]
        tau[rep] = 1 - interval[rep]
    
        # deploy regression model
        deploy_prediction = predictions[deploy_offset]
        failed[rep] = labels[deploy_offset] > .5
        engaged[rep] = (deploy_prediction < tau[rep])

        print(f"rep {rep} of {num_repetitions}: {1-delta} confidence interval = {interval[rep]:.5f}; tau = {tau[rep]:.5f}; deployment: failed = {failed[rep]}, engaged = {engaged[rep]}")

    with open("ctr_results.pkl","wb") as f:
        pk.dump((failed, engaged, interval, tau), f)

    with open("ctr_results.pkl","rb") as f:
        (failed, engaged, interval, tau) = pk.load(f)

    print(f"failed and disengaged:  {(failed & ~engaged).sum()} ({(failed & ~engaged).mean()*100:.1f}%)")
    print(f"failed and engaged:     {(failed & engaged).sum()} ({(failed & engaged).mean()*100:.1f}%)")
    print(f"success and disengaged: {(~failed & ~engaged).sum()} ({(~failed & ~engaged).mean()*100:.1f}%)")
    print(f"success and engaged:    {(~failed & engaged).sum()} ({(~failed & engaged).mean()*100:.1f}%)")
    
    import matplotlib.pyplot as pt
    # pt.subplot(1,2,1)
    # pt.hist(interval)
    # pt.title("Confidence interval size")
    # pt.subplot(1,2,2)
    pt.figure(figsize=(6,3))
    pt.hist(tau, bins=50, ec='k')
    pt.yscale('log')
    pt.title(f"Selected $\\tau$ for $\\delta={delta*100:.2f}\\%$")
    # pt.yscale("log")
    pt.gcf().supxlabel("$\\tau$")
    pt.gcf().supylabel("Frequency")
    pt.tight_layout()
    pt.savefig("ctr_tau.eps")
    pt.savefig("ctr_tau.png")
    pt.show()
