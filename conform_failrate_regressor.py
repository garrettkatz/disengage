import pickle as pk
import numpy as np
import torch as tr
from planar import PlanarEnv
from handcode_planar import Policy
from planar_features import FeatureExtractor
from get_failrate_data import do_rollouts, failure_function
from train_failrate_regressor import train_regressor

if __name__ == "__main__":

    do_reps = False

    # environment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion
    num_timesteps = 100

    # calibration parameters
    num_repetitions = 5000
    num_calib = 1000 # number of samples for calibration
    num_deploy = 1 # number of deployments after calibration
    delta = 0.003 # acceptable failure probability
    kernel = "linear" # kernel for feature extractor

    # setup environment and policy
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )
    policy = Policy(env)

    # setup regressor
    extractor = FeatureExtractor(env, kernel)
    regressor = tr.nn.Sequential(
        tr.nn.Linear(extractor.get_feature_dim(), 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 1),
        tr.nn.Sigmoid(),
        tr.nn.Flatten(0), # effectively squeezes
    )

    # load training/validation data
    with open("failrate_data_10k_512r.pkl", "rb") as f:
        (env, init_states, fail_rates) = pk.load(f)
    init_observation = env.get_observation(init_states)
    features, _ = extractor(init_observation)
    labels = fail_rates
    x = tr.tensor(features).float()
    y = tr.tensor(labels).float()

    # load pretrained model
    with open("tfr_results_10k_512r.pkl","rb") as f:
        (lc, predictions, labels) = pk.load(f)
    regressor.load_state_dict(tr.load("tfr_early_10k_512r.pt", weights_only=True))
    with tr.no_grad():
        predictions_early = regressor(x).numpy()
    stop = np.argmin(lc["valid"])

    # empirically check conformal guarantee
    if do_reps:

        failed = np.empty(num_repetitions, dtype=bool)
        engaged = np.empty(num_repetitions, dtype=bool)
        interval = np.empty(num_repetitions)
        tau = np.empty(num_repetitions)
        for rep in range(num_repetitions):
        
            # calibrate regression model
            calib_observation, _ = env.reset(batch_size=num_calib, upperhand=True)
            calib_states = env.state
            calib_failures = do_rollouts(env, policy, calib_states, 1, num_timesteps, failure_function, verbose=False) > .5
            calib_features = tr.tensor(extractor(calib_observation)[0]).float()
            with tr.no_grad():
                calib_predictions = regressor(calib_features).numpy()
            statistics = calib_failures * (1. - calib_predictions)
            sort_idx = np.argsort(statistics)
            statistics = statistics[sort_idx]
            calib_failures = calib_failures[sort_idx]
            interval[rep] = statistics[int(np.ceil((1-delta)*(num_calib+1)))]
            tau[rep] = 1 - interval[rep]
        
            # deploy regression model
            deploy_observation, _ = env.reset(batch_size=num_deploy, upperhand=True)
            deploy_states = env.state
            failures = do_rollouts(env, policy, deploy_states, 1, num_timesteps, failure_function, verbose=False)
            deploy_features = tr.tensor(extractor(deploy_observation)[0]).float()
            with tr.no_grad():
                deploy_predictions = regressor(deploy_features).numpy()
    
            failed[rep] = failures[0] > .5
            engaged[rep] = (deploy_predictions[0] < tau[rep])
    
            print(f"rep {rep} of {num_repetitions}: {1-delta} confidence interval = {interval[rep]:.5f}; tau = {tau[rep]:.5f}; deployment: failed = {failed[rep]}, engaged = {engaged[rep]}")

        with open("cfr_results.pkl","wb") as f:
            pk.dump((failed, engaged, interval, tau), f)

    with open("cfr_results.pkl","rb") as f:
        (failed, engaged, interval, tau) = pk.load(f)

    print("failed and disengaged: ", (failed & ~engaged).sum())
    print("failed and engaged:    ", (failed & engaged).sum())
    print("success and disengaged:", (~failed & ~engaged).sum())
    print("success and engaged:   ", (~failed & engaged).sum())
    
    import matplotlib.pyplot as pt
    # pt.subplot(1,2,1)
    # pt.hist(interval)
    # pt.title("Confidence interval size")
    # pt.subplot(1,2,2)
    pt.figure(figsize=(6,3))
    pt.hist(tau, bins=50, ec='k')
    pt.title("Selected $\\tau$ for $\\delta=0.3\\%$")
    pt.yscale("log")
    pt.gcf().supxlabel("$\\tau$")
    pt.gcf().supylabel("Frequency")
    pt.tight_layout()
    pt.savefig("cfr_tau.eps")
    pt.show()

    # pt.figure(figsize=(16,4))

    # pt.subplot(1,3,1)
    # pt.plot(lc["train"],"b-")
    # pt.plot(lc["valid"],"r-")
    # pt.yscale("log")
    # pt.legend(["Train","Validation"])
    # pt.ylabel("MSE")
    # pt.xlabel("Training iteration")
    # pt.title("Learning curves")

    # pt.subplot(1,3,2)
    # pt.scatter(labels[:num_train], predictions_early[:num_train], marker=".", color="b")
    # pt.scatter(labels[num_train:], predictions_early[num_train:], marker="+", color="r")
    # pt.xlim([min(predictions_early.min(),labels.min()), max(predictions_early.max(),labels.max())])
    # pt.ylim([min(predictions_early.min(),labels.min()), max(predictions_early.max(),labels.max())])
    # pt.legend(["Train","Validation"])
    # pt.xlabel("Label")
    # pt.ylabel("Prediction")
    # pt.title(f"Early stop ({stop}) performance")
    
    # pt.subplot(1,3,3)
    # x = np.arange(num_calib)
    # pt.plot(x[calib_failures], statistics[calib_failures], 'ro', label="failures")
    # pt.plot(x[~calib_failures], statistics[~calib_failures], 'go', label="successes")
    # pt.plot([0, num_calib], [interval]*2, 'k--', label="CI")
    # pt.xlabel("Sorted calibration point")
    # pt.ylabel("Statistic")
    # pt.legend()
    # pt.title("Calibration results")
    # pt.show()

