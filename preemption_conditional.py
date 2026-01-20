import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import matplotlib.pyplot as pt

def conform(params, delta_ratios, setup_mu, run_calib, run_final):

    basename = params["basename"]
    train_rep = params["train_rep"]
    train_obs_noise = params["train_obs_noise"]
    calib_obs_noise = params["calib_obs_noise"]
    final_obs_noise = params["final_obs_noise"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    calib_padding = params["calib_padding"]
    num_repetitions = params["num_repetitions"]
    load_mus = params["load_mus"]

    train_name = f"{basename}_train_rep{train_rep}_on{train_obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_trained.pt"
    results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
        f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"

    # load failrate and mus
    (failrate, _, _, mu_state_dicts) = tr.load(train_name, weights_only=True)
    mu = {}
    for L in range(1, max_leadtime + 1):
        mu[L] = setup_mu()
        if load_mus: mu[L].load_state_dict(mu_state_dicts[L])

    # setup calibration parameters
    deltas = [dr * failrate for dr in delta_ratios]
    num_calibration = int(np.ceil((1 - min(deltas) + calib_padding) / min(deltas))) # ensure non-infinite taus (unless no failures)

    deploy_failures = np.empty(num_repetitions, dtype=bool)
    deploy_durations = np.empty(num_repetitions, dtype=int)
    taus = np.empty((num_repetitions, len(deltas), max_leadtime))
    alarm_times = np.empty((num_repetitions, len(deltas), max_leadtime), dtype=int) # episode length signifies no alarm
    for rep in range(num_repetitions):

        # collect calibration episodes
        calib_observations, calib_failure = [], []
        for episode in range(num_calibration):
            failure, _, _, observations, _ = run_calib()
            calib_observations.append(tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32))
            calib_failure.append(failure)

        # final episode
        final_failure, _, _, final_observations, _ = run_final()
        final_observations = tr.tensor(np.concatenate(final_observations, axis=0)).to(tr.float32)
        deploy_failures[rep] = final_failure
        deploy_durations[rep] = len(final_observations)

        # conformal method for each L and delta
        for L in range(1, max_leadtime + 1):

            # calibration stats
            calib_predictions = [mu[L](obs).squeeze() for obs in calib_observations]
            z = [(preds[:max(1,len(preds)-L)].max().item() if fail else np.inf) for (preds, fail) in zip(calib_predictions, calib_failure)]
            z.sort(reverse=True)

            # predictions on final episode
            final_preds = mu[L](final_observations).squeeze()
            
            # test different deltas
            for d, delta in enumerate(deltas):

                # set tau
                tau = z[int(np.ceil((num_calibration + 1)*(1 - delta)))-1] # -1 for 0-based indexing
                print(f" {L=}, {delta=}, {tau=}")
    
                # check for alarm on final episode
                alarms = (final_preds[:max(1, len(final_preds)-L)] > tau).numpy()

                # save results
                taus[rep, d, L-1] = tau
                if alarms.any():
                    alarm_times[rep, d, L-1] = alarms.argmax()
                else:
                    alarm_times[rep, d, L-1] = len(final_observations)

        for d, delta in enumerate(deltas):
            print(f" {failrate=}, {num_updates=}, {delta=}, {num_calibration=}, {rep=}, {final_failure=}, dur={len(final_observations)}, alarms at:", alarm_times[rep, d])

    print(f"\n\n **** {load_mus=}, {train_rep=}, {calib_obs_noise=}, {final_obs_noise=} ***\n\n")

    print(f"{failrate=}, {num_updates=}, {num_calibration=}, {deltas=}:")
    sound_alarms =  deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
    false_alarms = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
    unpreempted = deploy_failures[:,None,None] & (alarm_times==deploy_durations[:,None,None])
    print(f" unpreempted rates (delta x L):")
    print(unpreempted.mean(axis=0).round(3))
    print(f" false alarm rates (delta x L):")
    print(false_alarms.mean(axis=0).round(3))

    with open(results_name, "wb") as f:
        pk.dump((failrate, delta_ratios, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times), f)


def show_results(params):

    basename = params["basename"]
    train_rep = params["train_rep"]
    train_obs_noise = params["train_obs_noise"]
    calib_obs_noise = params["calib_obs_noise"]
    final_obs_noise = params["final_obs_noise"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    load_mus = params["load_mus"]

    train_name = f"{basename}_train_rep{train_rep}_on{train_obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_trained.pt"
    results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
        f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"

    # load preemption results
    with open(results_name, "rb") as f:
        (failrate, delta_ratios, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)

    print(f"{failrate=}, {num_updates=}, {num_calibration=}, {deltas=}:")
    sound_alarms =  deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
    false_alarms = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
    unpreempted = deploy_failures[:,None,None] & (alarm_times==deploy_durations[:,None,None])
    print(f" unpreempted rates (delta x L):")
    print(unpreempted.mean(axis=0).round(3))
    print(f" false alarm rates (delta x L):")
    print(false_alarms.mean(axis=0).round(3))

    # pt.subplot(1,3,1)
    for d, (dr, delta) in enumerate(zip(delta_ratios, deltas)):
        pt.plot(np.arange(1, max_leadtime+1), false_alarms[:,d,:].mean(axis=0), 'x-', color=(dr,dr,1), label=f"false alarms (delta={delta})")
        pt.plot(np.arange(1, max_leadtime+1), unpreempted[:,d,:].mean(axis=0), 'o-', color=(1,dr,dr), label=f"unpreempted failures (delta={delta})")
        pt.plot(np.arange(1, max_leadtime+1), [delta]*max_leadtime, ':', color=(dr,)*3, label=f"delta={delta}")
    pt.xlabel("L")
    pt.ylabel("Rate")
    pt.legend()

    # pt.subplot(1,3,2)
    # for L in range(max_leadtime):
    #     quiet_time = alarm_times[~deploy_failures,L] / deploy_durations[~deploy_failures]
    #     pt.plot([L+1]*len(quiet_time), quiet_time, 'k.')
    # pt.xlim([-1, max_leadtime+1])
    # pt.xlabel("L")
    # pt.ylabel("Alarm-free portion of non-failures")

    # pt.subplot(1,3,3)
    # for L in range(max_leadtime):
    #     leadtime = deploy_durations[deploy_failures] - alarm_times[deploy_failures,L]
    #     pt.plot([L+1]*len(leadtime), leadtime, 'k.')
    # pt.xlim([-1, max_leadtime+1])
    # pt.xlabel("L")
    # pt.ylabel("Leadtime before failures")

    # pt.tight_layout()
    pt.show()

