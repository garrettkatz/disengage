"""
Train regressor on first-step fail rate
"""
import numpy as np
from time import perf_counter
from planar import PlanarEnv
from planar_features import FeatureExtractor
from handcode_planar import Policy
import pickle as pk
import matplotlib.pyplot as pt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def inv_sigmoid(x): return np.log(x) - np.log(1-x)

if __name__ == "__main__":

    do_rollouts = False
    do_train = False
    do_cal = False
    do_deploy = False
    do_reps = False
    do_show = True

    # setup experiment parameters
    num_allies = 2
    num_adversaries = 2
    view_angle = np.pi/16
    step_size = np.array([0.05, 0.05, .1]) # larger rotational motion

    num_reps = 1
    num_timesteps = 100
    num_samples = 4096 # number of samples for fitting regressor and calibration
    num_rollouts = 256 # number of rollouts to estimate state-conditioned failure rate
    num_train = 2048 # number of samples for fitting only
    num_deployments = 10_000 # number of samples to evaluate deployment

    # num_samples = 600 # number of samples for fitting regressor and calibration
    # num_rollouts = 32 # number of rollouts to estimate state-conditioned failure rate
    # num_train = 512 # number of samples for fitting only
    # num_deployments = 10_000 # number of samples to evaluate deployment

    confidence = 0.05 # confidence for Hoeffding interval
    acceptable = 0.1 # acceptable probability of failure at deployment

    # setup environment, policy, feature extraction
    env = PlanarEnv(
        num_allies=num_allies,
        num_adversaries=num_adversaries,
        step_size=step_size,
        view_angle=view_angle,
    )
    policy = Policy(env)
    # get_features = FeatureExtractor(env) # linear
    get_features = FeatureExtractor(env, kernel="quadratic")
    # get_features = FeatureExtractor(env, kernel="cubic") # good fit but overfit

    # set up hoeffding confidence interval
    hoeffding = np.sqrt(np.log(confidence/2) / (-2*num_rollouts))


    taus, deltas, thresholds, failrates, disrates, catrates, cathoeffs, marginals = [], [], [], [], [], [], [], []
    start_time = perf_counter()
    for rep in range(num_reps if do_reps else 0):

        # train regressor
        if do_rollouts:
    
            failures = np.zeros((num_samples, num_rollouts), dtype=bool)
            observation, _ = env.reset(batch_size=num_samples, upperhand=True)
            init_states = env.state
            for r in range(num_rollouts):
                env.state = init_states
                for t in range(num_timesteps):
                    _, viz = env._team_reward(env.state.adversaries, env.state.allies)
                    failures[:,r] = failures[:,r] | viz.any(axis=(-2,-1))    
                    action = policy(observation)
                    observation, _, _, _, _ = env.step(action)
                if (r % (num_rollouts//10)) == 0:
                    print(f"rollout {r} of {num_rollouts}: cross-sample failure rate = {failures[:,r].mean()}, hoeffding interval = {hoeffding}")
    
            failure_rates = failures.mean(axis=1)
            print(f"marginal failure rate stats: {failure_rates.mean()} +/- {failure_rates.std()}")
    
            with open(f"fsr_rollout_{rep}.pkl", "wb") as f:
                pk.dump((init_states, failures), f)
    
        with open(f"fsr_rollout_{rep}.pkl", "rb") as f:
            (init_states, failures) = pk.load(f)
    
        if do_train:
            observations = env.get_observation(init_states)
            features, _ = get_features(observations)
            labels = failures.mean(axis=1)
            print(f"{num_train} examples, {features.shape[-1]} features")
    
            # linear regression
            # regressor = np.linalg.lstsq(features, labels, rcond=None)[0]
            # predictions = features @ regressor
    
            targets = np.where(labels == 0, 1/(2*num_rollouts), labels)
            regressor = np.linalg.lstsq(features[:num_train], inv_sigmoid(targets[:num_train]), rcond=None)[0]
            predictions = sigmoid(features @ regressor)
    
            errors = np.fabs(predictions - labels)
    
            with open(f"fsr_train_{rep}.pkl", "wb") as f:
                pk.dump((init_states, failures, regressor, predictions, errors), f)
    
        with open(f"fsr_train_{rep}.pkl", "rb") as f:
            (init_states, failures, regressor, predictions, errors) = pk.load(f)
    
        if do_cal:
            calib = errors[num_train:]
            calib = np.sort(calib)
            quant = 1 - np.arange(len(calib))/(len(calib)+1)
    
            opt = np.argmin(quant + calib)
            # if do_show:
            #     pt.plot(quant, calib, 'k.-', label="threshold")
            #     pt.plot(quant, quant+calib, 'b.-', label="threshold + prob")
            #     pt.xlabel("Pr(error > threshold)")
            #     pt.ylabel("error threshold")
            #     pt.legend()
            #     pt.show()

            threshold = calib[opt]
            delta = quant[opt]
            tau = max(0, acceptable - (threshold + delta))

            print(f"Rep {rep} of {num_reps}:")    
            print(f"optimal delta+thresh = {quant[opt]:.3f}+{calib[opt]:.3f}={quant[opt]+calib[opt]:.3f}")
            print(f"tau = {tau:.3f}, overall catastrophe rate = {tau+delta+threshold:.3f} <= {acceptable:.3f}")

            taus.append(tau)
            deltas.append(delta)
            thresholds.append(threshold)
    
        if do_deploy:
            assert do_cal
    
            observations, _ = env.reset(batch_size=num_deployments, upperhand=True)
            features, _ = get_features(observations)
    
            disengage = (sigmoid(features @ regressor) >= tau)
            catastrophe = np.zeros(num_deployments, dtype=bool)
    
            for t in range(num_timesteps):
                _, viz = env._team_reward(env.state.adversaries, env.state.allies)
                catastrophe = catastrophe | viz.any(axis=(-2,-1))    
                action = policy(observations)
                observations, _, _, _, _ = env.step(action)
    
            deploy_hoeffding = np.sqrt(np.log(.999/2) / (-2*num_deployments))
    
            print(f"{disengage.mean():.3f} disengage rate, {(~disengage & catastrophe).sum() / num_deployments:.3f} failed deployment rate +/- {deploy_hoeffding:.5f} hoeffding interval")

            failrates.append(catastrophe.mean())
            disrates.append(disengage.mean())
            # catrates.append((~disengage & catastrophe).sum() / num_deployments)
            catrates.append((~disengage & catastrophe).mean())
            cathoeffs.append(deploy_hoeffding)
            marginals.append(~disengage[0] & catastrophe[0])

        print(f"rep {rep} done after {perf_counter()-start_time:.3f}s")

        with open(f"fsr_results.pkl", "wb") as f:
            pk.dump((taus, deltas, thresholds, failrates, disrates, catrates, cathoeffs, marginals), f)
    
    with open(f"fsr_results.pkl", "rb") as f:
        (taus, deltas, thresholds, failrates, disrates, catrates, cathoeffs, marginals) = pk.load(f)

    with open(f"fsr_train_0.pkl", "rb") as f:
        (init_states, failures, regressor, predictions, errors) = pk.load(f)

    print("fail rates, dis rates, catastrophe rates, + intervals, =")
    print(np.array(failrates))
    print(np.array(disrates))
    print(np.array(catrates))
    print(np.array(cathoeffs))
    print(np.array(catrates) + np.array(cathoeffs))
    print(f"vs acceptable = {acceptable}")

    print(f"marginal fail prob = {np.mean(marginals)}")

    if do_show:

        # sample tau selection
        calib = errors[num_train:]
        calib = np.sort(calib)
        quant = 1 - np.arange(len(calib))/(len(calib)+1)
        opt = np.argmin(quant + calib)

        pt.figure(figsize=(6,3), constrained_layout=True)
        pt.plot(quant, calib, 'r-', label="$h_0$")
        pt.plot(quant, quant+calib, 'b-', label="$h_0 + \\alpha_0$") # 
        pt.plot(quant, np.full(len(quant), quant[opt]+calib[opt]), 'k--', label="$\\delta - \\tau_0$") # 
        pt.xlabel("Quantile")
        pt.ylabel("Thresholds")
        pt.title(f"$h_0^* + \\alpha_0^* = {calib[opt]:.3f} + {quant[opt]:.3f} = {calib[opt] + quant[opt]:.3f}$")
        pt.legend()
        pt.savefig("calib.eps")
        pt.show()


        sorter = np.argsort(catrates)
        catrates = np.array(catrates)[sorter]
        cathoeffs = np.array(cathoeffs)[sorter]
        disrates = np.array(disrates)[sorter]

        pt.figure(figsize=(10,4), constrained_layout=True)
        pt.bar(np.arange(num_reps), catrates, color='k', label='$\\neg D_0 \wedge F_{>0}$')
        pt.bar(np.arange(num_reps), cathoeffs, bottom=catrates, color='r', label='$\\neg D_0 \wedge F_{>0}$ + .999 confidence interval')
        pt.bar(np.arange(num_reps), disrates, bottom=catrates+cathoeffs, color='b', label='$D_0$')
        pt.plot([0, num_reps], [acceptable]*2, 'k--', label='$\delta$')
        pt.legend(loc="upper right", framealpha=1)
        pt.xlabel("Sorted repetitions")
        pt.ylabel("Error Rates")
        pt.savefig("rep_results.eps")
        pt.show()

        # # sort by estimated failure rate
        # failure_rates = failures.mean(axis=1)
        # idx = np.argsort(failure_rates)

        # # empirical discrepancy between two independent rollout samples (split the big sample)
        # failure_rates_A = failures[:,:num_rollouts//2].mean(axis=1)
        # failure_rates_B = failures[:,num_rollouts//2:].mean(axis=1)

        # pt.figure(figsize=(12,4), constrained_layout=True)

        # # x = 1.01**np.arange(num_samples)
        # x = np.arange(num_samples)

        # ylo = failure_rates - hoeffding
        # yhi = failure_rates + hoeffding
        # pt.fill_between(x, ylo[idx], yhi[idx], color=(.75,)*3)

        # pt.plot(x, failure_rates_A[idx], 'b.', label="Subsample A")
        # pt.plot(x, failure_rates_B[idx], 'r.', label="Subsample B")
        # pt.plot(x, predictions[idx], 'g.-', label="predictions")
        # pt.plot(x, failure_rates[idx], 'k-', label="Estimated failure rate")
        # pt.xlabel("Ordered initial state samples")
        # pt.ylabel("Estimate of $\\text{Pr}(F_{\\geq 1})$")

        # # pt.xticks(x[::10], np.arange(0,num_samples,10))
        # # pt.xscale("function", functions=(lambda x: 1.01**x, lambda x: np.log(x)/np.log(1.01)))

        # # pt.ylim([.2, .38])

        # # pt.figure(figsize=(5,3), constrained_layout=True)
        # # pt.plot(overrate, 'k-', label='failed by $t$')
        # # # pt.plot(failrate, 'b--', label='fail at t')
        # # pt.xlabel("Timestep $t$")
        # # pt.ylabel("Estimate of $\\text{Pr}(F_{\\leq t})$")
        # # pt.legend()

        # pt.savefig("fsr.eps")
        # pt.show()


