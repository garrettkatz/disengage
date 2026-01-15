import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su
from lunar_lander_alive import perturb_action, failure_predicate, setup_mu

if __name__ == "__main__":

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000

    num_rollouts = 200 # 200 rollouts * 1000 timesteps = 200K is parity with original training
    gamma = .99 # discount factor for alive bonuses, tuned to be relatively sensitive within up to 100-step horizon
    blend = 0.001 # blend factor for target mu network
    learning_rate = 0.0001

    lead_time = 20
    num_calibration = 100
    num_repetitions = 100
    mu_basename = "ll_data/ll_alive"
    results_basename = "ll_data/ll_pa"
    checkpoints = [5000, 1000]
    delta = 0.05
    do_reps = True
    do_show = True

    if do_reps:

        # don't need grads for mu now
        tr.set_grad_enabled(False)
    
        # load environment and blackbox policy
        env, model = su.load(env_name, alg_name)
    
        # test each mu checkpoint
        failures = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        preempts = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        durations = np.empty((len(checkpoints), num_repetitions), dtype=int)
        preetimes = np.empty((len(checkpoints), num_repetitions), dtype=int)
        taus = np.empty((len(checkpoints), num_repetitions))
        for cp, checkpoint_num in enumerate(checkpoints):
    
            # load checkpoint
            mu = setup_mu()
            mu_state_dict = tr.load(f"{mu_basename}_r{num_rollouts}_lr{learning_rate}_g{gamma}_b{blend}_u{checkpoint_num}.pt", weights_only=True)
            mu.load_state_dict(mu_state_dict)

            # experimental repetitions to estimate preemption failure rate
            for rep in range(num_repetitions):
        
                # collect calibration data
                # z = np.empty(num_calibration)
                z = np.full(num_calibration, -np.inf)
                for episode in range(num_calibration):
            
                    # run an episode
                    failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)
            
                    # evaluate mu
                    features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
                    predictions = mu(features).squeeze()
    
                    # record z statistic
                    if failure:
                        if len(observations) <= lead_time:
                            z[episode] = predictions[0]
                        else:
                            z[episode] = predictions[:-lead_time].min()
                
                # set tau
                sort_idx = np.argsort(z)
                z = z[sort_idx]
                tau = z[int(np.ceil((1-delta)*(num_calibration+1)))]
            
                # deploy
                failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)
                features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
                predictions = mu(features).squeeze()
            
                # save results
                failures[cp, rep] = failure
                durations[cp, rep] = len(observations)
                preempts[cp, rep] = (predictions[:-1] < tau).any()
                # if preempts[cp, rep]: durations[cp, rep] = (predictions[:-1] > tau).to(int).argmax()
                if preempts[cp, rep]:
                    preetimes[cp, rep] = (predictions[:-1] < tau).to(int).argmax()
                else: preetimes[cp, rep] = -1
                taus[cp, rep] = tau
                np.savez(f"{results_basename}_L{lead_time}_{checkpoint_num}.npz", failures=failures[cp], preempts=preempts[cp], durations=durations[cp], taus=taus[cp])
                print(f"chkpt {checkpoint_num}, rep {rep} of {num_repetitions}: failure={failures[cp, rep]}, preempt={preempts[cp, rep]}, dur={durations[cp, rep]}, pretime={preetimes[cp, rep]}, tau={taus[cp, rep]}")

            print(f"chkpt {checkpoint_num}: {preempts[cp].mean()} preemption, {(failures & ~preempts)[cp].mean()} failure")

    if do_show:

        # get original failure rate
        (_, _, _, _, failrate) = tr.load(f"{mu_basename}_buffer.pt", weights_only=True)

        # get preemption data
        failures = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        preempts = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        durations = np.empty((len(checkpoints), num_repetitions), dtype=int)
        taus = np.empty((len(checkpoints), num_repetitions))

        for cp, checkpoint_num in enumerate(checkpoints):
            npz = np.load(f"{results_basename}_L{lead_time}_{checkpoint_num}.npz")
            failures[cp] = npz["failures"]
            preempts[cp] = npz["preempts"]
            durations[cp] = npz["durations"]
            taus[cp] = npz["taus"]
            print(f"chkpt {checkpoint_num}: {preempts[cp].mean():.3f} preemption, {(failures[cp] & ~preempts[cp]).mean():.3f} failure")

        import matplotlib.pyplot as pt

        fig = pt.figure(figsize=(6,3))

        pt.subplot(1,2,1)
        pt.plot(checkpoints, preempts.mean(axis=1), "go-", label="disengage")
        pt.plot(checkpoints, (failures & ~preempts).mean(axis=1), "rx-", label="failure")
        pt.plot(checkpoints, [delta]*len(checkpoints), 'k+:', label="delta")
        pt.plot(checkpoints, [1 - delta/failrate]*len(checkpoints), '^--', color=(.5,)*3, label="baseline")
        pt.xticks(checkpoints, rotation=45)
        pt.xlabel("Checkpoint")
        pt.ylabel("Rates")
        pt.legend()

        pt.subplot(1,2,2)
        for checkpoint_num, t in zip(checkpoints, taus):
            pt.plot(checkpoint_num + 100*np.random.randn(num_repetitions), t, 'k.')
        pt.xlabel("Checkpoint")
        pt.ylabel("$\\tau$")

        pt.tight_layout()
        pt.savefig(f"{results_basename}.eps")
        pt.show()


# import itertools as it
# import pickle as pk
# import numpy as np
# import torch as tr
# import sb_utils as su
# from cartpole_aux import perturb_action, failure_predicate, setup_mu

# if __name__ == "__main__":

#     do_reps = True
#     do_show = True

#     mu_basename = "cartpole_data/aux"
#     results_basename = "cartpole_data/preempt"
#     env_name = "CartPole-v1"
#     alg_name = "dqn"
#     num_timesteps = 499
#     num_calibration = 100
#     num_repetitions = 100
#     delta = 0.05

#     num_aux_repetitions = 3
#     lr = 1e-2
#     prediction_windows = [5, 27, 50]
#     checkpoints = [50, 100, 200, 500, 1000]

#     if do_reps:

#         # don't need grads for mu now
#         tr.set_grad_enabled(False)

#         for (H, aux_rep, chkpt) in it.product(prediction_windows, range(num_aux_repetitions), checkpoints):

#             # load environment and blackbox policy
#             env, model = su.load(env_name, alg_name)

#             # load checkpoint
#             mu = setup_mu()
#             mu.load_state_dict(tr.load(f"{mu_basename}_H{H}_lr{lr}_r{aux_rep}_u{chkpt}.pt", weights_only=True))

#             # allocate result data
#             failures = np.empty(num_repetitions, dtype=bool)
#             durations = np.empty(num_repetitions, dtype=int)
#             preempts = (num_timesteps+1) * np.ones(num_repetitions, dtype=int)
#             taus = np.empty(num_repetitions)
        
#             # experimental repetitions to estimate preemption failure rate
#             for rep in range(num_repetitions):
        
#                 # collect calibration data
#                 z = np.zeros(num_calibration) # 0 unless failure
#                 for episode in range(num_calibration):
            
#                     # run an episode
#                     failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)
            
#                     # evaluate mu
#                     features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
#                     probs = tr.sigmoid(mu(features).squeeze())
    
#                     # record z statistic
#                     if failure:
#                         if len(observations) <= H:
#                             z[episode] = (1 - probs[0])
#                         else:
#                             z[episode] = (1 - probs[:-H]).min()
                
#                 # set tau
#                 sort_idx = np.argsort(z)
#                 z = z[sort_idx]
#                 h = z[int(np.ceil((1-delta)*(num_calibration+1)))]
#                 tau = 1 - h
            
#                 # deploy
#                 failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)
#                 features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
#                 probs = tr.sigmoid(mu(features).squeeze())
            
#                 # save results
#                 failures[rep] = failure
#                 durations[rep] = len(observations)
#                 if (probs > tau).any(): preempts[rep] = (probs > tau).to(int).argmax()
#                 taus[rep] = tau

#                 print(f"{H=}, {lr=}, {aux_rep=}, {chkpt=}, {rep=}: fail={failures[rep]}, dur={durations[rep]}, preempt={preempts[rep]}, tau={taus[rep]}")

#             np.savez(f"{results_basename}_H{H}_lr{lr}_r{aux_rep}_u{chkpt}.npz",
#                 failures=failures, durations=durations, preempts=preempts, taus=taus)

#             print(f"{H=}, {lr=}, {aux_rep=}, {chkpt=}: {(preempts < durations).mean()} preemption, {(failures & (preempts > (durations-H))).mean()} failure")

#     if do_show:

#         import matplotlib.pyplot as pt

#         # get original failure rate
#         failrates = []
#         for (H, aux_rep) in it.product(prediction_windows, range(num_aux_repetitions)):
#             with open(f"{mu_basename}_H{H}_lr{lr}_r{aux_rep}.pkl","rb") as f:
#                 (failures, _, _, _, _, _) = pk.load(f)
#             failrates.append(failures)
#         failrate = np.mean(failrates) # justified given that all configurations use the same number of episodes

#         # organize plot by prediction horizon
#         fig = pt.figure(figsize=(6,3))
#         sp = 1
#         for (row, H) in enumerate(prediction_windows):

#             # allocate result data
#             failures = np.empty((num_aux_repetitions, len(checkpoints), num_repetitions), dtype=bool)
#             durations = np.empty((num_aux_repetitions, len(checkpoints), num_repetitions), dtype=int)
#             preempts = np.empty((num_aux_repetitions, len(checkpoints), num_repetitions), dtype=int)
#             taus = np.empty((num_aux_repetitions, len(checkpoints), num_repetitions))

#             for (aux_rep, (c, chkpt)) in it.product(range(num_aux_repetitions), enumerate(checkpoints)):
#                 npz = np.load(f"{results_basename}_H{H}_lr{lr}_r{aux_rep}_u{chkpt}.npz")
#                 failures[aux_rep, c] = npz["failures"]
#                 preempts[aux_rep, c] = npz["preempts"]
#                 durations[aux_rep, c] = npz["durations"]
#                 taus[aux_rep, c] = npz["taus"]
#                 print(f"{H=}, {lr=}, {aux_rep=}, {chkpt=}: {(preempts[aux_rep,c] < durations[aux_rep,c]).mean()} preemption, {(failures[aux_rep,c] & (preempts[aux_rep,c] > (durations[aux_rep,c]-H))).mean()} failure")

#             # preemption rates
#             pt.subplot(len(prediction_windows),2,sp)
#             sp += 1

#             pt.plot(checkpoints, (preempts < durations).mean(axis=2).T, "-", color=(.8,.8,1))
#             pt.plot(checkpoints, (preempts < durations).mean(axis=(0,2)), "bo-", label="preemption")

#             pt.plot(checkpoints, (failures & (preempts > durations-H)).mean(axis=2).T, "-", color=(1,.8,.8))
#             pt.plot(checkpoints, (failures & (preempts > durations-H)).mean(axis=(0,2)), "rx-", label="failure")

#             pt.plot(checkpoints, [delta]*len(checkpoints), 'k:', label="delta")
#             pt.plot(checkpoints, [1 - delta/failrate]*len(checkpoints), '--', color=(.5,)*3, label="baseline")

#             pt.xticks(checkpoints, rotation=45)
#             pt.xlabel("Checkpoint")
#             pt.ylabel(f"{H=}")
#             pt.legend()
#             if row == 0: pt.title("Rates")

#             # preemption times
#             pt.subplot(len(prediction_windows),2,sp)
#             sp += 1

#             pt.plot(durations[:,-1,:].flatten(), preempts[:,-1,:].flatten(), "k.")
#             pt.xlabel("Episode Duration")
#             pt.ylabel("Preemption time")
#             if row == 0: pt.title("Timing")

#         pt.tight_layout()
#         pt.savefig(f"{results_basename}.eps")
#         pt.show()




