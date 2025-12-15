import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su
from lunar_lander_dense_mu import failure_predicate, setup_mu

if __name__ == "__main__":

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    prediction_window = 50
    num_calibration = 100
    num_repetitions = 100
    mu_basename = "ll_dense_mu"
    results_basename = "lldp"
    checkpoints = [100, 700, 1300, 2000]
    delta = 0.05
    do_reps = False
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
        taus = np.empty((len(checkpoints), num_repetitions))
        for cp, checkpoint_num in enumerate(checkpoints):
    
            # load checkpoint
            mu = setup_mu()
            mu.load_state_dict(tr.load(f"{mu_basename}_H{prediction_window}_{checkpoint_num}.pt", weights_only=True))
    
            # experimental repetitions to estimate preemption failure rate
            for rep in range(num_repetitions):
        
                # collect calibration data
                z = np.empty(num_calibration)
                for episode in range(num_calibration):
            
                    # run an episode
                    failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate)
            
                    # evaluate mu
                    features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
                    probs = tr.sigmoid(mu(features).squeeze())
    
                    # record z statistic
                    z[episode] = failure * (1 - probs).min()
                
                # set tau
                sort_idx = np.argsort(z)
                z = z[sort_idx]
                h = z[int(np.ceil((1-delta)*(num_calibration+1)))]
                tau = 1 - h
            
                # deploy
                failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate)
                features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
                probs = tr.sigmoid(mu(features).squeeze())
            
                # save results
                failures[cp, rep] = failure
                durations[cp, rep] = len(observations)
                preempts[cp, rep] = (probs[:-1] > tau).any()
                if preempts[cp, rep]: durations[cp, rep] = (probs[:-1] > tau).to(int).argmax()
                taus[cp, rep] = tau
                np.savez(f"{results_basename}_H{prediction_window}_{checkpoint_num}.npz", failures=failures[cp], preempts=preempts[cp], durations=durations[cp], taus=taus[cp])
                print(f"chkpt {checkpoint_num}, rep {rep} of {num_repetitions}: failure={failures[cp, rep]}, preempt={preempts[cp, rep]}, dur={durations[cp, rep]}, tau={taus[cp, rep]}")

            print(f"chkpt {checkpoint_num}: {preempts[cp].mean()} preemption, {(failures & ~preempts)[cp].mean()} failure")

    if do_show:

        # get original failure rate
        with open(f"{mu_basename}_H{prediction_window}.pkl","rb") as f:
            (failures, _, _, _, _, _) = pk.load(f)
        failrate = np.mean(failures)
        print(f"original failrate {failrate}")

        # get preemption data
        failures = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        preempts = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        durations = np.empty((len(checkpoints), num_repetitions), dtype=int)
        taus = np.empty((len(checkpoints), num_repetitions))

        for cp, checkpoint_num in enumerate(checkpoints):
            npz = np.load(f"{results_basename}_H{prediction_window}_{checkpoint_num}.npz")
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
        pt.plot(checkpoints, [delta]*len(checkpoints), 'k:', label="delta")
        pt.xticks(checkpoints, rotation=45)
        pt.xlabel("Checkpoint")
        pt.ylabel("Rates")
        pt.legend()

        pt.subplot(1,2,2)
        for checkpoint_num, t in zip(checkpoints, taus):
            pt.plot(checkpoint_num + 5*np.random.randn(num_repetitions), t, 'k.')
        pt.xlabel("Checkpoint")
        pt.ylabel("$\\tau$")

        pt.tight_layout()
        pt.savefig(f"{results_basename}.eps")
        pt.show()


