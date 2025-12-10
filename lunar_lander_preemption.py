import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su
from lunar_lander_mu import failure_predicate, setup_mu

if __name__ == "__main__":

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    num_calibration = 100
    num_repetitions = 100
    basename = "ll_mu"
    checkpoints = list(range(100, 901, 100))
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
            mu.load_state_dict(tr.load(f"{basename}_{checkpoint_num}.pt", weights_only=True))
    
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
                if preempts[cp, rep]: durations[cp, rep] = (probs[:-1] > tau).argmin()
                taus[cp, rep] = tau
                np.savez("llp.npz", failures=failures, preempts=preempts, durations=durations, taus=taus)
                print(f"chkpt {checkpoint_num}, rep {rep} of {num_repetitions}: failure={failures[cp, rep]}, preempt={preempts[cp, rep]}, dur={durations[cp, rep]}, tau={taus[cp, rep]}")

            print(f"chkpt {checkpoint_num}: {preempts[cp].mean()} preemption, {(failures & ~preempts)[cp].mean()} failure")

    if do_show:

        # get original failure rate
        with open(f"{basename}.pkl","rb") as f:
            (failures, _, _, _, _, _) = pk.load(f)
        failrate = np.mean(failures)

        # get preemption data
        npz = np.load("llp.npz")
        failures = npz["failures"]
        preempts = npz["preempts"]
        taus = npz["taus"]

        import matplotlib.pyplot as pt

        for cp, checkpoint_num in enumerate(checkpoints):
            print(f"chkpt {checkpoint_num}: {preempts[cp].mean()} preemption [vs {delta / failrate}], {(failures & ~preempts)[cp].mean()} failure")

        pt.subplot(1,2,1)
        pt.plot(checkpoints, preempts.mean(axis=1), label="preemption")
        pt.plot(checkpoints, (failures & ~preempts).mean(axis=1), label="failure")
        pt.xticks(checkpoints)
        pt.ylabel("Rates")
        pt.legend()

        pt.subplot(1,2,2)
        for t in taus:
            pt.hist(t, alpha=.5)
        pt.xlabel("tau")
        pt.ylabel("count")
        pt.show()
