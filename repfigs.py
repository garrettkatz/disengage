import numpy as np
import pickle as pk
import matplotlib.pyplot as pt

# failhists
if False:

    # get original failure rate
    with open(f"ll_mu_H25.pkl","rb") as f:
        (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)
    
    failures = np.array(failures)
    durations = np.array(durations)
    
    fig = pt.figure(figsize=(12,4))
    
    pt.subplot(1,2,1)
    pt.hist(durations[failures], label="failure", alpha=.5, bins=50)
    pt.hist(durations[~failures], label="success", alpha=.5, bins=50)
    pt.legend()
    # pt.xlabel("Episode Duration")
    # pt.ylabel("Frequency")
    pt.yscale("log")
    pt.title(f"Lunar Lander, failrate = {failures.mean():.3f}")
    
    # get original failure rate
    with open(f"ruins_mu_H8.pkl","rb") as f:
        (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)
    
    failures = np.array(failures)
    durations = np.array(durations)
    
    pt.subplot(1,2,2)
    pt.hist(durations[failures], label="failure", alpha=.5, bins=50)
    pt.hist(durations[~failures], label="success", alpha=.5, bins=50)
    pt.legend()
    # pt.xlabel("Episode Duration")
    # pt.ylabel("Frequency")
    pt.yscale("log")
    pt.title(f"Ruins, failrate = {failures.mean():.3f}")
    
    fig.supxlabel("Episode Duration")
    fig.supylabel("Count")
    
    pt.tight_layout()
    pt.savefig("failhists.png")
    pt.show()
    
# mu curves
if False:
    
    fig = pt.figure(figsize=(12,6))

    with open(f"ll_dense_mu_H50.pkl","rb") as f:
        (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)
    
    window = 10
    
    pt.subplot(2,3,1)
    pt.plot(losses, 'b-', alpha=.1)
    pt.plot(np.arange(0, len(losses), window), np.array(losses).reshape(-1,window).mean(axis=1), 'b-')
    pt.ylabel("Loss")
    pt.ylim([0, 1.4])
    pt.title("Lunar Lander (H=50)")
    
    pt.subplot(2,3,4)
    pt.plot(accuracies, 'b-', alpha=.1)
    pt.plot(np.arange(0, len(accuracies), window), np.array(accuracies).reshape(-1,window).mean(axis=1), 'b-', label="correlated")
    pt.ylim([.3, 1.1])
    pt.ylabel("Accuracy")

    with open(f"ll_mu_H50.pkl","rb") as f:
        (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)

    pt.subplot(2,3,1)
    pt.plot(losses, 'r-', alpha=.1)
    pt.plot(np.arange(0, len(losses), window), np.array(losses).reshape(-1,window).mean(axis=1), 'r-')
    pt.ylim([0, 1.4])
    
    pt.subplot(2,3,4)
    pt.plot(accuracies, 'r-', alpha=.1)
    pt.plot(np.arange(0, len(accuracies), window), np.array(accuracies).reshape(-1,window).mean(axis=1), 'r-', label="decorrelated")
    pt.legend()
    pt.ylim([.3, 1.1])   

    with open(f"ll_mu_H25.pkl","rb") as f:
        (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)

    window = 50
    
    pt.subplot(2,3,2)
    pt.plot(losses, '-', c=(.8,.8,.8))
    pt.plot(np.arange(0, len(losses), window), np.array(losses).reshape(-1,window).mean(axis=1), 'k-')
    pt.ylim([0, 1.4])
    pt.title("Lunar Lander (H=25)")
    
    pt.subplot(2,3,5)
    pt.plot(accuracies, '-', c=(.8,.8,.8))
    pt.plot(np.arange(0, len(accuracies), window), np.array(accuracies).reshape(-1,window).mean(axis=1), 'k-')
    pt.ylim([.3, 1.1])
    
    with open(f"ruins_mu_H8.pkl","rb") as f:
        (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)
    
    window = 50
    
    pt.subplot(2,3,3)
    pt.plot(losses, '-', c=(.8,.8,.8))
    pt.plot(np.arange(0, len(losses), window), np.array(losses).reshape(-1,window).mean(axis=1), 'k-')
    # pt.ylabel("Loss")
    pt.ylim([0, 1.4])
    pt.title("Ruins (H=8)")
    
    pt.subplot(2,3,6)
    pt.plot(accuracies, '-', c=(.8,.8,.8))
    pt.plot(np.arange(0, len(accuracies), window), np.array(accuracies).reshape(-1,window).mean(axis=1), 'k-')
    # pt.ylabel("Accuracy")    
    pt.ylim([.3, 1.1])

    fig.supxlabel("Parameter update")
    pt.tight_layout()
    pt.savefig(f"mu_curves.png")
    pt.show()

# preemption curves
if True:

        fig = pt.figure(figsize=(10,5))

        # ll dense
        delta = 0.05
        num_repetitions = 100
        checkpoints = [100, 700, 1300, 2000]

        # get original failure rate
        with open(f"ll_dense_mu_H50.pkl","rb") as f: (failures, _, _, _, _, _) = pk.load(f)
        failrate = np.mean(failures)
        print(f"original failrate {failrate}")

        # get preemption data
        failures = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        preempts = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        durations = np.empty((len(checkpoints), num_repetitions), dtype=int)
        taus = np.empty((len(checkpoints), num_repetitions))

        for cp, checkpoint_num in enumerate(checkpoints):
            npz = np.load(f"lldp_H50_{checkpoint_num}.npz")
            failures[cp] = npz["failures"]
            preempts[cp] = npz["preempts"]
            durations[cp] = npz["durations"]
            taus[cp] = npz["taus"]
            print(f"chkpt {checkpoint_num}: {preempts[cp].mean():.3f} preemption, {(failures[cp] & ~preempts[cp]).mean():.3f} failure")

        pt.subplot(1,3,1)
        pt.plot(checkpoints, preempts.mean(axis=1), "bo-", label="disengagement")
        pt.plot(checkpoints, (failures & ~preempts).mean(axis=1), "rx-", label="unpreempted failure")
        pt.plot(checkpoints, [failrate]*len(checkpoints), 'k--', label="original failure")
        pt.plot(checkpoints, [delta]*len(checkpoints), 'k:', label="delta")
        pt.xticks(checkpoints, rotation=45)
        # pt.xlabel("Checkpoint")
        # pt.ylabel("Rates")
        pt.ylim([0, .9])
        pt.title("Lunar Lander (H=50)")


        # ll
        delta = 0.05
        num_repetitions = 100
        checkpoints = [100, 1500, 3000, 4500, 6000, 7500]

        # get original failure rate
        with open(f"ll_mu_H25.pkl","rb") as f: (failures, _, _, _, _, _) = pk.load(f)
        failrate = np.mean(failures)
        print(f"original failrate {failrate}")

        # get preemption data
        failures = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        preempts = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        durations = np.empty((len(checkpoints), num_repetitions), dtype=int)
        taus = np.empty((len(checkpoints), num_repetitions))

        for cp, checkpoint_num in enumerate(checkpoints):
            npz = np.load(f"llp_H25_{checkpoint_num}.npz")
            failures[cp] = npz["failures"]
            preempts[cp] = npz["preempts"]
            durations[cp] = npz["durations"]
            taus[cp] = npz["taus"]
            print(f"chkpt {checkpoint_num}: {preempts[cp].mean():.3f} preemption, {(failures[cp] & ~preempts[cp]).mean():.3f} failure")

        pt.subplot(1,3,2)
        pt.plot(checkpoints, preempts.mean(axis=1), "bo-", label="disengagement")
        pt.plot(checkpoints, (failures & ~preempts).mean(axis=1), "rx-", label="unpreempted failure")
        pt.plot(checkpoints, [failrate]*len(checkpoints), 'k--', label="original failure")
        pt.plot(checkpoints, [delta]*len(checkpoints), 'k:', label="delta")
        pt.xticks(checkpoints, rotation=45)
        # pt.xlabel("Checkpoint")
        # pt.ylabel("Rates")
        pt.ylim([0, .9])
        pt.legend()
        pt.title("Lunar Lander (H=25)")

        # pt.subplot(2,2,3)
        # for checkpoint_num, t in zip(checkpoints, taus):
        #     pt.plot(checkpoint_num + 5*np.random.randn(num_repetitions), t, 'k.')
        # # pt.xlabel("Checkpoint")
        # pt.ylabel("$\\tau$")

        # ruins
        delta = 0.05
        num_repetitions = 100
        checkpoints = [100, 1700, 3400, 5000, 6700, 8300, 10_000]

        # get original failure rate
        with open(f"ruins_mu_H8.pkl","rb") as f: (failures, _, _, _, _, _) = pk.load(f)
        failrate = np.mean(failures)
        print(f"original failrate {failrate}")

        # get preemption data
        failures = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        preempts = np.empty((len(checkpoints), num_repetitions), dtype=bool)
        durations = np.empty((len(checkpoints), num_repetitions), dtype=int)
        taus = np.empty((len(checkpoints), num_repetitions))

        for cp, checkpoint_num in enumerate(checkpoints):
            npz = np.load(f"ruinsp_H8_{checkpoint_num}.npz")
            failures[cp] = npz["failures"]
            preempts[cp] = npz["preempts"]
            durations[cp] = npz["durations"]
            taus[cp] = npz["taus"]
            print(f"chkpt {checkpoint_num}: {preempts[cp].mean():.3f} preemption, {(failures[cp] & ~preempts[cp]).mean():.3f} failure")

        pt.subplot(1,3,3)
        pt.plot(checkpoints, preempts.mean(axis=1), "bo-", label="disengagement")
        pt.plot(checkpoints, (failures & ~preempts).mean(axis=1), "rx-", label="unpreempted failure")
        pt.plot(checkpoints, [failrate]*len(checkpoints), 'k--', label="original failure")
        pt.plot(checkpoints, [delta]*len(checkpoints), 'k:', label="delta")
        pt.xticks(checkpoints, rotation=45)
        # pt.xlabel("Checkpoint")
        # pt.ylabel("Rates")
        pt.ylim([0, .9])
        pt.title("Ruins (H=8)")

        # pt.subplot(2,2,4)
        # for checkpoint_num, t in zip(checkpoints, taus):
        #     pt.plot(checkpoint_num + 5*np.random.randn(num_repetitions), t, 'k.')
        # # pt.xlabel("Checkpoint")
        # # pt.ylabel("$\\tau$")

        fig.supxlabel("Checkpoint")
        fig.supylabel("Rates")
        pt.tight_layout()
        pt.savefig(f"preempts.png")
        pt.show()

