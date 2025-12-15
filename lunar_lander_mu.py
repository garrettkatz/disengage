from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su

def failure_predicate(env, obs, reward, done, infos):
    return (reward[0] == -100)

def setup_mu():
    # MLP - same architecture as critic
    num_hidden = 256
    return tr.nn.Sequential(
        tr.nn.Linear(in_features=8, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )

if __name__ == "__main__":

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    prediction_window = 25
    resume = True
    do_training = True
    do_show = True
    num_updates = 2000
    batch_size = 16
    learning_rate = 0.0001
    report_period = 1
    checkpoint_period = 100
    basename = "ll_mu"

    mu = setup_mu()
    loss_fn = tr.nn.BCEWithLogitsLoss()
    optimizer = tr.optim.Adam(mu.parameters(), lr=learning_rate) # no overfit risk when always sampling new data?

    if resume:

        resume_data = tr.load(f"{basename}_H{prediction_window}.pt", weights_only=True)
        mu.load_state_dict(resume_data['mu'])
        optimizer.load_state_dict(resume_data['opt'])
        with open(f"{basename}_H{prediction_window}.pkl","rb") as f:
            (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)

        start_update = len(losses)
        num_updates += start_update

    else:

        start_update = 0
        failures = []
        durations = []
        probs = []
        accuracies = [] # full curve
        gradmaxs = []
        losses = []

    if do_training:

        env, model = su.load(env_name, alg_name)
        start_loop = perf_counter()
        for update in range(start_update, num_updates):
    
            # collect a batch
            batch_obs = []
            batch_lab = []
            for example in range(batch_size):
    
                # run an episode
                failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate)
    
                # save failure indicators and times
                failures.append(failure)
                durations.append(len(observations))
    
                # extract example input
                if failure:
                    # within prediction window of failure
                    lo = max(0, len(observations)-prediction_window)
                    idx = np.random.choice(np.arange(lo, len(observations)))
                    obs = observations[idx]
                else:
                    # uniformly over time
                    obs = observations[np.random.randint(len(observations))]
    
                # update batch
                batch_obs.append(obs)
                batch_lab.append(failure)
    
            # forward pass
            features = tr.tensor(np.concatenate(batch_obs, axis=0)).to(tr.float32)
            labels = tr.tensor(batch_lab).to(tr.float32)
            logits = mu(features).squeeze()
            loss = loss_fn(logits, labels)
            correct = (((labels > .5) & (logits > 0)) | ((labels < .5) & (logits <= 0))).to(tr.float).mean()
    
            # backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # update learning curves
            losses.append(loss.item())
            gradmaxs.append(max([p.grad.abs().max() for p in mu.parameters()]))
            probs.append(tr.sigmoid(logits).mean().item())
            accuracies.append(correct)
    
            # progress report
            if update % report_period == 0:
                failrate = np.mean(failures)
                failtime = np.mean([d for (d,f) in zip(durations, failures) if f])
                print(f"{update=} of {num_updates}: {failrate=:.3f}, {failtime=:.1f}, {loss=:.3f}, {correct=:.3f}, probs={probs[-1]:.3f}, |grad|<={gradmaxs[-1]:.5f}")


                # save latest model and optimizer for resuming later, along with metrics so far
                tr.save({
                    'mu': mu.state_dict(),
                    'opt': optimizer.state_dict(),
                }, f"{basename}_H{prediction_window}.pt")
                with open(f"{basename}_H{prediction_window}.pkl","wb") as f:
                    pk.dump((failures, durations, losses, accuracies, probs, gradmaxs), f)

            if (update+1) % checkpoint_period == 0:
                tr.save(mu.state_dict(), f"{basename}_H{prediction_window}_{update+1}.pt")
                with open(f"{basename}_H{prediction_window}_{update+1}.pkl","wb") as f:
                    pk.dump((failures, durations, losses, accuracies, probs, gradmaxs), f)

        total_time = perf_counter() - start_loop
        print(f"{total_time:.3f} seconds, {total_time / (num_updates - start_update):.3f} per update")

    if do_show:

        with open(f"{basename}_H{prediction_window}.pkl","rb") as f:
            (failures, durations, losses, accuracies, probs, gradmaxs) = pk.load(f)
    
        import matplotlib.pyplot as pt
        window = 50
        fig = pt.figure(figsize=(6,3))

        pt.subplot(1,3,1)
        pt.plot(losses, '-', c=(.8,.8,.8))
        pt.plot(np.arange(0, len(losses), window), np.array(losses).reshape(-1,window).mean(axis=1), 'k-')
        pt.ylabel("Metric")
        pt.title("Loss")

        pt.subplot(1,3,2)
        pt.plot(accuracies, '-', c=(.8,.8,.8))
        pt.plot(np.arange(0, len(accuracies), window), np.array(accuracies).reshape(-1,window).mean(axis=1), 'k-')
        pt.title("Accuracy")

        pt.subplot(1,3,3)
        pt.plot(gradmaxs, '-', c=(.8,.8,.8))
        pt.plot(np.arange(0, len(gradmaxs), window), np.array(gradmaxs).reshape(-1,window).mean(axis=1), 'k-')
        pt.title("Gradient norm")

        fig.supxlabel("Parameter update")
        pt.tight_layout()
        pt.savefig(f"{basename}.eps")
        pt.show()


        # pt.subplot(4,1,2)
        # pt.plot(accuracies)
        # pt.title("accuracies")
        # pt.subplot(4,1,3)
        # pt.plot(probs)
        # pt.title("probs")
        # pt.subplot(4,1,4)
        # pt.plot(gradmaxs)
        # pt.title("gradmaxs")
        # pt.show()
