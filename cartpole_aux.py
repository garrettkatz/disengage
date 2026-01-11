from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su

def perturb_action(a):
    if np.random.rand() < .05: a = 1 - a # 5% chance of flipping 0|1 action
    return a

def failure_predicate(env, obs, reward, done, infos):
    # early termination signals failure (large angle or horizontal coordinate)
    # need to use 499 timesteps below so time-limit reached does not count as failure
    return done[0]

def setup_mu():
    # MLP - same architecture as q-network except last layer
    num_hidden = 256
    return tr.nn.Sequential(
        tr.nn.Linear(in_features=4, out_features=num_hidden, bias=True),
        tr.nn.ReLU(),
        tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        tr.nn.ReLU(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )

if __name__ == "__main__":

    do_training = True
    do_show = True

    basename = "cartpole_data/aux"
    env_name = "CartPole-v1"
    alg_name = "dqn"
    num_timesteps = 499 # 500-1 ensures max timestep termination does not get counted as failure
    num_updates = 1000 # 100*500 = 50000 total timesteps (used for policy training), this is more
    report_period = 1
    checkpoint_period = 10
    fail_weight = .7/.3 # weight failure example loss by this to counter class imbalance (~ (1-failrate) / failrate)
    sharpen = 3. # sharpening factor for softmax logit in failed examples
    bucket = 20 # plots average over timesteps within this bucket size

    num_repetitions = 3
    prediction_windows = [5, 27, 50]
    learning_rates = [1e-1, 1e-2, 1e-3]

    if do_training:

        for (H, lr, rep) in it.product(prediction_windows, learning_rates, range(num_repetitions)):
    
            mu = setup_mu()
            optimizer = tr.optim.Adam(mu.parameters(), lr=lr)
    
            failures = []
            durations = []
            probs = []
            accuracies = []
            gradmaxs = []
            losses = []
            timesteps = []
            total_timesteps = 0
    
            env, model = su.load(env_name, alg_name)
            start_loop = perf_counter()
            for update in range(num_updates):
        
                # run an episode (one episode per update since ragged)
                failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)

                # save failure indicators and times
                failures.append(failure)
                durations.append(len(observations))

                # update cumulative timesteps
                total_timesteps += len(observations)

                # forward pass
                features = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
                logits = mu(features).squeeze()

                # in failure episode, maximum logit up to t-H should be positive
                if failure:
                    # use first observation if failure occurred within first H steps
                    if len(observations) <= H:
                        max_logit = logits[0]
                    else:
                        # max_logit = logits[-H-1] # inductive bias that closest timestep should be easiest to predict
                        # max_logit = logits[:-H].max() # hard max
                        max_logit = (tr.softmax(logits[:-H], dim=0) * logits[:-H]).sum() # denser signal

                    # loss = tr.nn.functional.relu(-max_logit) # penalize if max_logit is negative
                    loss = tr.nn.functional.softplus(-max_logit) # denser signal
                    accuracy = (max_logit >= 0).to(tr.float)

                # otherwise, all logits should be negative
                else:
                    # loss = tr.nn.functional.relu(logits).mean() # so penalize any positives
                    loss = tr.nn.functional.softplus(logits).mean() # denser signal
                    accuracy = (logits <= 0).to(tr.float).mean()

                weighted_loss = (fail_weight * loss) if failure else loss
        
                # backward pass and update
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
        
                # update learning curves
                losses.append(loss.item())
                gradmaxs.append(max([p.grad.abs().max() for p in mu.parameters()]))
                probs.append(tr.sigmoid(logits).mean().item())
                accuracies.append(accuracy)
        
                # progress report
                if update % report_period == 0:
                    failrate = np.mean(failures)
                    failtime = np.mean([d for (d,f) in zip(durations, failures) if f])
                    print(f"{H=}, {lr=}, {rep=}, {update=} of {num_updates}: {failrate=:.3f}, {failtime=:.1f}, {loss=:.3f}, {accuracy=:.3f}, probs={probs[-1]:.3f}, |grad|<={gradmaxs[-1]:.5f}")
    
                    # save latest model and optimizer for resuming later, along with metrics so far
                    tr.save({
                        'mu': mu.state_dict(),
                        'opt': optimizer.state_dict(),
                    }, f"{basename}_H{H}_lr{lr}_r{rep}.pt")
                    with open(f"{basename}_H{H}_lr{lr}_r{rep}.pkl","wb") as f:
                        pk.dump((failures, durations, losses, accuracies, probs, gradmaxs), f)
    
                if (update+1) % checkpoint_period == 0:
                    tr.save(mu.state_dict(), f"{basename}_H{H}_lr{lr}_r{rep}_u{update+1}.pt")
                    with open(f"{basename}_H{H}_lr{lr}_r{rep}_u{update+1}.pkl","wb") as f:
                        pk.dump((failures, durations, losses, accuracies, probs, gradmaxs), f)

            total_time = perf_counter() - start_loop
            print(f"{total_time:.3f} seconds, {total_time / num_updates:.3f} per update")

    if do_show:

        failures = {}
        durations = {}
        losses = {}
        accuracies = {}
        probs = {}
        gradmaxs = {}
        
        for (H, lr, rep) in it.product(prediction_windows, learning_rates, range(num_repetitions)):
            with open(f"{basename}_H{H}_lr{lr}_r{rep}.pkl","rb") as f:
                (failures[H,lr,rep], durations[H,lr,rep], losses[H,lr,rep], accuracies[H,lr,rep], probs[H,lr,rep], gradmaxs[H,lr,rep],) = pk.load(f)

        import matplotlib.pyplot as pt

        for (data, label) in [(accuracies, "Accuracy"), (losses, "Loss")]:
            fig = pt.figure(figsize=(6,3))
            sp = 0
            for ((row, H), (col, lr)) in it.product(enumerate(prediction_windows), enumerate(learning_rates)):
                sp += 1
                pt.subplot(len(prediction_windows), len(learning_rates), sp)
                # pt.title(f"{H=}, {lr=}")
                if row == 0: pt.title(f"{lr=}")
                if col == 0: pt.ylabel(f"{H=}")
    
                # all_data = []
                # min_len = np.inf
                # for rep in range(num_repetitions):
                #     pt.plot(data[H,lr,rep], '-', c=(.8,.8,.8))
                #     all_data.append(data[H,lr,rep])
                #     min_len = min(min_len, len(data[H,lr,rep]))
                # all_data = np.array([acc[:min_len] for acc in all_data])
                # pt.plot(all_data.mean(axis=0), 'k-')
                rep_data = np.array([data[H,lr,r] for r in range(num_repetitions)])
                bucket_data = rep_data.reshape(num_repetitions, -1, bucket).mean(axis=-1)
                pt.plot(rep_data.T, '-', c=(.8,.8,.8))
                # pt.plot(bucket/2 + np.arange(0, num_updates, bucket), bucket_data.T, '-', c=(.8,.8,.8))
                pt.plot(bucket/2 + np.arange(0, num_updates, bucket), bucket_data.mean(axis=0), 'k-')

                if label == "Loss":
                    pt.yscale("log")
                    # pt.ylim([1e-1, 1e1])
    
            fig.supxlabel("Parameter update")
            fig.supylabel(label)
            pt.tight_layout()
            pt.savefig(f"{basename}_{label}.eps")
        pt.show()





