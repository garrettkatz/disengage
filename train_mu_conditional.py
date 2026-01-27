from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import numpy as np
import matplotlib.pyplot as pt
import torch as tr

def perturb_obs_factory(relative):
    def perturb_obs(o):
        return o * np.random.uniform(1 - relative, 1 + relative, size=o.shape)
    return perturb_obs

def sample_batch(rollouts, failures, max_episode_length, batch_size):
    current_obs, next_obs, next_fail = [], [], []
    for r in np.random.randint(len(rollouts), size=batch_size):
        t = np.random.randint(max_episode_length-1)
        current_obs.append( rollouts[r][min(t, len(rollouts[r])-2)] )
        next_obs.append( rollouts[r][min(t+1, len(rollouts[r])-1)] )
        next_fail.append( (failures[r] and t+1 >= len(rollouts[r])) )
    current_obs = tr.stack(current_obs)
    next_obs = tr.stack(next_obs)
    next_fail = tr.tensor(next_fail).to(tr.float32)
    return current_obs, next_obs, next_fail

def collect_data(params, run_episode, report_period=1):

    buf_rep = params["buf_rep"]
    obs_noise = params["obs_noise"]
    total_timesteps = params["total_timesteps"]
    basename = params["basename"]

    rollouts = []
    failures = []
    timesteps_so_far = 0
    start_loop = perf_counter()
    while timesteps_so_far < total_timesteps:

        # simulate another rollout
        failure, _, _, observations, _ = run_episode()
        timesteps_so_far += len(observations)

        # update buffers
        rollouts.append(tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32))
        failures.append(failure)

        # progress report
        if len(rollouts) % report_period == 0: print(f"{timesteps_so_far} of {total_timesteps} timesteps ({len(rollouts)} rollouts)")

    failures = tr.tensor(failures).to(tr.float32)
    failrate = failures.mean().item()
    looptime = perf_counter()-start_loop
    print(f"Rollouts done in {looptime:.3f}s ({looptime/total_timesteps}s/timestep), {failrate=:.3f}")

    buffer_name = f"{basename}_buffer_rep{buf_rep}_on{obs_noise}.pt"
    tr.save((rollouts, failures, failrate), buffer_name)

def train(params, setup_mu):

    buf_rep = params["buf_rep"]
    train_rep = params["train_rep"]
    basename = params["basename"]
    obs_noise = params["obs_noise"]
    total_timesteps = params["total_timesteps"]
    max_episode_length = params["max_episode_length"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    train_batch_size = params["train_batch_size"]
    valid_batch_size = params["valid_batch_size"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    train_fraction = params["train_fraction"]
    report_period = params["report_period"]
    checkpoint_period = params["checkpoint_period"]
    bucket_size = params["bucket_size"]

    buffer_name = f"{basename}_buffer_rep{buf_rep}_on{obs_noise}.pt"
    train_basename = f"{basename}_train_rep{train_rep}_on{obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}"

    (rollouts, failures, failrate) = tr.load(buffer_name, weights_only=True)

    # make train/validation splits
    num_split = int(len(rollouts)*train_fraction)
    train_rollouts, valid_rollouts = rollouts[:num_split], rollouts[num_split:]
    train_failures, valid_failures = failures[:num_split], failures[num_split:]

    mu = {}
    loss_curve = {}
    early_stops = {}
    mse = tr.nn.MSELoss()

    # do training
    for L in range(1, max_leadtime + 1):

        mu[L] = setup_mu()
        loss_curve[L] = {"train": [], "valid": []}
        optimizer = tr.optim.AdamW(mu[L].parameters(), lr=learning_rate, weight_decay=weight_decay)

        start_loop = perf_counter()
        for update in range(1, num_updates+1):

            # sample a training batch
            current_obs, next_obs, next_fail = sample_batch(train_rollouts, train_failures, max_episode_length, train_batch_size)

            # setup targets
            with tr.no_grad():
                targets = next_fail if L==1 else mu[L-1](next_obs).squeeze()

            # calculate loss
            predictions = mu[L](current_obs).squeeze()
            loss = mse(predictions, targets)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # track metrics
            loss_curve[L]["train"].append(loss.item())

            # progress report
            if report_period is not None and update % report_period == 0:
                print(f"{L=}, {update=}: train loss={loss_curve[L]['train'][-1]:.5e}")

            # checkpointing and validation
            if update % checkpoint_period == 0:

                # save model
                tr.save(mu[L].state_dict(), f"{train_basename}_L{L}_u{update}.pt")

                # measure validation loss
                with tr.no_grad():

                    # sample a validation batch
                    current_obs, next_obs, next_fail = sample_batch(valid_rollouts, valid_failures, max_episode_length, valid_batch_size)
    
                    # setup targets
                    targets = next_fail if L==1 else mu[L-1](next_obs).squeeze()
    
                    # calculate loss
                    predictions = mu[L](current_obs).squeeze()
                    loss = mse(predictions, targets)
    
                    # track metrics
                    loss_curve[L]["valid"].append(loss.item())

                print(f"{L=}, {update=}: valid loss={loss_curve[L]['valid'][-1]:.5e}")

        training_time = perf_counter()-start_loop
        print(f"{L=}: updates done in {training_time:.3f}s ({training_time/num_updates}s per update)")

        # reload best checkpoint
        best_checkpoint = int(np.argmin(loss_curve[L]["valid"]))
        best_valid_loss = loss_curve[L]['valid'][best_checkpoint]
        early_stops[L] = (best_checkpoint + 1) * checkpoint_period
        early_state_dict = tr.load(f"{train_basename}_L{L}_u{early_stops[L]}.pt", weights_only=True)
        mu[L].load_state_dict(early_state_dict)
        print(f"{L=}: reloaded checkpoint {early_stops[L]} with valid loss {best_valid_loss}")

    tr.save((failrate, loss_curve, early_stops, {L: mu[L].state_dict() for L in range(1, max_leadtime+1)}), f"{train_basename}_trained.pt")

def show_results(params, setup_mu, Ls):

    pt.rcParams['font.family'] = 'serif'
    pt.rcParams['font.size'] = 12

    buf_rep = params["buf_rep"]
    train_rep = params["train_rep"]
    basename = params["basename"]
    obs_noise = params["obs_noise"]
    max_episode_length = params["max_episode_length"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    valid_batch_size = params["valid_batch_size"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    train_fraction = params["train_fraction"]
    checkpoint_period = params["checkpoint_period"]
    bucket_size = params["bucket_size"]

    buffer_name = f"{basename}_buffer_rep{buf_rep}_on{obs_noise}.pt"
    train_basename = f"{basename}_train_rep{train_rep}_on{obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}"

    # load buffer and mus
    (rollouts, failures, failrate) = tr.load(buffer_name, weights_only=True)
    (_, loss_curve, early_stops, mu_state_dicts) = tr.load(f"{train_basename}_trained.pt", weights_only=True)
    mu = {} 
    print(f"{failrate=:.3f}")
    print("early stops, valid losses:")
    for L in range(1, max_leadtime+1):
        print(L, early_stops[L], min(loss_curve[L]["valid"]))
        mu[L] = setup_mu()
        mu[L].load_state_dict(mu_state_dicts[L])

    # durations = list(map(len, rollouts))
    # pt.hist(durations, bins=50)
    # pt.xlabel("Episode duration")
    # pt.ylabel("Frequency")
    # pt.yscale("log")
    # pt.show()

    # make train/validation splits
    num_split = int(len(rollouts)*train_fraction)
    train_rollouts, valid_rollouts = rollouts[:num_split], rollouts[num_split:]
    train_failures, valid_failures = failures[:num_split], failures[num_split:]

    pt.figure(figsize=(15,3))

    # for L in range(1, max_leadtime + 1):
    for i, L in enumerate(Ls):

        # loss trendline
        buckets = np.array(loss_curve[L]["train"]).reshape(-1, bucket_size).mean(axis=1)
    
        pt.subplot(2, len(Ls), i+1)
        pt.plot(loss_curve[L]["train"], '-', color=(.8,)*3)
        pt.plot(np.arange(len(buckets))*bucket_size + bucket_size/2, buckets, 'k:', label="train")
        pt.plot(checkpoint_period * np.arange(1, len(loss_curve[L]["valid"])+1), loss_curve[L]["valid"], 'k-', label="valid")
        pt.xlabel("Update")
        pt.yscale("log")
        pt.title(str(L))
        if L==1:
            pt.ylabel("MSE")
            pt.legend()

        # predict on a validation batch for visualization
        with tr.no_grad():
            current_obs, next_obs, next_fail = sample_batch(valid_rollouts, valid_failures, max_episode_length, valid_batch_size)
            predictions = mu[L](current_obs).squeeze()
            targets = next_fail if L==1 else mu[L-1](next_obs).squeeze()

        pt.subplot(2, len(Ls), len(Ls) + i + 1)
        pt.plot(predictions, targets, 'k.', alpha=.5)
        pt.plot([0, 1], [0, 1], ':', color=(.8,)*3)
        pt.xlabel("Prediction")
        pt.ylabel("Target")

    pt.tight_layout()
    pt.show()


