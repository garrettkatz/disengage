"""
Train mu(s,h) to predict Pr(no fail in [t+1, t+h] | S_t = s) for any non-fail state s
"""
from time import perf_counter
import matplotlib.pyplot as pt
import numpy as np
import torch as tr

def sample_batch(rollouts, failures, batch_size, max_episode_length):
    # rollouts, failures: same format as buffer
    # rollouts[r][t,:]: observation vector at time t in rollout r
    # failures[r]: whether rollout r ended in failure
    # if failure, last state in rollout is failure state

    states, horizons, nofails = [], [], []
    for r in np.random.randint(len(rollouts), size=batch_size):

        # sample random timing
        t = np.random.randint(min(len(rollouts[r])-1, max_episode_length-1)) # exclude final/failure state
        tph = np.random.randint(t+1, max_episode_length)

        # no fail indicator in [t+1,t+h]
        nofail = (not failures[r]) or (tph < len(rollouts[r])-1)

        # grow the batch
        states.append(rollouts[r][t])
        horizons.append(float(tph-t))
        nofails.append(float(nofail))

    # package batch and return
    states = tr.stack(states)
    horizons = tr.tensor(horizons) / max_episode_length # normalize to (0,1)
    observations = tr.cat([states, horizons[:,None]], dim=-1)
    nofails = tr.tensor(nofails)
    return observations, nofails


def train(params, setup_mu):

    basename = params["basename"]
    buf_basename = params["buf_basename"]
    buf_rep = params["buf_rep"]
    train_rep = params["train_rep"]
    obs_noise = params["obs_noise"]
    total_timesteps = params["total_timesteps"]
    max_episode_length = params["max_episode_length"]
    num_updates = params["num_updates"]
    train_batch_size = params["train_batch_size"]
    valid_batch_size = params["valid_batch_size"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    train_fraction = params["train_fraction"]
    report_period = params["report_period"]
    checkpoint_period = params["checkpoint_period"]
    bucket_size = params["bucket_size"]

    # setup filenames
    buffer_name = f"{buf_basename}_buffer_rep{buf_rep}_on{obs_noise}.pt"
    train_basename = f"{basename}_train_rep{train_rep}_on{obs_noise}_lr{learning_rate}_wd{weight_decay}_nu{num_updates}"

    # load buffer and make train/validation splits
    (rollouts, failures, failrate) = tr.load(buffer_name, weights_only=True)
    num_split = int(len(rollouts)*train_fraction)
    train_rollouts, valid_rollouts = rollouts[:num_split], rollouts[num_split:]
    train_failures, valid_failures = failures[:num_split], failures[num_split:]

    # initialize training
    mu = setup_mu()
    bce = tr.nn.BCEWithLogitsLoss() # minimized at max likelihood estimate of bernoulli
    optimizer = tr.optim.AdamW(mu.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_curve = {"train": [], "valid": []}

    # training loop
    start_loop = perf_counter()
    for update in range(1,num_updates+1):

        # sample a batch of s_t, h, non-failure indicator
        observations, nofails = sample_batch(train_rollouts, train_failures, train_batch_size, max_episode_length)

        # backprop bce loss on indicator
        predictions = mu(observations).squeeze()
        loss = bce(predictions, nofails)

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track metrics
        loss_curve["train"].append(loss.item())

        # progress report
        if report_period is not None and update % report_period == 0:
            print(f"{update=}: train loss={loss_curve['train'][-1]:.5e}")

        # checkpointing and validation
        if update % checkpoint_period == 0:

            # save model
            tr.save(mu.state_dict(), f"{train_basename}_u{update}.pt")

            # measure validation loss
            with tr.no_grad():

                # sample a validation batch
                observations, nofails = sample_batch(valid_rollouts, valid_failures, valid_batch_size, max_episode_length)

                # calculate loss
                predictions = mu(observations).squeeze()
                loss = bce(predictions, nofails)

                # track metrics
                loss_curve["valid"].append(loss.item())

            print(f"{update=}: valid loss={loss_curve['valid'][-1]:.5e}")

    training_time = perf_counter()-start_loop
    print(f"updates done in {training_time:.3f}s ({training_time/num_updates}s per update)")

    # reload best checkpoint
    best_checkpoint = int(np.argmin(loss_curve["valid"]))
    best_valid_loss = loss_curve['valid'][best_checkpoint]
    early_stop = (best_checkpoint + 1) * checkpoint_period
    early_state_dict = tr.load(f"{train_basename}_u{early_stop}.pt", weights_only=True)
    mu.load_state_dict(early_state_dict)
    print(f"reloaded checkpoint {early_stop} with valid loss {best_valid_loss}")
    tr.save((failrate, loss_curve, early_stop, early_state_dict), f"{train_basename}_trained.pt")


def show_results(params, setup_mu):

    pt.rcParams['font.family'] = 'serif'
    pt.rcParams['font.size'] = 12

    basename = params["basename"]
    buf_basename = params["buf_basename"]
    buf_rep = params["buf_rep"]
    train_rep = params["train_rep"]
    obs_noise = params["obs_noise"]
    total_timesteps = params["total_timesteps"]
    max_episode_length = params["max_episode_length"]
    num_updates = params["num_updates"]
    train_batch_size = params["train_batch_size"]
    valid_batch_size = params["valid_batch_size"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    train_fraction = params["train_fraction"]
    report_period = params["report_period"]
    checkpoint_period = params["checkpoint_period"]
    bucket_size = params["bucket_size"]

    # setup filenames
    buffer_name = f"{buf_basename}_buffer_rep{buf_rep}_on{obs_noise}.pt"
    train_basename = f"{basename}_train_rep{train_rep}_on{obs_noise}_lr{learning_rate}_wd{weight_decay}_nu{num_updates}"

    # load buffer and make train/validation splits
    (rollouts, failures, failrate) = tr.load(buffer_name, weights_only=True)
    num_split = int(len(rollouts)*train_fraction)
    train_rollouts, valid_rollouts = rollouts[:num_split], rollouts[num_split:]
    train_failures, valid_failures = failures[:num_split], failures[num_split:]

    # load best validation model
    (_, loss_curve, early_stop, mu_state_dict) = tr.load(f"{train_basename}_trained.pt", weights_only=True)
    print(f"{failrate=:.3f}")
    print(f"{early_stop=}: {min(loss_curve['valid'])}")
    mu = setup_mu()
    mu.load_state_dict(mu_state_dict)

    pt.figure(figsize=(16,4))

    # loss trendline
    buckets = np.array(loss_curve["train"]).reshape(-1, bucket_size).mean(axis=1)

    pt.subplot(1, 4, 1)
    pt.plot(loss_curve["train"], '-', color=(.8,)*3)
    pt.plot(np.arange(len(buckets))*bucket_size + bucket_size/2, buckets, 'k:', label="train")
    pt.plot(checkpoint_period * np.arange(1, len(loss_curve["valid"])+1), loss_curve["valid"], 'k-', label="valid")
    pt.xlabel("Update")
    pt.yscale("log")
    pt.title("Learning curve")
    pt.ylabel("BCE")
    pt.legend()

    # predict on a validation batch for visualization
    observations, nofails = sample_batch(valid_rollouts, valid_failures, valid_batch_size, max_episode_length)
    with tr.no_grad():
        predictions = mu(observations).squeeze()

    # not training on logits anymore, so pass through softmax for visualization
    predictions = tr.sigmoid(predictions)

    print("Confusion:")
    for (p,n) in [(0,0),(0,1),(1,0),(1,1)]:
        count = (((predictions > .5) == p) & (nofails == n)).sum()
        print(f"{p=},{n=}: {count} ({count / valid_batch_size}%)")

    pt.subplot(1, 4, 2)
    pt.plot(predictions, nofails + 0.1*tr.randn_like(nofails), 'k.', alpha=.1)
    pt.plot([0, 1], [0, 1], ':', color=(.8,)*3)
    pt.xlabel("Prediction")
    pt.ylabel("Target")
    pt.title("Output scatterplot")

    pt.subplot(1, 4, 3)
    pt.plot(max_episode_length * observations[:,-1], (predictions - nofails).abs(), 'k.', alpha=.1)
    pt.xlabel("Horizon")
    pt.ylabel("|prediction-label|")
    pt.title("Horizon performance")

    pt.subplot(1, 4, 4)
    # pt.bar(np.arange(1, max_episode_length), [(h == np.round(max_episode_length * observations[:,-1])).sum() for h in range(1, max_episode_length)], align="edge")
    pt.plot(np.arange(1, max_episode_length), [(h == np.round(max_episode_length * observations[:,-1])).sum() for h in range(1, max_episode_length)], 'k-')
    pt.xlabel("Horizon")
    pt.ylabel("Frequency")
    pt.title("Horizon distribution")

    pt.tight_layout()
    pt.show()


if __name__ == "__main__":

    for rep in [4]:#range(5):

        params = {
            "basename": "ru_data/ru_nt",
            "buf_basename": "ru_data/ru_cond",
            "buf_rep": 0,
            "train_rep": rep,

            "obs_noise": .05,
            "max_episode_length": 240,
            "total_timesteps": 200_000,

            "num_updates": 10_400,
            "train_batch_size": 781,
            "valid_batch_size": 10000,

            "learning_rate": 5e-4,#5e-3,
            "weight_decay": .1,
            "train_fraction": .8, # fraction of rollouts used for training

            "report_period": 10,#None,
            "checkpoint_period": 100,
            "bucket_size": 10, # buckets for learning curve trendlines
        }

        def setup_mu():
            # MLP - same architecture as ruins critic
            num_hidden = 64
            return tr.nn.Sequential(
                # tr.nn.Linear(in_features=12, out_features=num_hidden, bias=True),
                tr.nn.Linear(in_features=13, out_features=num_hidden, bias=True), # +1 for scalar horizon
                tr.nn.Tanh(),
                tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
                tr.nn.Tanh(),
                tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
            )

        # train(params, setup_mu)

    show_results(params, setup_mu)

