"""
Train mu(s) to predict Pr(no fail in [t+1, t+h] | S_t = s) for every h in [1,T] at any non-fail state s
Uses monotonic architecture via sum over h-wise output head of nonpositive deltas
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
    # returns horizons h in [1, max_episode_length - 1 = T]
    # returns nofails: indicator of no failure in [t+1,t+h]

    states, horizons, nofails = [], [], []
    for r in np.random.randint(len(rollouts), size=batch_size):

        # random time-step for state
        t = np.random.randint(min(len(rollouts[r])-1, max_episode_length-1)) # exclude final/failure state

        # half of samples for numerators, half for denominators
        if False: #np.random.choice([True, False]):
            # numerator sampling: horizon is [t+1,L-1] for some L
            tph = np.random.randint(t+1, max_episode_length)
        else:
            # denominator sampling: horizon is [t+1,T]
            tph = max_episode_length - 1

        # no fail indicator in [t+1,t+h]
        nofail = (not failures[r]) or (tph < len(rollouts[r])-1)

        # grow the batch
        states.append(rollouts[r][t])
        horizons.append(tph-t)
        nofails.append(float(nofail))

    # package batch and return
    observations = tr.stack(states)
    horizons = tr.tensor(horizons)
    nofails = tr.tensor(nofails)
    return observations, horizons, nofails

def setup_mu_factory(max_episode_length):
    """
    output shape is (max_episode_length-1 = T,)
    output[h-1] = estimated probability of no failure in [t+1,t+h]
    Pr(NF in next H) >= Pr(NF in next H+1)
    """

    def setup_mu():

        class Mu(tr.nn.Module):

            def __init__(self):
                super(Mu, self).__init__()
                # MLP - same architecture as ruins critic except many output heads
                num_hidden = 64
                self.ff = tr.nn.Sequential(
                    tr.nn.Linear(in_features=12, out_features=num_hidden, bias=True),
                    tr.nn.Tanh(),
                    tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
                    tr.nn.Tanh(),
                    tr.nn.Linear(in_features=num_hidden, out_features=max_episode_length, bias=True),
                )
                self.sp = tr.nn.Softplus()

            def forward(self, obs):
                heads = self.ff(obs)
                # h0 = heads[...,:1]
                # deltas = self.sp(heads[...,1:])
                # logits = h0 - deltas.cumsum(dim=-1) # monotonically decreasing
                logits = heads[...,1:]
                return logits # (..., max_ep_len-1)

        return Mu()

    return setup_mu

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
        observations, horizons, nofails = sample_batch(train_rollouts, train_failures, train_batch_size, max_episode_length)

        # backprop bce loss on indicator
        all_logits = mu(observations)
        predictions = tr.take_along_dim(all_logits, horizons[:,None]-1, dim=-1).squeeze()
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
                observations, horizons, nofails = sample_batch(valid_rollouts, valid_failures, valid_batch_size, max_episode_length)

                # calculate loss
                all_logits = mu(observations)
                predictions = tr.take_along_dim(all_logits, horizons[:,None]-1, dim=-1).squeeze()
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

    pt.subplot(1, 5, 1)
    pt.plot(loss_curve["train"], '-', color=(.8,)*3)
    pt.plot(np.arange(len(buckets))*bucket_size + bucket_size/2, buckets, 'k:', label="train")
    pt.plot(checkpoint_period * np.arange(1, len(loss_curve["valid"])+1), loss_curve["valid"], 'k-', label="valid")
    pt.xlabel("Update")
    pt.yscale("log")
    pt.title("Learning curve")
    pt.ylabel("BCE")
    pt.legend()

    # predict on a validation batch for visualization
    observations, horizons, nofails = sample_batch(valid_rollouts, valid_failures, valid_batch_size, max_episode_length)
    with tr.no_grad():
        all_logits = mu(observations)
        predictions = tr.take_along_dim(all_logits, horizons[:,None]-1, dim=-1).squeeze()

    # not training on logits anymore, so pass through softmax for visualization
    predictions = tr.sigmoid(predictions)

    print("Confusion:")
    for (p,n) in [(0,0),(0,1),(1,0),(1,1)]:
        count = (((predictions > .5) == p) & (nofails == n)).sum()
        print(f"{p=},{n=}: {count} ({count / valid_batch_size}%)")

    pt.subplot(1, 5, 2)
    pt.plot(predictions, nofails + 0.1*tr.randn_like(nofails), 'k.', alpha=.1)
    pt.plot([0, 1], [0, 1], ':', color=(.8,)*3)
    pt.xlabel("Prediction")
    pt.ylabel("Target")
    pt.title("Output scatterplot")

    pt.subplot(1, 5, 3)
    pt.plot(horizons, predictions, 'k.', alpha=.1)
    avgs = [predictions[horizons==h].mean() for h in range(1, max_episode_length)]
    pt.plot(np.arange(1, max_episode_length), avgs, 'b-')
    pt.xlabel("Horizon")
    pt.ylabel("prediction")
    pt.title("Per-Horizon Predictions")

    pt.subplot(1, 5, 4)
    pt.plot(horizons, (predictions - nofails).abs(), 'k.', alpha=.1)
    avgs = [(predictions - nofails).abs()[horizons==h].mean() for h in range(1, max_episode_length)]
    pt.plot(np.arange(1, max_episode_length), avgs, 'b-')
    pt.xlabel("Horizon")
    pt.ylabel("|prediction-label|")
    pt.title("Horizon performance")

    pt.subplot(1, 5, 5)
    # pt.bar(np.arange(1, max_episode_length), [(h == np.round(max_episode_length * observations[:,-1])).sum() for h in range(1, max_episode_length)], align="edge")
    pt.plot(np.arange(1, max_episode_length), [(h == horizons).sum() for h in range(1, max_episode_length)], 'k-')
    pt.xlabel("Horizon")
    pt.ylabel("Frequency")
    pt.title("Horizon distribution")

    pt.tight_layout()
    pt.show()

    # look at monotonicity
    num_obs = 10
    T_probs = []
    pt.figure(figsize=(16,4))
    for n in range(num_obs):
        obs = observations[np.random.randint(len(observations))]
        with tr.no_grad():
            predictions = mu(obs)
        pt.plot(tr.arange(1,predictions.shape[-1]+1), predictions, '-')
        T_probs.append(tr.sigmoid(predictions)[-1])

    print('t=0 probs of no failure in [t+1,T]')
    print(T_probs)
    print('mean', tr.stack(T_probs).mean())
    # pt.xlim([0,100])
    pt.xlabel("Horizon")
    pt.ylabel("Probability logit")
    pt.title(f"Per-horizon predictions on {num_obs} random observations")
    pt.tight_layout()
    pt.show()


if __name__ == "__main__":

    for rep in range(5):

        params = {
            "basename": "ru_data/ru_mono",
            "buf_basename": "ru_data/ru_cond",
            "buf_rep": rep,
            "train_rep": rep,

            "obs_noise": .05,
            "max_episode_length": 240,
            "total_timesteps": 200_000,

            "num_updates": 10_400,
            "train_batch_size": 781,
            "valid_batch_size": 10000,

            "learning_rate": 1e-5,#5e-3,
            "weight_decay": .1,
            "train_fraction": .8, # fraction of rollouts used for training

            "report_period": 10,#None,
            "checkpoint_period": 100,
            "bucket_size": 10, # buckets for learning curve trendlines
        }

        train(params, setup_mu_factory(params["max_episode_length"]))

    show_results(params, setup_mu_factory(params["max_episode_length"]))


