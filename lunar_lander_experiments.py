from line_profiler import profile # python -m kernprof -lvr thisfile.py
import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su

def failure_predicate(env, obs, reward, done, infos):
    return (reward[0] == -100)

# @profile
def train_mu(num_episodes, num_timesteps, batch_size, report_period, checkpoint_base):

    # MLP - same architecture as critic
    num_hidden = 128 # ~75% accuracy
    mu = tr.nn.Sequential(
        tr.nn.Linear(in_features=8, out_features=num_hidden, bias=True),
        # tr.nn.Tanh(),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        # tr.nn.Tanh(),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )
    # print(f"MLP: {sum([p.numel()for p in regressor.parameters()])} parameters")

    loss_fn = tr.nn.BCEWithLogitsLoss()
    # optimizer = tr.optim.AdamW(mu.parameters(), lr=0.0001, weight_decay=.01)
    optimizer = tr.optim.Adam(mu.parameters(), lr=.0001) # no overfit risk when always sampling new data?

    failures = np.empty(num_episodes, dtype=bool)
    durations = np.empty(num_episodes)
    probs = []
    accuracies = [] # full curve
    gradmaxs = []
    losses = []

    obs_batch = []
    env, model = su.load("LunarLander-v3", "a2c")
    for episode in range(num_episodes):
        # run an episode
        failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate)

        # save failure indicators and times
        failures[episode] = failure
        durations[episode] = len(observations)

        # sample uniformly over time
        obs = observations[np.random.randint(len(observations))]
        # obs = observations[-5]

        # train to predict failure label
        obs_batch.append(obs)
        if len(obs_batch) == batch_size:
            features = tr.tensor(np.concatenate(obs_batch, axis=0))
            logits = mu(features).squeeze()
            label = tr.tensor(failures[episode-batch_size+1:episode+1]).to(tr.float)
            # label = tr.ones(batch_size)
            loss = loss_fn(logits, label)
            losses.append(loss.item())
    
            loss.backward()
            gradmaxs.append(max([p.grad.abs().max() for p in mu.parameters()]))

            optimizer.step()
            optimizer.zero_grad()  
    
            probs.append(tr.sigmoid(logits).mean().item())
            correct = (((label > .5) & (logits > 0)) | ((label < .5) & (logits <= 0))).to(tr.float).mean()
            # print(logits)
            # print(label)
            # print(correct)
            accuracies.append(correct)

            obs_batch = []

        # # train to predict failure label
        # logits = mu(tr.tensor(obs)).squeeze()
        # label = tr.tensor(failure).to(tr.float)
        # loss = loss_fn(logits, label)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # gradmaxs.append(max([p.grad.abs().max() for p in mu.parameters()]))

        # probs.append(tr.sigmoid(logits).item())
        # correct = ((failure and logits > 0) or (not failure and logits <= 0))
        # accuracies.append(correct)
        # accuracy_ema = ema * accuracy_ema + (1 - ema) * correct

        if episode % report_period == report_period-1:
            failrate = failures[:episode+1].mean()
            failtime = durations[:episode+1][failures[:episode+1]].mean()
            # print(f"{episode=} of {num_episodes}: {failrate=:.3f}, {failtime=:.1f}, accuracy[-30:]={np.mean(accuracies[-30:]):.3f}, probs[-30:]={np.mean(probs[-30:]):.3f}, |grad[-30:]|<={np.mean(gradmaxs[-30:]):.5f}")
            print(f"{episode=} of {num_episodes}: {failrate=:.3f}, {failtime=:.1f}, {loss=:.3f}, accuracy~{accuracies[-1]:.3f}, probs~{probs[-1]:.3f}, |grad|<={gradmaxs[-1]:.5f}")
            tr.save(mu.state_dict(), f"{checkpoint_base}.pt")
            with open(f"{checkpoint_base}.pkl","wb") as f:
                pk.dump((losses, accuracies, probs, gradmaxs), f)

    return mu, accuracies

def run_repetition(num_training_episodes):

    mu, accuracies = train_mu(
        num_training_episodes,
        num_timesteps = 1000,
        batch_size = 32,
        report_period = 32, # should >= batch size
        checkpoint_base = "lle_mu",
    )
    # fail_rate = f(confusion)
    # delta = 0.5 * fail_rate
    # tau = calibrate(mu, delta)
    # status = deploy(mu)
    
    # return mu, delta, tau, status

if __name__ == "__main__":

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_training_episodes = 1000
    num_repetitions = 1

    for rep in range(num_repetitions):
        _ = run_repetition(
            num_training_episodes,
        )

    with open(f"lle_mu.pkl","rb") as f:
        (losses, accuracies, probs, gradmaxs) = pk.load(f)

    import matplotlib.pyplot as pt
    pt.subplot(1,4,1)
    pt.plot(losses)
    pt.subplot(1,4,2)
    pt.plot(accuracies)
    pt.subplot(1,4,3)
    pt.plot(probs)
    pt.subplot(1,4,4)
    pt.plot(gradmaxs)
    pt.show()


