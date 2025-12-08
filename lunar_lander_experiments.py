import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su

def failure_predicate(env, obs, reward, done, infos):
    return (reward[0] == -100)

def train_mu(num_episodes, num_timesteps, report_period, ema, checkpoint_base):

    # MLP - same architecture as critic
    num_hidden = 64 # ~75% accuracy
    mu = tr.nn.Sequential(
        tr.nn.Linear(in_features=8, out_features=num_hidden, bias=True),
        # tr.nn.Tanh(),
        # tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )
    # print(f"MLP: {sum([p.numel()for p in regressor.parameters()])} parameters")

    loss_fn = tr.nn.BCEWithLogitsLoss()
    optimizer = tr.optim.AdamW(mu.parameters(), lr=0.01, weight_decay=.01)

    accuracies = [] # full curve
    accuracy_ema = 0.5 # exponential moving average, start like random guessing

    env, model = su.load("LunarLander-v3", "a2c")
    for episode in range(num_episodes):
        # run an episode
        failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate)

        # sample uniformly over time
        obs = observations[np.random.randint(len(observations))]

        # train to predict failure label
        logits = mu(tr.tensor(obs)).squeeze()
        label = tr.tensor(failure).to(tr.float)
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = ((failure and logits > 0) or (not failure and logits <= 0))
        accuracies.append(correct)
        accuracy_ema = ema * accuracy_ema + (1 - ema) * correct

        if episode % report_period == 0:
            print(f"{episode=} of {num_episodes}: {accuracy_ema=}")
            tr.save(mu.state_dict(), f"{checkpoint_base}.pt")
            with open(f"{checkpoint_base}.pkl","wb") as f: pk.dump(accuracies, f)

    return mu, accuracies

def run_repetition(num_training_episodes):

    mu, accuracies = train_mu(
        num_training_episodes,
        num_timesteps = 1000,
        report_period = 10,
        ema = .99,
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
    num_training_episodes = 5000
    num_repetitions = 1

    for rep in range(num_repetitions):
        _ = run_repetition(
            num_training_episodes,
        )


