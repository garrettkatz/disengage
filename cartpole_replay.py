from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import itertools as it
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
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

    basename = "cartpole_data/replay"
    env_name = "CartPole-v1"
    alg_name = "dqn"
    episode_timesteps = 499 # 500-1 ensures max timestep termination does not get counted as failure

    num_repetitions = 1
    buffer_sizes = [50000]
    horizons = [5]
    training_presentations = 1_600_000 # total times an observation is passed through network during training
    learning_rate = 1e-5
    half_batch = 16
    report_period = 10

    if do_training:

        env, model = su.load(env_name, alg_name)

        for (horizon, bufsize, rep) in it.product(horizons, buffer_sizes, range(num_repetitions)):

            # fill buffer
            start_buffer = perf_counter()
            total_timesteps = 0
            safe_episodes = []
            fail_episodes = []            
            while total_timesteps < bufsize:

                # collect an episode
                failure, _, _, observations, _ = su.run(env, model, episode_timesteps, failure_predicate, perturb_action=perturb_action)
                total_timesteps += len(observations)

                # save episode
                buf = fail_episodes if failure else safe_episodes
                buf.append(tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32))

            print(f"{horizon=}, {bufsize=}, {rep=}: buffer filled ({total_timesteps} steps, {len(safe_episodes)+len(fail_episodes)} episodes, {perf_counter()-start_buffer:.3f}s)")

            # do training
            start_training = perf_counter()
            presentations = 0
            updates = 0
            mu = setup_mu()
            optimizer = tr.optim.SGD(mu.parameters(), lr=learning_rate)
            loss_fn = tr.nn.BCEWithLogitsLoss()
            losses = []
            accuracies = []
            gradmaxs = []
            while presentations < training_presentations:
                updates += 1

                obs, labels = [], []
                for hb in range(half_batch):
                    # sample one of each episode type (safe/fail)
                    safe_episode = safe_episodes[np.random.randint(len(safe_episodes))]
                    fail_episode = fail_episodes[np.random.randint(len(fail_episodes))]
    
                    # sample one observation from each
                    safe_obs = safe_episode[np.random.randint(len(safe_episode))] # uniform
                    fail_obs = fail_episode[max(0, len(fail_episode) - np.random.randint(horizon, 2*horizon))] # uniform near horizon boundary

                    # add to batch
                    obs.extend([safe_obs, fail_obs])
                    labels.extend([0., 1.]) # failure indicator

                obs = tr.stack(obs)
                labels = tr.tensor(labels)
                presentations += len(obs)

                # forward pass
                logits = mu(obs).squeeze()

                # loss and accuracy
                loss = loss_fn(logits, labels)
                accuracy = ((logits.detach().numpy() > 0) == labels.numpy()).mean()

                # # sample one of each episode type (safe/fail)
                # safe_episode = safe_episodes[np.random.randint(len(safe_episodes))]
                # fail_episode = fail_episodes[np.random.randint(len(fail_episodes))]
                # presentations += len(safe_episode) + len(fail_episode)

                # # forward passes
                # safe_logits = mu(safe_episode).squeeze()
                # fail_logits = mu(fail_episode).squeeze()

                # # in safe episode, all logits should be negative
                # safe_loss = tr.nn.functional.softplus(safe_logits).mean() # softplus instead of relu for denser signal
                # safe_accuracy = (safe_logits.detach().numpy() <= 0).mean()

                # # in failure episode, maximum logit up to fail time - horizon should be positive
                # # use first observation if failure occurred within first horizon steps
                # if len(fail_logits) <= horizon:
                #     max_logit = fail_logits[0]
                # else:
                #     max_logit = (tr.softmax(fail_logits[:-horizon], dim=0) * fail_logits[:-horizon]).sum() # softmax for denser signal

                # fail_loss = tr.nn.functional.softplus(-max_logit) # softplus instead of relu for denser signal
                # fail_accuracy = (max_logit.detach().numpy() >= 0)

                # # equal weight between success and failure loss to counteract class imbalance
                # loss = (safe_loss + fail_loss)/2
                # accuracy = (safe_accuracy + fail_accuracy)/2

                # gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                # update learning curves
                losses.append(loss.item())
                gradmaxs.append(max([p.grad.abs().max() for p in mu.parameters()]))
                accuracies.append(accuracy)

                if updates % report_period == 0:
                    print(f"{horizon=}, {bufsize=}, {rep=}: {presentations=}, loss={loss.item():.3f}, acc={accuracy:.3f} ({perf_counter()-start_training:.3f}s)")
                    # print(f" safe acc {safe_accuracy:.3f}, fail acc {fail_accuracy:.3f}")
                    # print(f" safe loss {safe_loss.item():.3f}, fail loss {fail_loss.item():.3f}")
                    # print(f" safe max {safe_logits.detach().max().item():.3f}, fail max {fail_logits.detach().max().item():.3f}")

            pt.subplot(1,2,1)
            pt.plot(losses)
            pt.subplot(1,2,2)
            pt.plot(accuracies)
            pt.title("acc")
            pt.show()

