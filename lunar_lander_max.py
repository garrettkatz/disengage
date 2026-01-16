from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import sb_utils as su

perturb_action = None

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

    do_sampling = True
    do_training = True
    resume = False
    do_show = True

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    num_rollouts = 200 # 200 rollouts * 1000 timesteps = 200K is parity with original training
    learning_rate = 0.001
    max_lead_time = 10

    train_fraction = .8 # for train/test split
    num_updates = 20000 # 5000 is parity with original training
    batch_size = 32 # 40 observation pairs is parity with original training
    report_period = 10
    checkpoint_period = 1000
    loss_bucket_size = 100 # buckets for learning curve trendlines
    accu_bucket_size = 10 # buckets for accuracy curve trendlines
    basename = "ll_data/ll_max"

    # initializations
    mu = setup_mu()
    env, model = su.load(env_name, alg_name)
    # optimizer = tr.optim.Adam(mu.parameters(), lr=learning_rate)
    optimizer = tr.optim.AdamW(mu.parameters(), lr=learning_rate, weight_decay=.5)
    # optimizer = tr.optim.SGD(mu.parameters(), lr=learning_rate)

    # fill buffer
    if do_sampling:
        safe_episodes = []
        fail_episodes = []
        start_loop = perf_counter()
        for rollout in range(num_rollouts):
    
            # simulate rollout
            failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)
            observations = tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32)
            (fail_episodes if failure else safe_episodes).append(observations)
    
        failrate = len(fail_episodes) / num_rollouts
        print(f"Rollouts done in {perf_counter()-start_loop:.3f}s, {failrate=:.3f}")

        tr.save((safe_episodes, fail_episodes, failrate), f"{basename}_buffer.pt")

    if do_training:

        (safe_episodes, fail_episodes, failrate) = tr.load(f"{basename}_buffer.pt", weights_only=True)

        # make train/test split
        safe_split, fail_split = int(train_fraction * len(safe_episodes)), int(train_fraction * len(fail_episodes))
        safe_episodes, safe_episodes_test = safe_episodes[:safe_split], safe_episodes[safe_split:]
        fail_episodes, fail_episodes_test = fail_episodes[:fail_split], fail_episodes[fail_split:]

        if resume:
            (loss_curve, accu_curve, mu_state_dict, opt_state_dict) = tr.load(f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_trained.pt", weights_only=True)
            mu.load_state_dict(mu_state_dict)
            optimizer.load_state_dict(opt_state_dict)
            start_update = len(loss_curve)
        else:
            loss_curve = []
            accu_curve = []
            start_update = 0

        # do training
        start_loop = perf_counter()
        for update in range(start_update, num_updates):
    
            # sample episodes
            safe_batch = [safe_episodes[i] for i in np.random.randint(len(safe_episodes), size=batch_size)]
            fail_batch = [fail_episodes[i] for i in np.random.randint(len(fail_episodes), size=batch_size)]

            # feed through mu
            safe_preds = [mu(episode) for episode in safe_batch]
            fail_preds = [mu(episode) for episode in fail_batch]

            # accumulate loss over possible lead times
            loss = 0.
            for lead_time in range(1, max_lead_time+1):
                safe_max = tr.stack([p[:max(1, len(p)-lead_time)].max() for p in safe_preds])
                fail_max = tr.stack([p[:max(1, len(p)-lead_time)].max() for p in fail_preds])
                loss = loss + tr.nn.functional.softplus(safe_max[:,None] - fail_max).mean() / max_lead_time

            # # lead_time = np.random.randint(min(max_lead_time, len(fail_episode)))
            # # lead_time = min(max_lead_time, len(fail_episode))

            # # feed observations through mu: safe decorrelated max-to-max version batched
            # # safe_predictions = tr.stack([mu(episode).max() for episode in safe_batch])
            # # fail_predictions = tr.stack([mu(episode).max() for episode in fail_batch])
            # safe_predictions = tr.stack([mu(episode[:max(1,len(episode)-max_lead_time)]).max() for episode in safe_batch])
            # fail_predictions = tr.stack([mu(episode[:max(1,len(episode)-max_lead_time)]).max() for episode in fail_batch])

            # # loss = tr.nn.functional.relu(safe_predictions[:,None] - fail_predictions).mean()
            # loss = tr.nn.functional.softplus(safe_predictions[:,None] - fail_predictions).mean()

            # # feed observations through mu: safe decorrelated version
            # safe_input = safe_episode[np.random.randint(len(safe_episode)-lead_time)]
            # fail_inputs = fail_episode[:len(fail_episode)-lead_time]
            # safe_prediction = mu(safe_input.unsqueeze(0)).squeeze()
            # fail_prediction = mu(fail_inputs).max()

            # # feed observations through mu: safe decorrelated max-to-max version
            # safe_inputs = safe_episode[:len(safe_episode)-lead_time]
            # fail_inputs = fail_episode[:len(fail_episode)-lead_time]
            # safe_prediction = mu(safe_inputs).max()
            # fail_prediction = mu(fail_inputs).max()

            # # loss for any threat to disengage rate
            # # loss = tr.nn.functional.relu(safe_prediction - fail_prediction)
            # loss = tr.nn.functional.softplus(safe_prediction - fail_prediction)
    
            # # feed observations through mu
            # safe_inputs = safe_episode[:len(safe_episode)-lead_time]
            # fail_inputs = fail_episode[:len(fail_episode)-lead_time]
            # safe_predictions = mu(safe_inputs).squeeze()
            # fail_predictions = mu(fail_inputs).squeeze()

            # # loss for any threat to disengage rate
            # loss = tr.nn.functional.relu(safe_predictions - fail_predictions.max()).mean()
            # # loss = tr.nn.functional.softplus(safe_predictions - fail_predictions.max()).mean()

            # parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # track metrics
            loss_curve.append(loss.item())
    
            # progress report
            if update % report_period == 0:

                # estimate 'disengage rate' on test split
                with tr.no_grad():
                    safe_max = [mu(episode[:max(1,len(episode)-max_lead_time)]).max().item() for episode in safe_episodes_test]
                    fail_max = [mu(episode[:max(1,len(episode)-max_lead_time)]).max().item() for episode in fail_episodes_test]
                    tau = min(fail_max)
                    acc = float(np.mean([(m <= tau) for m in safe_max]))

                accu_curve.append(acc)
                print(f"{update=}: train loss={loss.item():.5e}, test accuracy = {acc:.3f}")

            # checkpointing
            if (update + 1) % checkpoint_period == 0:
                tr.save(mu.state_dict(), f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_u{update+1}.pt")
    
        training_time = perf_counter()-start_loop
        print(f"Updates done in {training_time:.3f}s ({training_time/(num_updates-start_update)}s per update)")

        tr.save((loss_curve, accu_curve, mu.state_dict(), optimizer.state_dict()), f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_trained.pt")

    if do_show:

        (safe_episodes, fail_episodes, failrate) = tr.load(f"{basename}_buffer.pt", weights_only=True)
        (loss_curve, accu_curve, mu_state_dict, _) = tr.load(f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_trained.pt", weights_only=True)
        mu.load_state_dict(mu_state_dict)

        print(f"{failrate=:.3f}")

        safe_split, fail_split = int(train_fraction * len(safe_episodes)), int(train_fraction * len(fail_episodes))
        safe_episodes, safe_episodes_test = safe_episodes[:safe_split], safe_episodes[safe_split:]
        fail_episodes, fail_episodes_test = fail_episodes[:fail_split], fail_episodes[fail_split:]

        # evaluate trained mu on full buffers
        with tr.no_grad():
            safe_maxs = [mu(episode[:max(1, len(episode)-max_lead_time)]).max() for episode in safe_episodes]
            fail_maxs = [mu(episode[:max(1, len(episode)-max_lead_time)]).max() for episode in fail_episodes]
            safe_maxs_test = [mu(episode[:max(1, len(episode)-max_lead_time)]).max() for episode in safe_episodes_test]
            fail_maxs_test = [mu(episode[:max(1, len(episode)-max_lead_time)]).max() for episode in fail_episodes_test]

        # trendlines
        loss_buckets = np.array(loss_curve).reshape(-1, loss_bucket_size).mean(axis=1)
        accu_buckets = np.array(accu_curve).reshape(-1, accu_bucket_size).mean(axis=1)
    
        pt.subplot(1,4,1)
        pt.plot(loss_curve, '-', color=(.8,)*3)
        pt.plot(np.arange(len(loss_buckets))*loss_bucket_size + loss_bucket_size/2, loss_buckets, 'k-')
        pt.xlabel("Update")
        pt.ylabel("Train MSE")
        pt.yscale("log")
        pt.subplot(1,4,2)
        pt.plot(accu_curve, '-', color=(.8,)*3)
        pt.plot(np.arange(len(accu_buckets))*accu_bucket_size + accu_bucket_size/2, accu_buckets, 'k-')
        pt.xlabel("Update")
        pt.ylabel("Test Acc")
        pt.yscale("log")
        pt.subplot(1,4,3)
        pt.plot([0]*len(fail_maxs), fail_maxs, 'r.', alpha=.5, label="fail")
        pt.plot([1]*len(safe_maxs), safe_maxs, 'b.', alpha=.5, label="safe")
        pt.legend()
        pt.xlabel("Time Remaining")
        pt.ylabel("Prediction")
        pt.title("Train split")
        pt.subplot(1,4,4)
        pt.plot([0]*len(fail_maxs_test), fail_maxs_test, 'r.', alpha=.5, label="fail")
        pt.plot([1]*len(safe_maxs_test), safe_maxs_test, 'b.', alpha=.5, label="safe")
        pt.legend()
        pt.xlabel("Time Remaining")
        pt.ylabel("Prediction")
        pt.title("Test split")
        pt.tight_layout()
        pt.show()


