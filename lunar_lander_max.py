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

    resume = False
    do_sampling = True
    do_training = True
    do_show = True

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    num_rollouts = 200 # 200 rollouts * 1000 timesteps = 200K is parity with original training
    learning_rate = 0.001
    max_lead_time = 10

    num_updates = 2000 # 5000 is parity with original training
    batch_size = 40 # 40 observation pairs is parity with original training
    report_period = 10
    checkpoint_period = 1000
    bucket_size = 100 # buckets for learning curve trendlines
    basename = "ll_data/ll_max"

    # initializations
    mu = setup_mu()
    env, model = su.load(env_name, alg_name)
    optimizer = tr.optim.Adam(mu.parameters(), lr=learning_rate)
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

        # do training
        start_loop = perf_counter()
        loss_curve = []
        for update in range(num_updates):
    
            # sample episodes
            safe_batch = [safe_episodes[i] for i in np.random.randint(len(safe_episodes), size=batch_size)]
            fail_batch = [fail_episodes[i] for i in np.random.randint(len(fail_episodes), size=batch_size)]

            # lead_time = np.random.randint(min(max_lead_time, len(fail_episode)))
            # lead_time = min(max_lead_time, len(fail_episode))

            # feed observations through mu: safe decorrelated max-to-max version batched
            safe_predictions = tr.stack([mu(episode).max() for episode in safe_batch])
            fail_predictions = tr.stack([mu(episode).max() for episode in fail_batch])

            # loss = tr.nn.functional.relu(safe_predictions[:,None] - fail_predictions).mean()
            loss = tr.nn.functional.softplus(safe_predictions[:,None] - fail_predictions).mean()

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
                print(f"{update=}: loss={loss.item():.5e}")

            # checkpointing
            if (update + 1) % checkpoint_period == 0:
                tr.save(mu.state_dict(), f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_u{update+1}.pt")
    
        training_time = perf_counter()-start_loop
        print(f"Updates done in {training_time:.3f}s ({training_time/num_updates}s per update)")

        tr.save((loss_curve, mu.state_dict()), f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_trained.pt")

    if do_show:

        (safe_episodes, fail_episodes, failrate) = tr.load(f"{basename}_buffer.pt", weights_only=True)
        (loss_curve, mu_state_dict) = tr.load(f"{basename}_r{num_rollouts}_lr{learning_rate}_lt{max_lead_time}_trained.pt", weights_only=True)
        mu.load_state_dict(mu_state_dict)

        print(f"{failrate=:.3f}")

        # evaluate trained mu on full buffers
        with tr.no_grad():
            safe_maxs = [mu(episode).max() for episode in safe_episodes]
            fail_maxs = [mu(episode).max() for episode in fail_episodes]
        # safe_leads = tr.tensor([t for episode in safe_episodes for t in reversed(range(len(episode)))])
        # fail_leads = tr.tensor([t for episode in fail_episodes for t in reversed(range(len(episode)))])

        # loss trendline
        buckets = np.array(loss_curve).reshape(-1, bucket_size).mean(axis=1)
    
        pt.subplot(1,2,1)
        pt.plot(loss_curve, '-', color=(.8,)*3)
        pt.plot(np.arange(len(buckets))*bucket_size + bucket_size/2, buckets, 'k-')
        pt.xlabel("Update")
        pt.ylabel("MSE")
        pt.yscale("log")
        pt.subplot(1,2,2)
        # pt.plot(fail_leads[:1000], fail_predictions[:1000], 'r.', alpha=.5, label="fail")
        # pt.plot(safe_leads[:1000], safe_predictions[:1000], 'b.', alpha=.5, label="safe")
        pt.plot([0]*len(fail_maxs), fail_maxs, 'r.', alpha=.5, label="fail")
        pt.plot([1]*len(safe_maxs), safe_maxs, 'b.', alpha=.5, label="safe")
        pt.legend()
        pt.xlabel("time remaining")
        pt.ylabel("prediction")
        pt.tight_layout()
        pt.show()


