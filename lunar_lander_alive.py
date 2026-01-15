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
    do_sampling = False
    do_training = True
    do_show = True

    env_name = "LunarLander-v3"
    alg_name = "a2c"
    num_timesteps = 1000
    num_rollouts = 200 # 200 rollouts * 1000 timesteps = 200K is parity with original training
    gamma = .99 # discount factor for alive bonuses, tuned to be relatively sensitive within up to 100-step horizon
    blend = 0.0001 # blend factor for target mu network
    learning_rate = 0.05
    td_steps = 1

    num_updates = 5000 # 5000 is parity with original training
    batch_size = 40
    report_period = 10
    checkpoint_period = 100
    bucket_size = 100 # buckets for learning curve trendlines
    basename = "ll_data/ll_alive"

    # initializations
    alive_bonus = 1 - gamma # per-step alive bonus, entails maximum discounted return in [0,1]
    mu = setup_mu()
    mu_target = setup_mu()
    mu_target.load_state_dict(mu.state_dict()) # initialize with same parameters
    for t in mu_target.parameters(): t.requires_grad = False # but no gradients
    env, model = su.load(env_name, alg_name)
    mse = tr.nn.MSELoss()
    # optimizer = tr.optim.Adam(mu.parameters(), lr=learning_rate)
    optimizer = tr.optim.SGD(mu.parameters(), lr=learning_rate)

    # fill buffer
    if do_sampling:
        current_buffer = [] # current observation
        td_buffer = [] # observations for bootstrapping 
        fail_buffer = [] # whether observation is in a failed episode
        time_buffer = [] # time to end of episode (undiscounted alive bonus return)
        num_failures = 0
        start_loop = perf_counter()
        for rollout in range(num_rollouts):
    
            # simulate rollout
            failure, _, _, observations, _ = su.run(env, model, num_timesteps, failure_predicate, perturb_action=perturb_action)
            if failure: num_failures += 1
    
            # extend buffers
            current_buffer.extend(observations)
            # td_buffer.extend([observations[-1]] * len(observations))
            td_buffer.extend(observations[td_steps:] + [observations[-1]] * min(len(observations), td_steps))
            fail_buffer.extend([failure] * len(observations))
            time_buffer.extend(reversed(range(len(observations))))
    
        current_buffer = tr.tensor(np.concatenate(current_buffer, axis=0)).to(tr.float32)
        td_buffer = tr.tensor(np.concatenate(td_buffer, axis=0)).to(tr.float32)
        fail_buffer = tr.tensor(fail_buffer).to(tr.int32) # 0|1
        time_buffer = tr.tensor(time_buffer).to(tr.float32)
        failrate = num_failures / num_rollouts
        # print(current_buffer.shape)
        # print(fail_buffer)
        # print(time_buffer)
        print(f"Rollouts done in {perf_counter()-start_loop:.3f}s, {failrate=:.3f}")

        tr.save((current_buffer, td_buffer, fail_buffer, time_buffer, failrate), f"{basename}_buffer.pt")

    if do_training:

        (current_buffer, td_buffer, fail_buffer, time_buffer, failrate) = tr.load(f"{basename}_buffer.pt", weights_only=True)

        # do training
        start_loop = perf_counter()
        loss_curve = []
        for update in range(num_updates):
    
            # sample a batch
            batch_idx = np.random.randint(len(current_buffer), size=batch_size)
    
            # feed observations through mu
            predictions = mu(current_buffer[batch_idx]).squeeze()
            with tr.no_grad(): # don't backprop through bootstrap
                bootstraps = mu_target(td_buffer[batch_idx]).squeeze()
            time_remaining = time_buffer[batch_idx]
            fail_mask = fail_buffer[batch_idx]

            bootstrap_offset = tr.minimum(tr.tensor(td_steps), time_remaining)

            # learning target for n-step TD, where n is until episode end
            labels = alive_bonus * (1 - gamma**bootstrap_offset) / (1 - gamma)
    
            # add bootstrap unless failed and time is up
            bootstrap_mask = tr.maximum(1 - fail_mask, time_remaining > td_steps).to(tr.float32)
            labels += gamma**bootstrap_offset * bootstraps * bootstrap_mask
    
            # loss and semigradient update
            loss = mse(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # target network update
            for t, p in zip(mu_target.parameters(), mu.parameters()): t.data.lerp_(p.data, blend)
    
            # track metrics
            loss_curve.append(loss.item())
    
            # progress report
            if update % report_period == 0:
                print(f"{update=}: loss={loss.item():.5e}")

            # checkpointing
            if (update + 1) % checkpoint_period == 0:
                tr.save(mu.state_dict(), f"{basename}_r{num_rollouts}_lr{learning_rate}_g{gamma}_b{blend}_u{update+1}.pt")
    
        training_time = perf_counter()-start_loop
        print(f"Updates done in {training_time:.3f}s ({training_time/num_updates}s per update)")

        tr.save((loss_curve, mu.state_dict()), f"{basename}_r{num_rollouts}_lr{learning_rate}_g{gamma}_b{blend}_trained.pt")

    if do_show:

        (current_buffer, td_buffer, fail_buffer, time_buffer, failrate) = tr.load(f"{basename}_buffer.pt", weights_only=True)
        (loss_curve, mu_state_dict) = tr.load(f"{basename}_r{num_rollouts}_lr{learning_rate}_g{gamma}_b{blend}_trained.pt", weights_only=True)
        mu.load_state_dict(mu_state_dict)

        print(f"{failrate=:.3f}")

        # evaluate trained mu on full buffer, no bootstraps
        with tr.no_grad():
            predictions = mu(current_buffer).squeeze()
            labels = alive_bonus * (1 - gamma**time_buffer) / (1 - gamma)

        # loss trendline
        buckets = np.array(loss_curve).reshape(-1, bucket_size).mean(axis=1)
    
        pt.subplot(1,2,1)
        pt.plot(loss_curve, '-', color=(.8,)*3)
        pt.plot(np.arange(len(buckets))*bucket_size + bucket_size/2, buckets, 'k-')
        pt.xlabel("Update")
        pt.ylabel("MSE")
        pt.yscale("log")
        pt.subplot(1,2,2)
        pt.plot(predictions[fail_buffer==1], labels[fail_buffer==1], 'r.', alpha=.5, label="fail")
        pt.plot(predictions[fail_buffer==0], labels[fail_buffer==0], 'b.', alpha=.5, label="safe")
        pt.plot([0, 1], [0, 1], ':', color=(.8,)*3)
        pt.legend()
        pt.xlabel("Prediction")
        pt.ylabel("Label")
        pt.tight_layout()
        pt.show()
