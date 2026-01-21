from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import sb_utils as su
import train_mu_conditional as tm

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

def failure_predicate(env, obs, reward, done, infos):
    # Farama docs say fail when:
    # Cart Position is greater than ±2.4 
    # Pole Angle is greater than ±12°
    pos, ang = obs[0,0], obs[0,2]
    return bool((abs(pos) >= 2.4) or (abs(ang) >= .2095))

class EpisodeRunner:

    def __init__(self, obs_noise, max_episode_length):
        self.env, self.model = su.load("CartPole-v1", "dqn")
        self.max_episode_length = max_episode_length
        self.perturb_obs = tm.perturb_obs_factory(obs_noise)

    def __call__(self):
        return su.run(self.env, self.model, self.max_episode_length, failure_predicate, perturb_obs=self.perturb_obs)

    def close(self):
        pass

if __name__ == "__main__":

    do_sampling = False
    do_training = False
    do_show = True

    # parity with original policy training:
    # 50K timesteps in buffer
    # 25000 updates * 64 batch size = 1,600,000 states fed through network

    for rep in range(5):
        print(f"\n\n ******* {rep=} ********\n\n")

        params = {
            "basename": "cp_data/cp_cond",
            "buf_rep": rep,
            "train_rep": rep,
    
            "obs_noise": .4,
            "max_episode_length": 500,
            "total_timesteps": 50_000,
    
            "max_leadtime": 25,
            "num_updates": 25000,
            "train_batch_size": 64,
            "valid_batch_size": 5_000,
    
            "learning_rate": 1e-5,
            "weight_decay": .5,
            "train_fraction": .8, # fraction of rollouts used for training
    
            "report_period": None,
            "checkpoint_period": 100,
            "bucket_size": 10, # buckets for learning curve trendlines
        }
    
        if do_sampling:
            tm.collect_data(params, EpisodeRunner(params["obs_noise"], params["max_episode_length"]))
    
        if do_training:
            tm.train(params, setup_mu)
    
        # if do_show:
        #     tm.show_results(params, setup_mu, [1, 2, 5, 10, 25])

    if do_show:

        pt.rcParams['font.family'] = 'serif'
        pt.rcParams['font.size'] = 12

        Ls = [1, 2, 5, 10, 25]
        valid_curves = {L: [] for L in Ls}

        for rep in range(5):
    
            params = {
                "basename": "cp_data/cp_cond",
                "buf_rep": rep,
                "train_rep": rep,
        
                "obs_noise": .4,
                "max_episode_length": 500,
                "total_timesteps": 50_000,
        
                "max_leadtime": 25,
                "num_updates": 25000,
                "train_batch_size": 64,
                "valid_batch_size": 5_000,
        
                "learning_rate": 1e-5,
                "weight_decay": .5,
                "train_fraction": .8, # fraction of rollouts used for training
        
                "report_period": None,
                "checkpoint_period": 100,
                "bucket_size": 10, # buckets for learning curve trendlines
            }

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
            for L in range(1, max_leadtime+1):
                mu[L] = setup_mu()
                mu[L].load_state_dict(mu_state_dicts[L])

            # make train/validation splits
            num_split = int(len(rollouts)*train_fraction)
            train_rollouts, valid_rollouts = rollouts[:num_split], rollouts[num_split:]
            train_failures, valid_failures = failures[:num_split], failures[num_split:]

            for L in Ls:
                valid_curves[L].append(loss_curve[L]["valid"])

        pt.figure(figsize=(15,3))

        for n, L in enumerate(Ls):
            valid_curves[L] = np.stack(valid_curves[L])
            mean = valid_curves[L].mean(axis=0)
            stdv = valid_curves[L].std(axis=0)
            chkpt = checkpoint_period * np.arange(1, len(loss_curve[L]["valid"])+1)

            pt.subplot(1, len(Ls)+1, n+1)
            pt.title(f"L = {L}")
            pt.fill_between(chkpt, mean-stdv, mean+stdv, color=(1,.5,.5), alpha=.5)
            pt.plot(chkpt, mean, 'k-')
            if n == 0: pt.ylabel("MSE")
            if n==len(Ls)//2: pt.xlabel("Update")
            # pt.yscale("log")

        pt.subplot(1, len(Ls)+1, len(Ls)+1)
        L = 1

        # predict on a validation batch for visualization
        with tr.no_grad():
            current_obs, next_obs, next_fail = tm.sample_batch(valid_rollouts, valid_failures, max_episode_length, valid_batch_size)
            predictions = mu[L](current_obs).squeeze()
            targets = next_fail if L==1 else mu[L-1](next_obs).squeeze()

        pt.plot(predictions, targets, 'k.')
        pt.plot([0, 1], [0, 1], ':', color=(.8,)*3)
        pt.xlabel("Prediction")
        pt.ylabel("Target")
        pt.title(f"L = {L}")

        pt.tight_layout()
        pt.savefig("cp_mu.pdf")
        pt.show()
        

