# based on https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/enjoy.py
import os, sys, yaml
from line_profiler import profile # python -m kernprof -lvr thisfile.py
import numpy as np
import torch as tr
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import get_model_path
from rl_zoo3.load_from_hub import download_from_hub

def load(env, alg_name, render=False):

    env_name = EnvironmentName(env)
    algo = alg_name
    gym_packages = []
    folder = "rl-trained-agents"
    seed = np.random.randint(2**16)
    load_best = False
    load_checkpoint = None
    load_last_checkpoint = False
    num_threads = 0
    verbose = 0
    norm_reward = False
    n_envs = 1
    device="auto"

    # Going through custom gym packages to let them register in the global registory
    for env_module in gym_packages:
        importlib.import_module(env_module)

    try:
        _, model_path, log_path = get_model_path(
            0,
            folder,
            algo,
            env_name,
            load_best,
            load_checkpoint,
            load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=0,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                0,
                folder,
                algo,
                env_name,
                load_best,
                load_checkpoint,
                load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        tr.set_num_threads(num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    log_dir = None

    env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        vec_env_cls=ExperimentManager.default_vec_env_cls,
    )

    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            # load models with different obs bounds
            # Note: doesn't work with channel last envs
            # "observation_space": env.observation_space,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)

    return env, model

# @profile
def run(env, model, n_timesteps, failure_predicate, render=False, stochastic=False):
    deterministic = not stochastic

    obs = env.reset()

    observations = [obs]
    actions = []

    failure = False
    ep_len = 0
    ep_rew = 0
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    for timestep in range(n_timesteps):

        action, lstm_states = model.predict(
            obs,  # type: ignore[arg-type]
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        obs, reward, done, infos = env.step(action)    
        episode_start = done

        observations.append(obs)
        actions.append(action)

        if render: env.render("human")

        ep_rew += reward[0]
        ep_len += 1

        failure = failure_predicate(env, obs, reward, done, infos)
        if failure: done = True

        if done: break

    return failure, ep_rew, ep_len, observations, actions

if __name__ == "__main__":
    # model, _ = load_model("a2c", "LunarLander-v3")

    env = "LunarLander-v3"
    algo = "a2c"
    n_timesteps = 1000
    render = True
    def failure_predicate(env, obs, reward, done, infos): return (reward[0] == -100)

    # env = "CartPole-v1"
    # algo = "dqn"
    # n_timesteps = 500
    # render = False
    # # Farama docs say fail when:
    # # Pole Angle is greater than ±12°
    # # Termination: Cart Position is greater than ±2.4 
    # def failure_predicate(env, obs, reward, done, infos):
    #     pos, ang = obs[0,0], obs[0,2]
    #     return (abs(ang) >= .2095) or (abs(pos) >= 2.4)

    env, model = load(env, algo, render)
    print(env)
    # print(model.q_net) # dqn

    failure, ep_rew, ep_len, observations, actions = run(
        env, model, n_timesteps, failure_predicate, render, stochastic=False)

    print(f"{len(observations)} timesteps, last obs:")
    print(observations[-1])
    input(f"{failure=:b}")


