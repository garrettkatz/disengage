# Conformal Preemption

This repository contains code for the paper:

Conformal Preemption of Failures in Sequential Decision-Making Agents. Anonymous et al.

The last commit at the time of paper submission is: 2777f69303fb910bb3ef615d4039c78f6cc33d90 from Jan 28, 2026

## Installation

Our code depends on several common Python libraries for machine learning and scientific computing, including numpy, scipy, pandas, matplotlib, and pytorch, as well as line_profiler, so those will need to be installed.

The experiments use several existing RL benchmarks with their own dependencies which also must be installed.  We recommend installing these benchmarks and dependencies in their own virtual environments.  The following steps worked for us on Ubuntu:

### gym-pybullet-drones

We did not use a virtual environment for gym-pybullet-drones, though you may want to.  You can follow the installation instructions [here](https://github.com/utiasDSL/gym-pybullet-drones#installation).


### Stable Baselines 3

First install with
```
$ sudo apt-get install swig cmake ffmpeg
$ git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
$ cd rl-baselines3-zoo/
$ sudo apt-get install python3.12-venv
$ python -m venv .rlzoo
$ source .rlzoo/bin/activate
$ pip install "autorom[accept-rom-license]"
$ pip install -r requirements.txt
$ pip install -e .[plots,tests]
```

Then when running any SB3-related experiments, wrap them in
```
$ source .rlzoo/bin/activate
$ python ...
$ deactivate
```

You will need to prepend a relative path in front of `.rlzoo/bin/activate` when running our code from the same directory as this README file.

### Humanoid Bench:

First install with the following commands (you will also need uv installed):
```
$ git clone https://github.com/carlosferrazza/humanoid-bench.git
$ cd humanoid-bench
$ uv venv .humanoid-bench --python 3.11
$ source .humanoid-bench/bin/activate
$ uv pip install -e .
$ uv pip install "jax[cpu]==0.4.28"
$ uv pip install -r requirements_jaxrl.txt
$ uv pip install -r requirements_dreamer.txt
$ uv pip install -r requirements_tdmpc.txt
$ uv pip install stable-baselines3==2.3.2
```

After installation, some Humanoid Bench code raised errors about missing assets, but we resolved these by copying some existing assets to have lower-case file extensions:

```
$ cp humanoid_bench/assets/mjx/assets/left_ankle_link.STL humanoid_bench/assets/mjx/assets/left_ankle_link.stl
$ cp humanoid_bench/assets/mjx/assets/right_ankle_link.STL humanoid_bench/assets/mjx/assets/right_ankle_link.stl
$ python -m humanoid_bench.mjx.mjx_test --with_full_model # should work after stls renamed
```

Now when running any Humanoid-Bench-related experiments, wrap them in
```
$ source .humanoid-bench/bin/activate
$ python ...
$ deactivate
```

You will need to prepend a relative path in front of `.humanoid-bench/bin/activate` when running our code from the same directory as this README file.

## Running the code

Paper results can be regenerated with the following scripts, which rely on some other scripts in this repository:

### Tabular Examples

Run `fmp.py` to generate Figure 2. In the `main()` function set `do_reps=False` to visualize previously generated results, and `do_reps=True` to rerun the experiment before visualization.

### Ruins Policy

Run `train_ruins.py` to retrain a PPO policy from scratch (may take about a day).  Then run `ruins_plot_ppo.py` to plot the reward curve in Figure 4 (right).

After training, determine the model checkpoint you want to use, and update the `policy_checkpoint_name` variable in the `EpisodeRunner` class of `ruins_conditional.py` accordingly.

### Original Failure Rates

Run `[env]_failure_rates.py` to generate the data for Figure 5, where `[env]` is one of `cartpole`, `lunar_lander`, `humanoid`, or `ruins`.  Make sure to activate the respective virtual environments before each one.  The first time, you may also need to create the subdirectories `cp_data`, `ll_data`, `hu_data,` and `ru_data` where results will be saved.

Each script will run 1000 episodes to estimate failure rates which may take about an hour.  To only visualize previously generated results, set the `do_rollouts` variable to `False`.

### Mu Training

Run `[env]_conditional.py` to generate the data for Figure 6, again replacing `[env]` and activating virtual environments accordingly.  Set the variable `do_sampling` to `True` to regenerate episode buffers, and `do_training` to `True` to rerun the training.  Combined these steps may take several hours.  They also save many checkpoints in the `[env]_data` subdirectories, so make sure you have several GB of available space or reduce the `num_update`/`checkpoint_period` variables.

When training is done, run `all_mu_conditional.py` to generate Figure 6.

### Conformal Preemption

Run `[env]_preemption_conditional.py` to generate the data for the remaining figures.  Set the variable `do_conform` to `True` to rerun the experiments and `False` to only visualize previously generated results.  The visualizations are Figures 7 and 9-11.

After regenerating results for all environments, run `all_preemption.py` to create the summary Figure 8.

