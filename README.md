# Conformal Preemption

This repository contains code for the paper:

Conformal Preemption of Failures for Sequential Decision Making. Anonymous et al.

# Installation

The experiments use several existing RL benchmarks with their own dependencies.  We recommend installing these benchmarks and dependencies in their own virtual environments.  These commands worked for us on Ubuntu:

## Stable Baselines 3

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

## Humanoid Bench:

First install with
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

## gym-pybullet-drones

We did not use a virtual environment for gym-pybullet-drones, though you may want to.  You can follow the installation instructions at

https://github.com/utiasDSL/gym-pybullet-drones#installation

# Running the code

Paper results can be regenerated with the following scripts, which rely on some other scripts in this repository:

## 

