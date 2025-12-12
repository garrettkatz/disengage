# disengage

using stable baselines 3: first install with
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
Then you can run
```
$ source .rlzoo/bin/activate
$ python enjoy.py --algo dqn --env SpaceInvadersNoFrameskip-v4
$ deactivate
```

using humanoid-bench:
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
$ python -m humanoid_bench.test_env --env h1hand-walk-v0 # random actions but should work
$ cp humanoid_bench/assets/mjx/assets/left_ankle_link.STL humanoid_bench/assets/mjx/assets/left_ankle_link.stl # for some reason misnamed?
$ cp humanoid_bench/assets/mjx/assets/right_ankle_link.STL humanoid_bench/assets/mjx/assets/right_ankle_link.stl # for some reason misnamed?
$ python -m humanoid_bench.mjx.mjx_test --with_full_model # should work after stls renamed
```
now change to this repo folder and use test_humanoid.py / humanoid_utils to do something like mjx_test
