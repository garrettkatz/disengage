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
