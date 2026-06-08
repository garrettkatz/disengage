if __name__ == "__main__":

    import pickle as pk
    import zmq
    import numpy as np
    import torch as tr
    import time
    import ruins_utils as ru

    do_test = False
    do_show = True
    render = False

    if do_test:
    
        # set up ruins environment and policy
        env, ppo_policy = ru.load("41652_ppo_drone", render)
        
        # set up ZeroMQ client for jetson communication
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        # Connect to Jetson Nano's USB-C IP
        jetson_ip = "192.168.55.1" 
        socket.connect(f"tcp://{jetson_ip}:5555")
    
        # start an episode 
        obs, info = env.reset()
        observations = [obs]
        actions = []
        rewards = []
        alarms = []
        watchdog_times = []
        failed = False
    
        start = time.time()
        max_steps = env.EPISODE_LEN_SEC*env.CTRL_FREQ
        for i in range(1, max_steps):
    
            start_jetson = time.perf_counter()
    
            # Send state to Jetson
            state = obs
            # img = np.random.randint(255, size=(48,64), dtype=np.uint8)
            # socket.send_multipart([state.tobytes(), img.tobytes()], copy=False)
            img = np.random.randint(255, size=(48,64), dtype=np.uint8)
            socket.send_multipart([state.tobytes()], copy=False)
    
            # receive result
            frames = socket.recv_multipart(copy=False)
            alarm = np.frombuffer(frames[0], dtype=bool)
    
            jetson_time = time.perf_counter() - start_jetson
    
            alarms.append(alarm)
            watchdog_times.append(jetson_time)
    
            with tr.no_grad():
                action = ppo_policy.select_action(obs)
    
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            failed = env.failure_predicate() or (truncated and not terminated)
            rewards.append(reward)
    
            # # perturb after checking failure
            # if perturb_obs is not None: obs = perturb_obs(obs)
    
            observations.append(obs)
            actions.append(action)
    
            print(f"timestep {i} of {max_steps}: {jetson_time=:.3f}, reward = {sum(rewards):.3f}, {terminated=}, {truncated=}, {failed=}")
            if render:
                env.render()
                ru.sync(i, start, env.CTRL_TIMESTEP)
    
            if failed:
                # print("Failed!")
                break
    
            if terminated or truncated:
                # if truncated: print("Trunk!")
                break
    
        # clear buffer
        ppo_policy.buffer.clear()
    
        # timing results
        watchdog_times = np.array(watchdog_times)
        control_period = 1. / env.CTRL_FREQ
        print(f"watchdog times = {watchdog_times.mean()} +/- {watchdog_times.std()} <= {watchdog_times.max()}")
        print(f"control period = {control_period}")
        with open("oer.pkl","wb") as f:
            pk.dump((watchdog_times, control_period), f)

    if do_show:

        with open("oer.pkl","rb") as f:
            (watchdog_times, control_period) = pk.load(f)

        import matplotlib as mpl
        import matplotlib.pyplot as pt
        mpl.rcParams['font.family'] = 'serif'

        # pt.figure(figsize=(6,5))
        pt.figure(figsize=(3,2))
        pt.hist(watchdog_times)
        # pt.plot([control_period, control_period], [0,100], 'r--', label="Simulation Control Period")
        pt.plot([control_period, control_period], [0,100], 'r--', label="Control Period")
        pt.legend()

        ax_sec = pt.gca()
        # ax_sec.set_xlabel("Per-timestep preemption processing time (Seconds)")
        ax_sec.set_xlabel("Processing time (Seconds)")
        # ax_sec.set_ylabel(f"Count (out of {len(watchdog_times)} time-steps total)")
        ax_sec.set_ylabel(f"Count")

        # ax_freq = ax_sec.twiny()
        # ax_freq.set_xlim(ax_sec.get_xlim())
        # ax_freq.set_xbound(ax_sec.get_xbound())
        # ax_freq.set_xlabel('Per-timestep preemption processing frequency (Hz)')
        # ax_freq.xaxis.set_major_formatter(pt.FuncFormatter(lambda x, pos: f"{1./x:.2f}" if x > 0 else ""))

        # pt.title("Jetson Timing Histogram")
        pt.tight_layout()
        pt.savefig("oer.pdf")
        pt.show()

