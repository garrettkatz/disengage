

if __name__ == "__main__":

    import zmq
    import numpy as np
    import torch as tr
    tr.set_grad_enabled(False) # no training right now

    use_gpu = True
    conformity_model_path = "conformity_model.pt"
    L = 3 # lead time
    tau = .9

    # copied from ruins_conditional.py
    def setup_mu():
        # MLP - same architecture as critic
        num_hidden = 64
        return tr.nn.Sequential(
            tr.nn.Linear(in_features=12, out_features=num_hidden, bias=True),
            tr.nn.Tanh(),
            tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
            tr.nn.Tanh(),
            tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
        )

    # load pretrained model weights
    (_, _, _, mu_state_dicts) = tr.load(conformity_model_path, weights_only=True)
    mu = setup_mu()
    mu.load_state_dict(mu_state_dicts[L])

    cpu = tr.device("cpu")
    if use_gpu:
        assert tr.cuda.is_available()
        device = tr.device("cuda:0")
        mu.to(device)
    else:
        device = cpu
    print(f"using device {device}")

    # networking for host environment
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # Bind to all interfaces on port 5555
    socket.bind("tcp://0.0.0.0:5555")
    print("Jetson Server is ready, waiting for PyBullet states...")

    # busy loop listening
    while True:
        print("Requesting message...")

        # get the observation message
        frames = socket.recv_multipart(copy=False)
        state = np.frombuffer(frames[0], dtype=np.float64)
        # img = np.frombuffer(frames[1], dtype=np.uint8).reshape((48,64))
        print("Message received:")
        print(state)
        # print(img[:2,:3])

        # process an observation
        state = tr.tensor(state).to(tr.float32)
        if use_gpu: state = state.to(device)
        pred = mu(state)
        if use_gpu: pred = pred.to(cpu)
        print(f"{pred=} >= {tau=}? {pred >= tau}")

        # send prediction back
        alarm = (pred >= tau).squeeze().numpy().astype(bool)
        socket.send_multipart([alarm.tobytes()], copy=False)


