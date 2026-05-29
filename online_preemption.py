

if __name__ == "__main__":

    import zmq
    import numpy as np
    import torch as tr
    tr.set_grad_enabled(False) # no training right now

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

    # networking for host environment
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # Bind to all interfaces on port 5555
    socket.bind("tcp://0.0.0.0:5555")
    print("Jetson Server is ready, waiting for PyBullet states...")

    # get a message

    # state_buffer = socket.recv(copy=False)
    # state = np.frombuffer(state_buffer, dtype=np.float64)#.reshape(12)
    frames = socket.recv_multipart(copy=False)
    state = np.frombuffer(frames[0], dtype=np.float64)
    img = np.frombuffer(frames[1], dtype=np.uint8).reshape((48,64))
    print("Message received:")
    print(state)
    print(img[:2,:3])

    # message = socket.recv_json()
    # state_list = message['state']
    # print("Message received:")
    # print(state_list)

    # process an observation
    obs = tr.randn(12)
    pred = mu(obs)
    print(f"{pred=} >= {tau=}? {pred >= tau}")


