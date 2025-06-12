import pickle as pk
import numpy as np
from policy_iteration import index_to_state

if __name__ == "__main__":

    # try linear and quadratic fits to true value functions on several gridtracks of increasing size

    # load a small gridtrack soln

    env_params = "4_5_1_1_6_3_1"
    # this env has linear|quadratic mean utility residuals around 2.18|0

    # env_params = "3_5_1_2_5_1_2"
    # # this env has linear|quadratic mean utility residuals around 2.97|0.21

    build_fname = f"mdp_{env_params}.pkl"
    with open(build_fname, "rb") as f:
        (env, gamma, r, P, actions) = pk.load(f)

    pi_fname = f"pi_{env_params}.pkl"
    with open(pi_fname, "rb") as f:
        (pi, u) = pk.load(f)

    # get all observations for all states
    obs = []
    for idx in range(len(r)):
        env.current_state = index_to_state(env, idx)
        obs.append(env._observe_state().flatten())
    obs = np.stack(obs)
    obs = np.concatenate((obs, np.ones((len(obs),1))), axis=1) # bias
    print(obs.shape)

    # try linear fit
    w_lin = np.linalg.lstsq(obs, u, rcond=None)[0]
    print(w_lin)
    residuals = np.fabs(obs @ w_lin - u)
    print(f"linear residual max={residuals.max()}, mean={residuals.mean()}")

    # try quadratic fit
    features = (obs[:,None,:] * obs[:,:,None]).reshape(len(obs), -1)
    print(features.shape)
    w_quad = np.linalg.lstsq(features, u, rcond=None)[0]
    print(w_quad)
    residuals = np.fabs(features @ w_quad - u)
    print(f"quadratic residual max={residuals.max()}, mean={residuals.mean()}")

    # run linear value-fitted agent and measure how many disengagements

    # # try value iteration algorithm with linear model
    # N = 30
    # M = 10
    # num_itrs = 1
    # ws = {0: np.zeros(obs.shape[1])}
    # for itr in range(num_itrs):

    #     # sample base points
    #     idxs = np.random.randint(len(obs), size=N)

    #     # sparse P assumption: get all transitions, not a random sample
    #     for idx in idxs:
    #         pass
        


