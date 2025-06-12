from line_profiler import profile # python -m kernprof -lvr policy_iteration.py
import itertools as it
import numpy as np
import scipy.sparse as sp
import gridtrack as gt

@profile
def state_to_index(env, state):
    digits = np.concatenate([state.trackers, state.targets], axis=0).flatten()
    idx = digits @ env.powers
    return idx

@profile
def index_to_state(env, idx):

    digits = []
    for power in reversed(env.powers):
        digits.append(idx // power)
        idx = idx % power

    agents = np.array(list(reversed(digits))).reshape(-1, 2)
    return gt.GridTrackState(env,
        trackers = agents[:env.num_trackers],
        targets = agents[env.num_trackers:])

def get_num_states(env):
    # state calculation (includes unreachable states)
    num_cells = env.shape[0] * env.shape[1]
    return num_cells**(env.num_trackers + env.num_targets)

@profile
def build_mdp(env):

    N = get_num_states(env)
    K = 9 ** env.num_trackers
    r = np.zeros(N)
    
    # sparsity data for fast construction in policy iteration loop
    P = {k: {} for k in range(K)}

    # all possible target motions for vectorization
    all_target_motions = np.array(list(it.product((-1,0,+1), repeat=2*env.num_targets)))

    # all possible actions
    actions = np.array(list(it.product((-1,0,+1), repeat=2*env.num_trackers))).reshape(K, -1, 2)

    for i in range(N):
        # if i == 20: break
        print(f"building {i} of {N}...")

        state = index_to_state(env, i)
        r[i] = env.reward_function(state)

        for k, action in enumerate(actions):

            ## convert all transitions to indices, vectorized for speed
            trackers, all_targets = state.all_transits(action)
            tracker_digits = trackers.flatten()
            target_digits = all_targets.reshape(-1, env.num_targets*2)
            nt2 = 2*env.num_trackers
            js = tracker_digits @ env.powers[:nt2] + target_digits @ env.powers[nt2:]
            P[k][i] = js

    return r, P, actions

@profile
def transition_matrix(env, P, pi):
    N = len(pi)
    rows = [P[k][i] for i,k in enumerate(pi)]
    indptr = np.empty(len(rows)+1)
    indptr[0] = 0
    indptr[1:] = np.array(list(map(len, rows))).cumsum()
    indices = np.concatenate(rows)
    data = np.full(len(indices), 1. / (9.**env.num_targets))
    M = sp.csr_array((data, indices, indptr), shape=(N,N))
    return M

@profile
def policy_iteration(env, r, g, P, decimals=5, num_iters=None):

    N = len(r)
    K = len(P)
    pi = np.zeros(N).astype(int)
    I = sp.eye_array(N)
    u = r # initial guess for u for iterative solvers

    for itr in it.count():
        if itr == num_iters: break

        # build policy-specific sparse transition matrix
        M = transition_matrix(env, P, pi)

        # policy evaluation
        # u_ = sp.linalg.bicgstab(I - g*M, r, x0=u)
        # u_ = sp.linalg.cgs(I - g*M, r, x0=u)
        u = sp.linalg.spsolve(I - g*M, r)
    
        # policy improvement (one-step look-ahead)
        Q = []
        for k in range(K):
            M = transition_matrix(env, P, np.full((N,), k))
            Q.append(M @ u)
        Q = np.stack(Q)
        Q = Q.round(decimals) # round-off error prevents convergence
        pi_new = Q.argmax(axis=0)

        if (pi == pi_new).all(): break

        print(f"policy iteration {itr}, {(pi != pi_new).sum()} changes")
        pi = pi_new
    
    return pi, u

if __name__ == "__main__":

    import pickle as pk
    import matplotlib.pyplot as pt

    do_build = True
    do_pi = True

    # setup discount factor and flag penalty so that flag penalty acts like a hard constraint
    # just a heuristic, idea is that benefit of seeing targets is outweighed by a distant penalty
    # horizon is board size since any flag can be reached (not considering walls) in that time
    # not working perfectly but reasonably close
    num_rows = 4
    num_cols = 5
    num_targets = 1
    T = max(num_rows, num_cols)

    gamma = .9    
    flag_penalty = num_targets * (1 - gamma**(T+2)) / ((1 - gamma) * gamma**(T+1)) + 1
    # input(f"gamma = {gamma}, flag penalty = {flag_penalty}")

    env = gt.GridTrackEnv.randomized(
        num_rows = num_rows,
        num_cols = num_cols,
        num_walls = 2,
        num_trackers = 1,
        num_targets = num_targets,
        num_flags = 3,
        visibility = 1,
        flag_penalty = flag_penalty,
        fully_observable=True,
    )

    # # wall down the middle
    # wall_mask = np.zeros((5,5), dtype=bool)
    # wall_mask[1:, 2] = True
    # flags = np.array([(4,4)])
    # env = gt.GridTrackEnv(
    #     wall_mask, flags,
    #     num_trackers = 1,
    #     num_targets = num_targets,
    #     visibility = 1,
    #     flag_penalty = flag_penalty,
    #     fully_observable=True,
    # )


    # reset to initialize agent positions
    env.reset()
    num_states = get_num_states(env)

    # visualize one state for param sanity check
    pt.ion()
    pt.show()
    env.render(pt.gca(), hang=True, msg=f"num_states={num_states}, look okay?")

    # # test state index mapping
    # # pt.ion()
    # # pt.show()
    # for idx in range(num_states):
    #     env.current_state = state = index_to_state(env, idx)
    #     assert state_to_index(env, state) == idx
    #     # env.render(pt.gca(), hang=False, msg=f"idx={idx} of {num_states}")
    #     # pt.pause(0.01)
    #     print(f"testing idx={idx} of {num_states}")

    # input(f"{num_states} states, {num_states * 9**(env.num_trackers + env.num_targets)} nonzeros in P, build?")

    env_params = "_".join(map(str, [
        env.num_rows,
        env.num_cols,
        env.num_trackers,
        env.num_targets,
        len(env.walls),
        len(env.flags),
        env.visibility,
    ]))
    build_fname = f"mdp_{env_params}.pkl"
    if do_build:
        r, P, actions = build_mdp(env)
        with open(build_fname, "wb") as f:
            pk.dump((env, gamma, r, P, actions), f)

    with open(build_fname, "rb") as f:
        (env, gamma, r, P, actions) = pk.load(f)

    pi_fname = f"pi_{env_params}.pkl"
    if do_pi:
        pi, u = policy_iteration(env, r, gamma, P, num_iters=500)
        with open(pi_fname, "wb") as f:
            pk.dump((pi, u), f)

    with open(pi_fname, "rb") as f:
        (pi, u) = pk.load(f)

    # pt.plot(r, label='r')
    # pt.plot(u, label='u')
    # pt.legend()

    # pt.figure()
    # for k in range(9):
    #     M = transition_matrix(env, P, np.full((len(pi),), k))
    #     pt.subplot(3,3,k+1)
    #     pt.imshow(M.toarray())
    # pt.show()

    # run optimal policy

    # ## live plot
    # fig, axs = pt.subplots(1,2)
    # pt.ion()
    # pt.show()

    # env.reset()
    # for t in range(1000):
    #     i = state_to_index(env, env.current_state)
    #     k = pi[i]
    #     action = actions[k]

    #     observation, reward, terminated, truncated, info = env.step(action)

    #     obs_img = observation[:,:,:3]
    #     obs_img[:,:,1] += observation[:,:,3]*.5
    #     axs[0].imshow(obs_img)
    #     env.render(axs[1], hang=False, msg=f"t={t}: reward={reward}")
    #     pt.pause(0.01)


    # ## animation
    # import matplotlib.animation as animation

    # # precompute intermediate states along plan
    # env.reset()
    # states = []
    # for t in range(100):
    #     i = state_to_index(env, env.current_state)
    #     states.append(i)
    #     k = pi[i]
    #     action = actions[k]
    #     observation, reward, terminated, truncated, info = env.step(action)

    # # animate the state sequence
    # fig, axs = pt.subplots(1,2)    
    # def drawframe(n):
    #     i = states[n]
    #     env.current_state = index_to_state(env, i)
    #     observation = env._observe_state()
        
    #     obs_img = observation[:,:,:3]
    #     obs_img[:,:,1] += observation[:,:,3]*.5
    #     axs[0].imshow(obs_img)
    #     env.render(axs[1], hang=False, msg=f"t={n}")

    # # blit=True re-draws only the parts that have changed.
    # print("animating...")
    # anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=500, blit=False)
    # print("saving...")
    # anim.save("gridtrack.mp4")

