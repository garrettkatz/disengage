import itertools as it
import numpy as np
import cvxpy as cp

def sample_mp(S, L):
    """
    S states, at least L steps to get to last state which is failure
    """

    # make transition probabilities
    # P[i,j] = P(s'=j|s=i)

    P = np.tril(np.random.rand(S,S), k=S-L) # takes at least L steps to get to last state
    # P[-1,:-1] = 0 # last state is failure and absorbing
    P[-1,:-2] = 0 # last state is failure and almost absorbing (could backtrack 1)
    P[-1,-1] += 3
    P /= P.sum(axis=1, keepdims=True)

    # initial state distribution (unlikely to start in failure)
    p0 = np.random.rand(S)
    p0[:-1] += 3
    p0 /= p0.sum()

    # failure indicator
    f = np.zeros(S, dtype=bool)
    f[-1] = True

    return p0, P, f

def brute(p0, P, f, T):

    # all possible paths up to T
    paths = [[]]
    for t in range(T+1):
        paths = [p + [s] for p in paths for s in range(S) if len(p)==0 or P[p[-1], s] > 0]
    paths = np.array(paths)

    # leading path probabilities: path_probs[p,t] = Pr(paths[p][:t+1])
    path_probs = np.ones((len(paths), T+1))
    for p, path in enumerate(paths):
        path_probs[p,0] = p0[path[0]]
        for t in range(1,T+1):
            path_probs[p,t] = path_probs[p,t-1] * P[path[t-1], path[t]]

    # failure paths
    fails = (paths == (S-1)).any(axis=1)

    # last episode timestep, either fail or termination
    end_times = np.where(fails, (paths == (S-1)).argmax(axis=1), T+1)
    # print(np.concatenate((paths, end_times[:,None]), axis=1))

    # failure rate
    failrate = path_probs[fails,-1].sum()

    # enumerate all possible binary alarm masks
    best_flarm_rate, best_ufail_rate, best_mask = np.inf, None, None
    for mask in map(np.array, it.product((True, False), repeat=len(P))):

        # rule out any masks that raise the same and more alarms than best so far
        if best_mask is not None and (mask >= best_mask).all(): continue

        # paths with alarms up to lead time before termination
        alarms = np.array([mask[p[:int(t)]].any() for (p, t) in zip(paths, end_times)])

        # unpreempted fail rate: Pr(no alarm to T-L & fail up to T)
        ufail_rate = path_probs[~alarms & fails, -1].sum()

        # don't consider further if unpreempted failure rate constraint violated
        if ufail_rate > delta: continue

        # false alarm rate: Pr(alarm up to T-L & non-fail up to T)
        flarm_rate = path_probs[alarms & ~fails, -1].sum()

        # track any new bests
        if flarm_rate < best_flarm_rate:
            best_flarm_rate = flarm_rate
            best_ufail_rate = ufail_rate
            best_mask = mask

    return best_flarm_rate, best_ufail_rate, best_mask


if __name__ == "__main__":

    S = 3
    L = 1
    T = 1

    p0, P, f = sample_mp(S,L)
    print(p0)
    print(P)
    print(f)

    f0 = p0[f].sum()
    print(f"initial state failrate = {f0}")

    # delta = 0.5*f0 + .5
    # print(f"setting {delta=} (halfway between {f0} and 1)")
    delta = 0.9*f0 + .1
    print(f"setting {delta=} (slightly bigger than {f0})")

    # form lifted probs
    print('lifted:')
    _p_ = np.concatenate([p0[~f], np.zeros(2*(~f).sum()), p0[f], np.zeros(S)])
    P_ss, P_sf = P[~f][:,~f], P[~f][:,f]
    P_fs, P_ff = P[f][:,~f], P[f][:,f]
    Z_ss, Z_sf = np.zeros_like(P_ss), np.zeros_like(P_sf)
    Z_fs, Z_ff = np.zeros_like(P_fs), np.zeros_like(P_ff)
    _P_q = np.block([
        [P_ss, Z_ss, Z_ss, P_sf, Z_ss, Z_sf],
        [Z_ss, P_ss, Z_ss, Z_sf, Z_ss, P_sf],
        [Z_ss, Z_ss, P_ss, P_sf, Z_ss, Z_sf],
        [Z_fs, Z_fs, P_fs, P_ff, Z_fs, Z_ff],
        [Z_ss, Z_ss, Z_ss, Z_sf, P_ss, P_sf],
        [Z_fs, Z_fs, Z_fs, Z_ff, P_fs, P_ff],
    ])
    _P_w = np.block([
        [Z_ss, P_ss, Z_ss, Z_sf, Z_ss, P_sf],
        [Z_ss, P_ss, Z_ss, Z_sf, Z_ss, P_sf],
        [Z_ss, Z_ss, P_ss, P_sf, Z_ss, Z_sf],
        [Z_fs, Z_fs, P_fs, P_ff, Z_fs, Z_ff],
        [Z_ss, Z_ss, Z_ss, Z_sf, P_ss, P_sf],
        [Z_fs, Z_fs, Z_fs, Z_ff, P_fs, P_ff],
    ])
    print(_p_)
    print(_P_q)
    print(_P_w)

    # knapsack for 1-step problem
    occupancy_measures = [
        cp.Variable(shape=(len(_p_),2), nonneg=True)
        for _ in range(T+1)]
    objective = cp.Minimize(occupancy_measures[T][(~f).sum():2*(~f).sum(),:].sum()) # false alarm
    constraints = [
        occupancy_measures[0].sum(axis=1) == _p_,
        # occupancy_measures[1].sum(axis=1) == (occupancy_measures[0][:,0] @ _P_q) + (occupancy_measures[0][:,1] @ _P_w),
        occupancy_measures[T][2*(~f).sum():2*(~f).sum()+S,:].sum() <= delta,
    ]
    for t in range(T):
        constraints.append(
            occupancy_measures[t+1].sum(axis=1) == (occupancy_measures[t][:,0] @ _P_q) + (occupancy_measures[t][:,1] @ _P_w),
        )
    prob = cp.Problem(objective, constraints)
    prob.solve()
    policy = {}
    for t in range(T+1):
        print(f'occ[{t}] (sum)')
        omt = occupancy_measures[t].value
        nn = (~f).sum()
        nf = f.sum()
        print(omt.round(3).T)
        print(omt.sum())
        print(f'effective alarm policy:')
        policy[t] = np.zeros((S,2))
        policy[t][~f] += omt[0*nn:1*nn]
        policy[t][~f] += omt[1*nn:2*nn]
        policy[t][~f] += omt[2*nn:3*nn]
        policy[t][~f] += omt[3*nn+nf:4*nn+nf]
        policy[t][f] += omt[3*nn:3*nn+nf]
        policy[t][f] += omt[4*nn+nf:]
        policy[t] /= policy[t].sum(axis=1,keepdims=True)
        print(policy[t].round(3).T)
    print(f'FA rate')
    print(occupancy_measures[T].value[(~f).sum():2*(~f).sum(),:].sum())
    print(f'UF rate, <=? {delta=}')
    print(occupancy_measures[T].value[2*(~f).sum():2*(~f).sum()+S,:].sum())

    best_flarm_rate, best_ufail_rate, best_mask = brute(p0, P, f, T)
    print(f"brute {best_flarm_rate=}, {best_ufail_rate=}, mask:")
    print(best_mask)

    ## simulation check
    num_reps = 1000
    ufs = fas = 0
    for rep in range(num_reps):
        i = np.random.choice(S, p=p0)
        if f[i]:
            ufs += 1
            continue
        alarm = fail = False
        for t in range(T):
            if (np.random.rand() > policy[t][i,0]): alarm = True
            i = np.random.choice(S,p=P[i])
            if f[i]:
                fail = True
                if not alarm: ufs += 1
                break
        if alarm and not fail: fas += 1

    print(f"empirical uf rate = {ufs/num_reps}, fa rate = {fas/num_reps}")

