"""
finite markov process
"""
import numpy as np
import matplotlib.pyplot as pt
import pickle as pk

np.set_printoptions(linewidth=1000)

def sample_fmp(S, T, L, min_failrate=0):

    while True:

        # make transition probabilities
        # P[i,j] = P(s'=j|s=i)

        # P = (np.random.rand(S,S) < 2/S)
        # P /= P.sum

        P = np.tril(np.random.rand(S,S), k=S-L) # takes at least L steps to get to last state
        P[-1,:-1] = 0 # last state is failure and absorbing
        P /= P.sum(axis=1, keepdims=True)
        # print(P)

        # all possible paths up to T
        paths = [[0]]
        for t in range(1,T):
            paths = [p + [s] for p in paths for s in range(S) if P[p[-1], s] > 0]
        paths = np.array(paths)

        # leading path probabilities: path_probs[p,t] = Pr(paths[p][:t+1])
        path_probs = np.ones((len(paths), T))
        for p, path in enumerate(paths):
            for t in range(1,T):
                path_probs[p,t] = path_probs[p,t-1] * P[path[t-1], path[t]]
    
        # failure paths
        fails = (paths == (S-1)).any(axis=1)

        # last episode timestep, either fail or termination
        end_times = np.where(fails, (paths == (S-1)).argmax(axis=1), T-1)
        # print(np.concatenate((paths, end_times[:,None]), axis=1))

        # failure rate
        failrate = path_probs[fails,-1].sum()

        # rejection sampling
        if failrate >= min_failrate: return P, paths, path_probs, fails, end_times, failrate

def conform(fmp, L, delta, mu, verbose=False):

    P, paths, path_probs, fails, end_times, failrate = fmp

    # path_mus[p,t] = mu(s_t) in path p
    path_mus = mu[paths]
    # print(path_mus)
    # print(paths)
    # print(mu)

    # z: 1 if no failure, else max_{t<=t_F-L} mu_t
    path_zs = np.array([
        pm[:int(t_F)-L+1].max(axis=0) if fail else 1
        for (pm, t_F, fail) in zip(path_mus, end_times, fails)])
    # print(np.concatenate((path_probs, end_times[:,None], path_zs[:,None]), axis=1))

    # select tau such that: Pr(z <= tau) <= delta
    # issue warning when mu > tau
    z_uni = np.concatenate(([-np.inf], np.unique(path_zs)))
    z_cdf = np.array([path_probs[path_zs <= z, -1].sum() for z in z_uni])
    tau = z_uni[max(0, (z_cdf > delta).argmax()-1)]
    if verbose:
        print(np.stack([z_uni, z_cdf]))
        print(f"{tau=}")

    # paths with warnings up to lead time before termination
    alarms = np.array([(pm[:int(t)-L+1].max(axis=0) > tau) for (pm, t) in zip(path_mus, end_times)])

    # unpreempted fail rate: Pr(no alarm to T-L & fail up to T)
    ufail_rate = path_probs[~alarms & fails, -1].sum()

    # false alarm rate: Pr(alarm up to T-L & non-fail up to T)
    flarm_rate = path_probs[alarms & ~fails, -1].sum()

    if ufail_rate > delta:
        print("P",P)
        print("mu", mu)
        print("path mus, path zs, fails, path probs")
        # print(path_mus.shape, path_zs.shape, fails.shape, path_probs.shape)
        print(np.concatenate([path_mus, path_zs[:,None], fails[:,None], path_probs[:,-1:]], axis=1))
        print("z cdf")
        print(np.stack([z_uni, z_cdf]))
        print(f"{tau=}")
        input('...?')

    return ufail_rate, flarm_rate

def constant_mu(fmp, L, delta):
    P, paths, path_probs, fails, end_times, failrate = fmp
    S = len(P)
    mu = np.ones(S)
    ufail_rate, flarm_rate = conform(fmp, L, delta, mu)
    return flarm_rate, ufail_rate, mu

def random_search_mu(fmp, L, delta, num_guesses, verbose=False):
    P, paths, path_probs, fails, end_times, failrate = fmp
    S = len(P)

    min_uf_rate, min_fa_rate, min_mu = None, np.inf, None
    for _ in range(num_guesses):
        if verbose: print(f"guess {_} of {num_guesses}")
        mu = np.random.rand(S)
        ufail_rate, flarm_rate = conform(fmp, L, delta, mu)
        if flarm_rate < min_fa_rate:
            min_fa_rate, min_uf_rate, min_mu = flarm_rate, ufail_rate, mu

    ufail_rate, flarm_rate = conform(fmp, L, delta, min_mu, verbose=True)

    return min_fa_rate, min_uf_rate, min_mu

def near_state_mu(fmp, L, delta):
    P, paths, path_probs, fails, end_times, failrate = fmp
    S = len(P)
    mu = (np.arange(S) >= S-1-L).astype(float)
    ufail_rate, flarm_rate = conform(fmp, L, delta, mu)
    return flarm_rate, ufail_rate, mu

def conditional_mu(fmp, L, delta):
    # mu(s) = Pr(fail within L steps | s)
    P, paths, path_probs, fails, end_times, failrate = fmp
    mu = np.array([
        np.linalg.matrix_power(P, L)[s,-1]
        for s in range(len(P))])

    ufail_rate, flarm_rate = conform(fmp, L, delta, mu)
    return flarm_rate, ufail_rate, mu

def joint_mu(fmp, L, delta):
    # mu(s) = Pr(at s & fail within L steps)
    P, paths, path_probs, fails, end_times, failrate = fmp
    S = len(P)

    mu = np.empty(S)
    for s in range(S):
        within = np.array([s in p[t-L:t] for (p, t) in zip(paths, end_times)])
        mu[s] = path_probs[fails & within, -1].sum()

    ufail_rate, flarm_rate = conform(fmp, L, delta, mu)
    return flarm_rate, ufail_rate, mu

def indicator_mu(fmp, L, delta):
    # mu(s) = 1[Pr(fail within L steps | s) > delta]
    P, paths, path_probs, fails, end_times, failrate = fmp
    mu = np.array([
        np.linalg.matrix_power(P, L)[s,-1] > delta
        for s in range(len(P))]).astype(float)

    ufail_rate, flarm_rate = conform(fmp, L, delta, mu)
    return flarm_rate, ufail_rate, mu

def main():

    S = 6
    T = 6
    L = 2 # lead time
    min_failrate = 0
    delta_ratio = .5
    guesses = 500
    num_reps = 50
    do_reps = True

    if do_reps:

        failrate = np.empty(num_reps)
        fa_rate = {
            "random once": np.empty(num_reps),
            "random many": np.empty(num_reps),
            "conditional": np.empty(num_reps),
            "indicator": np.empty(num_reps),
            # "joint": np.empty(num_reps),
        }
        uf_rate = {
            "random once": np.empty(num_reps),
            "random many": np.empty(num_reps),
            "conditional": np.empty(num_reps),
            "indicator": np.empty(num_reps),
            # "joint": np.empty(num_reps),
        }
        for rep in range(num_reps):
            print(rep)
    
            P, paths, path_probs, fails, end_times, failrate[rep] = fmp = sample_fmp(S, T, L, min_failrate)
            print(P)
            print(f"min end time = {end_times.min()}")

            delta = failrate[rep] * delta_ratio
            print(f"fail rate = {failrate[rep]}")
            print(f"{delta=}")
        
            # print("\nnear state:")
            # flarm_rate, ufail_rate, mu = near_state_mu(fmp, L, delta)
            # print(f"{flarm_rate=}")
            # print(f"{ufail_rate=}")
        
            # print("\nconstant:")
            # flarm_rate, ufail_rate, mu = constant_mu(fmp, L, delta)
            # print(f"{flarm_rate=}")
            # print(f"{ufail_rate=}")

            print("\nrandom once:")
            fa_rate["random once"][rep], uf_rate["random once"][rep], mu = random_search_mu(fmp, L, delta, 1, verbose=False)
            print(f"uw: {fa_rate['random once'][rep]}")
            print(f"uf: {uf_rate['random once'][rep]}")

            print("\nrandom many:")
            fa_rate["random many"][rep], uf_rate["random many"][rep], mu = random_search_mu(fmp, L, delta, guesses, verbose=False)
            print(f"uw: {fa_rate['random many'][rep]}")
            print(f"uf: {uf_rate['random many'][rep]}")
        
            print("\nconditional:")
            fa_rate["conditional"][rep], uf_rate["conditional"][rep], mu = conditional_mu(fmp, L, delta)
            print(f"uw: {fa_rate['conditional'][rep]}")
            print(f"uf: {uf_rate['conditional'][rep]}")

            print("\nindicator:")
            fa_rate["indicator"][rep], uf_rate["indicator"][rep], mu = indicator_mu(fmp, L, delta)
            print(f"uw: {fa_rate['indicator'][rep]}")
            print(f"uf: {uf_rate['indicator'][rep]}")

            # print("\njoint:")
            # fa_rate["joint"][rep], uf_rate["joint"][rep], mu = joint_mu(fmp, L, delta)
            # print(f"uw: {fa_rate['joint'][rep]}")
            # print(f"uf: {uf_rate['joint'][rep]}")

        with open("fmp.pkl","wb") as f:
            pk.dump((failrate, fa_rate, uf_rate), f)

    with open("fmp.pkl","rb") as f:
        (failrate, fa_rate, uf_rate) = pk.load(f)

    # labels = ["conditional", "joint", "random once", "random many"]
    labels = ["conditional", "indicator", "random once", "random many"]

    best_rate = np.stack([fa_rate[label] for label in labels]).min(axis=0)
    print("Win rates")
    for label in labels:
        print(label, (fa_rate[label] == best_rate).mean())

    pt.subplot(2,1,1)
    # idx = np.argsort(best_rate)
    idx = np.argsort(fa_rate["random many"])
    pt.plot(fa_rate["conditional"][idx], 'o', mfc='none', mec='b', label="conditional")
    # pt.plot(fa_rate["joint"][idx], 's', mfc='none', mec='m', label="joint")
    pt.plot(fa_rate["indicator"][idx], 's', mfc='none', mec='m', label="indicator")
    pt.plot(fa_rate["random many"][idx], 'g.', label="random many")
    pt.plot(fa_rate["random once"][idx], 'r+', label="random once")
    # pt.title("Unnecessary warning")
    pt.ylabel("False alarm rate")
    # pt.ylabel("Rate")
    # pt.legend()

    pt.subplot(2,1,2)
    best_rate = np.stack([uf_rate[label] / (failrate * delta_ratio) for label in labels]).min(axis=0)
    # idx = np.argsort(best_rate)
    idx = np.argsort(uf_rate["random many"] / (failrate * delta_ratio))
    pt.plot(uf_rate["conditional"][idx] / (failrate[idx] * delta_ratio), 'o', mfc='none', mec='b', label="conditional")
    # pt.plot(uf_rate["joint"][idx] / (failrate[idx] * delta_ratio), 's', mfc='none', mec='m', label="joint")
    pt.plot(uf_rate["indicator"][idx] / (failrate[idx] * delta_ratio), 's', mfc='none', mec='m', label="indicator")
    pt.plot(uf_rate["random many"][idx] / (failrate[idx] * delta_ratio), 'g.', label="random many")
    pt.plot(uf_rate["random once"][idx] / (failrate[idx] * delta_ratio), 'r+', label="random once")
    pt.ylabel("Unpreemted failure rate / delta")
    # pt.title("Unpreemted failure")
    pt.legend()

    pt.gcf().supxlabel("Experimental repetition (sorted)")
    # pt.subplot(1,3,3)
    # for label, style in zip(labels, ("r.", "g.", "b.")):
    #     pt.plot(fa_rate[label] / failrate, uf_rate[label] / (failrate * delta_ratio), style, label=label)
    # pt.xlabel("unnecessary warning / fail rate")
    # pt.ylabel("unpreempted failure / delta")
    # pt.legend()

    pt.tight_layout()
    pt.show()


if __name__ == "__main__": main()
