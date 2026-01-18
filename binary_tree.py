"""
Small synthetic domain experiments
"""
import numpy as np
import matplotlib.pyplot as pt

class Node:
    def __init__(self, fail=False, p=None, children=None):
        if children is None: children = []
        self.fail = fail
        self.p = p
        self.children = children
        self.mu = 0 if len(children) == 0 else np.random.rand()

    def __str__(self, prefix=""):
        s = f"{prefix}fail={self.fail}, mu={self.mu:.5f}, p={self.p}"
        for child in self.children:
            s += "\n" + child.__str__(prefix+" ")
        return s

    def get_mu_array(self):
        if len(self.children) == 0: return np.array([[self.mu]])
        m0 = self.children[0].get_mu_array()
        m1 = self.children[1].get_mu_array()
        m01 = np.concatenate((m0, m1), axis=0)
        return np.concatenate((self.mu * np.ones((len(m01),1)), m01), axis=1)

    def get_visit_probs(self):
        if len(self.children) == 0: return np.array([[1.]])
        p0 = self.children[0].get_visit_probs()
        p1 = self.children[1].get_visit_probs()
        p01 = np.concatenate((self.p * p0, (1-self.p)*p1), axis=0)
        return np.concatenate((np.ones((len(p01),1)), p01), axis=1)

    def get_fails(self):
        if len(self.children) == 0: return [self.fail]
        return self.children[0].get_fails() + self.children[1].get_fails()

    def fail_rate(self, leadtime=-1):
        # -1 means failing any point in the future
        if len(self.children) == 0 or leadtime == 0: return float(self.fail)
        return self.p * self.children[0].fail_rate(leadtime-1) + \
               (1-self.p) * self.children[1].fail_rate(leadtime-1)

    def disengage_rate(self, tau):
        if len(self.children) == 0: return 0
        if self.mu > tau: return 1
        return self.p * self.children[0].disengage_rate(tau) + \
               (1-self.p) * self.children[1].disengage_rate(tau)

    def set_mu_to_failrate(self, leadtime=-1):
        if len(self.children) > 0:
            self.mu = self.fail_rate(leadtime)
            for child in self.children: child.set_mu_to_failrate(leadtime)

    def set_mu_to_joint(self, leadtime=-1, p=1):
        if len(self.children) > 0:
            self.mu = p * self.fail_rate(leadtime)
            self.children[0].set_mu_to_joint(leadtime, p * self.p)
            self.children[1].set_mu_to_joint(leadtime, p * (1-self.p))

    def set_mu_to_joint_indicator(self, delta, leadtime=-1, p=1):
        if len(self.children) > 0:
            self.mu = p * (self.fail_rate(leadtime) > delta)
            self.children[0].set_mu_to_joint_indicator(delta, leadtime, p * self.p)
            self.children[1].set_mu_to_joint_indicator(delta, leadtime, p * (1-self.p))

    def set_mu_to_failrate_indicator(self, delta, leadtime=-1):
        if len(self.children) > 0:
            self.mu = float(self.fail_rate(leadtime) > delta)
            for child in self.children: child.set_mu_to_failrate_indicator(delta, leadtime)

    def set_mu_random(self):
        self.mu = 0 if len(self.children) == 0 else np.random.rand()
        for child in self.children: child.set_mu_random()

def build_tree(depth, failrate):
    if depth == 0:
        return Node(fail = (np.random.rand() < failrate))
    else:
        return Node(p = np.random.rand(), children = [
            build_tree(depth-1, failrate),
            build_tree(depth-1, failrate)])

def conform(tree, delta, leadtime, verbose=True):

    mu = tree.get_mu_array()
    probs = tree.get_visit_probs()
    fails = tree.get_fails()

    z = np.ones(len(fails))
    z[fails] = mu[fails,:-leadtime].max(axis=1)
    path_probs = probs[:,-1]

    z_uni = np.concatenate([np.array([0]), np.unique(z)])
    probs = np.array([path_probs[z==x].sum() for x in z_uni])
    # z_cdf = np.concatenate((np.array([0]), probs.cumsum()))
    z_cdf = probs.cumsum()

    # have to have <= delta chance of being less than tau
    tau = z_uni[max(np.flatnonzero(z_cdf <= delta))]
    # dis = tree.disengage_rate(tau)

    # get the rate of disengagement GIVEN no failure
    safe = ~np.array(fails)
    safe_probs = path_probs[safe]
    safe_probs /= safe_probs.sum()
    dis = safe_probs[mu[safe,:-leadtime].max(axis=1) >= tau].sum()

    if verbose:
        print("\nz dist:")
        print(np.stack([z, path_probs]).T)
        print("\nz cdf:")
        print(np.stack([z_uni,z_cdf]).T)
        print(f"\ntau_{delta=}: {tau}")

    return dis, tau

def random_search(tree, delta, leadtime, guesses):
    min_dis, min_tau, min_mu = np.inf, None, None
    for _ in range(guesses):
        tree.set_mu_random()
        guess_dis, guess_tau = conform(tree, delta, leadtime, verbose=False)
        if guess_dis < min_dis:
            min_dis = guess_dis
            min_tau = guess_tau
            min_mu = tree.get_mu_array()
    return min_dis, min_tau, min_mu

def main():

    num_reps = 100
    failrate_sampling = .35
    failrate_target = .3
    delta = .05
    depth = 4
    leadtime = 2
    guesses = 500

    failrates = np.empty(num_reps)
    random_dis = np.empty(num_reps)
    conditional_dis = np.empty(num_reps)
    joint_dis = np.empty(num_reps)
    ji_dis = np.empty(num_reps)
    guess_dis = np.empty(num_reps)
    for rep in range(num_reps):
        print(f"{rep=} of {num_reps}")

        while True:
            tree = build_tree(depth, failrate_sampling)
            failrates[rep] = tree.fail_rate()
            if failrates[rep] > failrate_target: break

        tree.set_mu_to_failrate(leadtime)
        # print(tree)
        # print('mu')
        # print(tree.get_mu_array())
        # print('visit probs')
        # print(tree.get_visit_probs())
        # print('fails')
        # print(tree.get_fails())
        conditional_dis[rep], conditional_tau = conform(tree, delta, leadtime, verbose=False)

        # input('.')

        tree.set_mu_random()
        random_dis[rep], random_tau = conform(tree, delta, leadtime, verbose=False)

        guess_dis[rep], guess_tau, guess_mu = random_search(tree, delta, leadtime, guesses)

        # tree.set_mu_to_failrate_indicator(delta, leadtime)
        tree.set_mu_to_joint(leadtime)
        joint_mu = tree.get_mu_array()
        joint_dis[rep], joint_tau = conform(tree, delta, leadtime, verbose=False)

        tree.set_mu_to_joint_indicator(delta, leadtime)
        ji_dis[rep], ji_tau = conform(tree, delta, leadtime, verbose=False)
    
        # print(f"fail rate = {tree.fail_rate()}")
        # print(f"{random_dis=}, {random_tau=}")
        # print(f"{conditional_dis=}, {failrate_tau=}")

        # ## !! Keep this, useful for analysis in paper
        # if guess_dis[rep] < joint_dis[rep]:
        #     print(tree)
        #     path_probs = tree.get_visit_probs()[:,-1]
        #     fails = np.array(tree.get_fails())
        #     guess_unpre = path_probs[fails & (guess_mu[:,:-leadtime].max(axis=1) < guess_tau)].sum()
        #     joint_unpre = path_probs[fails & (joint_mu[:,:-leadtime].max(axis=1) < joint_tau)].sum()
        #     print(f"guess mu, unpre = {guess_unpre:.4f}, dis={guess_dis[rep]}, tau={guess_tau}:")
        #     print(np.concatenate((guess_mu, fails[:,None]), axis=1).round(4))
        #     print(np.concatenate((guess_mu >= guess_tau, fails[:,None]), axis=1).astype(int))
        #     print(f"joint mu, unpre = {joint_unpre:.4f}, dis={joint_dis[rep]}, tau={joint_tau}:")
        #     print(np.concatenate((joint_mu, fails[:,None]), axis=1).round(4))
        #     print(np.concatenate((joint_mu >= joint_tau, fails[:,None]), axis=1).astype(int))
        #     print("path probs:")
        #     print(path_probs.round(4))
        #     # print("tree fails:")
        #     # print(np.array(tree.get_fails()))
        #     input('...')

    labels = ["guess", "random", "conditional", "joint", "joint indicator"]
    all_dis = np.stack([guess_dis, random_dis, conditional_dis, joint_dis, ji_dis])
    best_dis = all_dis.min(axis=0)

    print("win rates:")
    for (lab, dis) in zip(labels, all_dis):
        print(lab, (dis  == best_dis).mean())

    idx = np.argsort(best_dis)
    # idx = np.argsort(joint_dis)
    # idx = np.argsort(conditional_dis)
    # idx = np.argsort(failrates)
    pt.subplot(1,2,1)
    pt.plot(guess_dis[idx], 'o', mec='k',mfc='w', label="guess")
    pt.plot(random_dis[idx], 'r.', label="random")
    pt.plot(conditional_dis[idx], 'b.', label="conditional")
    pt.plot(joint_dis[idx], 'm.', label="joint")
    pt.plot(ji_dis[idx], 'c.', label="joint indicator")
    pt.plot(failrates[idx], 'k.', label="failrate")
    pt.xlabel("Sorted index")
    pt.ylabel("Disengage rate")
    pt.legend()
    pt.subplot(1,2,2)
    pt.plot(all_dis, 'k.')
    pt.violinplot(all_dis.T, positions=range(len(all_dis)))
    pt.xticks(range(len(all_dis)), labels, rotation=45)
    pt.show()
        

if __name__ == "__main__": main()
