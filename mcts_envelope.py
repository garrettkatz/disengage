import pickle as pk
import numpy as np

class Node:
    def __init__(self, state, policy, action=None, branches=10, depth=0):
        self.state = state
        self.visit_count = 0 # N
        self.score_total = 0 # numerator of Q
        self.score_estimate = 0 # Q
        self.score_ema = 0. # exponential moving average
        self.child_list = None

        self.policy = policy
        self.action = action
        self.branches = branches
        self.depth = depth

    def children(self):
        # Only generate children the first time they are requested and memoize
        if self.child_list == None:
            self.child_list = []
            for b in range(self.branches):
                if b == 0:
                    action = self.policy(self.state.env.get_observation(self.state))
                else:
                    action = self.state.env.action_space.sample()
                next_state = self.state.env.transition(self.state, action)
                next_node = Node(self.state, self.policy, action, self.branches, self.depth+1)
                self.child_list.append(next_node)

        # Return the memoized child list thereafter
        return self.child_list

    # Helper to collect child visit counts into a list
    def N_values(self):
        return [c.visit_count for c in self.children()]

    # Helper to collect child estimated utilities into a list
    # Utilities are from the current player's perspective
    def Q_values(self):
        children = self.children()

        # empirical average child utilities
        # special case to handle 0 denominator for never-visited children
        Q = [0 if c.visit_count == 0 else c.score_total / c.visit_count for c in children]

        return Q

# optimize strategy: choose the best child for the current player
def optimize(node):
    return node.children()[np.argmax(node.Q_values())]

# explore strategy: choose the least-visited child
def explore(node):
    return node.children()[0] # TODO

# upper-confidence bound strategy
def uct(node):
    const = 3 # TODO: vary
    # max_c Qc + const*sqrt(ln(Np) / Nc)
    Q = np.array(node.Q_values())
    N = np.array(node.N_values())
    # special cases for 0 visits
    U = Q + const * np.sqrt(np.log(max(node.visit_count, 1)) / np.maximum(N, 1))
    return node.children()[np.argmax(U)]

# choose_child = optimize
# choose_child = explore
choose_child = uct

def rollout(node, max_depth):
    if max_depth == 0:
        result = 0
    else:
        result = rollout(choose_child(node), max_depth-1)
    result += node.state.env.reward_function(node.state)
        
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    node.score_ema = 0.99*node.score_ema + 0.01*result
    return result

if __name__ == "__main__":

    do_search = True
    num_inits = 1
    num_timesteps = 100
    num_episodes = 30
    num_rollouts = 200

    with open("fqvi_bb.pkl", "rb") as f: (env, policy) = pk.load(f)

    for init in range(num_inits if do_search else 0):

        # random initial state
        init_observation, info = env.reset(seed=np.random.randint(10000))
        init_state = env.state

        # expand for batching over episodes
        init_state.allies = np.broadcast_to(init_state.allies, (num_episodes, env.num_allies, 3)) 
        init_state.adversaries = np.broadcast_to(init_state.adversaries, (num_episodes, env.num_adversaries, 3))
        init_observation = np.broadcast_to(init_observation, (num_episodes,) + init_observation.shape)
    
        # # use same opponent motions in each rollout?
        # opponent_motions = env.step_size * env.rng.uniform(-1, 1, (num_episodes, num_timesteps, env.num_adversaries, 3))
    
        # collect baseline batched rewards with given policy
        rewards = np.empty((num_episodes, num_timesteps))
        actions = np.empty((num_episodes, num_timesteps, env.num_allies, 3))
        env.state = init_state
        observation = init_observation
        for t in range(num_timesteps):
            actions[:,t] = policy(observation)
            observation, rewards[:,t], _, _, _ = env.step(actions[:,t])

        # collapse init in mcts root node
        init_state.allies = init_state.allies[0]
        init_state.adversaries = init_state.adversaries[0]
        root = Node(init_state, policy, branches=10)

        # do some rollouts
        for r in range(num_rollouts):
            print(f"{r=} of {num_rollouts}")
            rollout(root, max_depth=num_timesteps)

        print(f"{num_inits=}, avg baseline rewards = {rewards.sum(axis=1).mean(axis=0):.5f}")
        print(f"avg rollout rewards = {root.score_estimate:.5f}")

