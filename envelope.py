"""
Build performance profile of black-box agents
"""
from line_profiler import profile # python -m kernprof -lvr envelope.py
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

@profile
def main():

    resume = True
    do_search = True
    num_inits = 50
    num_episodes = 30
    num_timesteps = 200
    num_attempts = 30
    leading_steps = 15

    # load fqvi agent as example
    with open("fqvi_bb.pkl", "rb") as f: (env, policy) = pk.load(f)
    # with open("fqvi_bb.pkl", "rb") as f: env = pk.load(f)

    if resume:
        with open("envelope_best.pkl","rb") as f:
            (rewards, better_rewards, _, _, _, _) = pk.load(f)
        # save largest counterfactual deviation found so far
        biggest_difference = (better_rewards - rewards).sum(axis=1).mean(axis=0)

    else:
        biggest_difference = None

    for init in range(num_inits if do_search else 0):

        # run several episodes from same initial state
        init_observation, info = env.reset(seed=np.random.randint(10000))
        init_state = env.state
    
        # expand for batching over episodes
        init_state = init_state.expand_to(num_episodes)
        init_observation = np.broadcast_to(init_observation, (num_episodes,) + init_observation.shape)
    
        # use same opponent motions in each episode
        opponent_motions = env.step_size * env.rng.uniform(-1, 1, (num_episodes, num_timesteps, env.num_adversaries, 3))
    
        # collect batched rewards with given policy
        rewards = np.empty((num_episodes, num_timesteps))
        actions = np.empty((num_episodes, num_timesteps, env.num_allies, 3))
        env.state = init_state
        observation = init_observation
        for t in range(num_timesteps):
            actions[:,t] = policy(observation)
            observation, rewards[:,t], _, _, _ = env.step(actions[:,t], opponent_motions[:,t])
    
        # heuristic method to find better reward envelope: first several time-steps have very low action variance
        better_rewards = np.empty((num_attempts, num_episodes, num_timesteps))
        init_actions = []
        for a in range(num_attempts):
    
            print(f"init {init} of {num_inits}, attempt {a} of {num_attempts}, {biggest_difference=}")
            # init_actions = [env.action_space.sample() for _ in range(leading_steps)]
            init_actions.append( [env.action_space.sample()] * leading_steps )
    
            env.state = init_state
            observation = init_observation
            for t in range(num_timesteps):
                if t < leading_steps:
                    actions[:,t] = np.broadcast_to(init_actions[a][t], (num_episodes, env.num_allies, 3))
                else:
                    actions[:,t] = policy(observation)
                observation, better_rewards[a,:,t], _, _, _ = env.step(actions[:,t], opponent_motions[:,t])

        # get best better rewards
        best = better_rewards.sum(axis=2).mean(axis=1).argmax()
        better_rewards = better_rewards[best]
        better_actions = init_actions[best]

        # save largest counterfactual deviation found so far
        difference = (better_rewards - rewards).sum(axis=1).mean(axis=0)
        if biggest_difference is None or difference > biggest_difference:
            biggest_difference = difference
            print(f"new {biggest_difference=}")
            with open("envelope_best.pkl","wb") as f:
                pk.dump((rewards, better_rewards, better_actions, init_observation, init_state, opponent_motions), f)

    with open("envelope_best.pkl","rb") as f:
        (rewards, better_rewards, better_actions, init_observation, init_state, opponent_motions) = pk.load(f)

    # render envelope
    # pt.subplot(1,2,1)
    pt.plot(np.arange(num_timesteps), np.arange(1, num_timesteps+1)*env.get_max_reward(), 'g--', label="theoretically optimal")

    cumulative_rewards = rewards.cumsum(axis=1)
    pt.fill_between(np.arange(num_timesteps), cumulative_rewards.min(axis=0), cumulative_rewards.max(axis=0), alpha=.1, color='r')
    pt.plot(cumulative_rewards.mean(axis=0), 'r-', label="black-box")

    better_cumulative_rewards = better_rewards.cumsum(axis=1)
    pt.fill_between(np.arange(num_timesteps), better_cumulative_rewards.min(axis=0), better_cumulative_rewards.max(axis=0), alpha=.1, color='b')
    pt.plot(better_cumulative_rewards.mean(axis=0), 'b-', label="counterfactual")
    pt.xlabel("Time-step")
    pt.ylabel("Cumulative Reward")
    pt.title(f"Performance mean and envelope ({num_episodes} episodes)")
    pt.legend()

    # pt.subplot(1,2,2)
    # actions = actions.reshape((num_episodes, num_timesteps, env.num_allies*3))
    # action_means = actions.mean(axis=0)
    # action_stds = np.linalg.norm(actions - action_means[None], axis=2).mean(axis=0)
    # pt.plot(action_stds)

    pt.savefig("envelope.png")
    pt.show()

    # get episode where counterfactual is most better than original
    most_better = (better_rewards.sum(axis=1) - rewards.sum(axis=1)).argmax()

    # unbatch
    init_observation = init_observation[most_better]
    init_state.allies = init_state.allies[most_better]
    init_state.adversaries = init_state.adversaries[most_better]

    # make animations for original and counterfactual
    input('[Enter] to animate...')
    observation = init_observation
    env.state = init_state
    states = []
    for t in range(num_timesteps):
        states.append(env.state)
        action = policy(observation[None])[0]
        observation, _, _, _, _ = env.step(action, opponent_motions[most_better, t])

    observation = init_observation
    env.state = init_state
    cfxs = []
    for t in range(num_timesteps):
        cfxs.append(env.state)
        if t < leading_steps:
            action = better_actions[t]
        else:
            action = policy(observation[None])[0]
        observation, _, _, _, _ = env.step(action, opponent_motions[most_better, t])

    pt.close()

    import matplotlib.animation as animation

    fig, axs = pt.subplots(1,2)
    def drawframe(n):
        states[n].render(axs[0])
        cfxs[n].render(axs[1])
        axs[0].set_title("black-box")
        axs[1].set_title("counterfactual")

    print("animating...")
    anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=50, blit=False)
    print("saving...")
    anim.save("envelope.mp4")
    
if __name__ == "__main__": main()
