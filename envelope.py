"""
Build performance profile of black-box agents
"""
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from fqvi import Policy, FeatureExtractor

if __name__ == "__main__":

    num_episodes = 30
    num_timesteps = 100
    num_attempts = 500
    leading_steps = 15

    # load fqvi agent as example
    with open("fqvi_bb.pkl", "rb") as f: (env, policy) = pk.load(f)
    # with open("fqvi_bb.pkl", "rb") as f: env = pk.load(f)

    # run several episodes from same initial state
    init_observation, info = env.reset(seed=np.random.randint(10000))
    init_state = env.state

    # use same opponent motions in each episode
    opponent_motions = env.step_size * env.rng.uniform(-1, 1, (num_episodes, num_timesteps) + env.state.adversaries.shape)

    rewards = np.empty((num_episodes, num_timesteps))
    actions = np.empty((num_episodes, num_timesteps, env.num_allies, 3))
    for e in range(num_episodes):

        observation = init_observation
        env.state = init_state

        for t in range(num_timesteps):
            actions[e,t] = policy(observation)
            observation, rewards[e,t], _, _, _ = env.step(actions[e,t], opponent_motions[e,t])

    # heuristic method to find better reward envelope: first several time-steps have very low action variance
    # ...
    better_rewards = np.empty((num_attempts, num_episodes, num_timesteps))
    init_actions = []
    for a in range(num_attempts):
        print(f"attempt {a} of {num_attempts}")
        # init_actions = [env.action_space.sample() for _ in range(leading_steps)]
        init_actions.append( [env.action_space.sample()] * leading_steps )
        for e in range(num_episodes):
    
            observation = init_observation
            env.state = init_state
    
            for t in range(num_timesteps):
                if t < leading_steps:
                    action = init_actions[a][t]
                else:
                    action = policy(observation)
                observation, better_rewards[a,e,t], _, _, _ = env.step(action, opponent_motions[e,t])

    # get best better rewards
    best = better_rewards.sum(axis=2).mean(axis=1).argmax()
    better_rewards = better_rewards[best]
    better_actions = init_actions[best]

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

    # make animations for original and counterfactual
    input('[Enter] to animate...')
    observation = init_observation
    env.state = init_state
    states = []
    for t in range(num_timesteps):
        states.append(env.state)
        action = policy(observation)
        observation, _, _, _, _ = env.step(action, opponent_motions[most_better, t])

    observation = init_observation
    env.state = init_state
    cfxs = []
    for t in range(num_timesteps):
        cfxs.append(env.state)
        if t < leading_steps:
            action = better_actions[t]
        else:
            action = policy(observation)
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

    # pt.close()

    # fig, axs = pt.subplots(1,1)
    # def drawframe(n):
    #     cfxs[n].render(axs)

    # print("animating cfx...")
    # anim = animation.FuncAnimation(fig, drawframe, frames=len(cfxs), interval=50, blit=False)
    # print("saving...")
    # anim.save("envelope_cfx.mp4")
    
