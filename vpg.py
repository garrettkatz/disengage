"""
Vanilla Policy Gradient implementation
"""

def train(env, actor, config):

    for update in range(config.updates):
        for episode in range(config.episodes):
            observation, info = env.reset()
            rewards, log_probs = [], []
            for t in range(config.timesteps):
                action, log_prob = actor(observation)
                observation, reward, _, _, _ = env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)
            
    

