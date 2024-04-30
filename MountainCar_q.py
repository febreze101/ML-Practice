import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=True, render=False):
    
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)
    env.metadata['render_fps']=60

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    if (is_training):
         q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:
        f = open('mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    learning_rate_a_decay = 0.0001
    discount_factor_g = 0.9

    epsilon = 1
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]          # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)


        terminated = False              # true when reached goal

        rewards = 0

        while (not terminated and rewards > -1000):
            # choose random action (0=drive left, 1=stay still, 2=drive right)
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])
            
            new_state, reward, terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        learning_rate_a = max(learning_rate_a - learning_rate_a_decay, .3)

        rewards_per_episode[i] = rewards

    env.close()
    
    if is_training:
        f = open("mountain_car.pkl", "wb")
        pickle.dump(q, f)
        f.close

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'Graphs/moutain_car.png')

if __name__ == '__main__':
    run(20000, is_training=True, render=False)