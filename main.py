from neuralNet import *
import random
import AlexNN
import cv2
import matplotlib.pyplot as plt

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000


def to_gray_scale(screen):
    return cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)


def is_black(screen):
    return np.max(screen) == 0


def crop_height(screen, top=150, bottom=350):
    return screen[top:bottom, :]


def preprocess(screen):
    return crop_height(to_gray_scale(screen))


def main():

    nn = AlexNN.DQN((2, 10, 2), activations=['relu', 'relu', ''], cost_func='mse')
    data = [(np.random.uniform(0, 1, (2, 1)), np.random.uniform(0, 1, (2, 1))) for x in range(10000)]
    print(f'v: {nn.forward(data[0][0])}\nt: {data[0][1]}')
    nn.SGD(10, 1000, data, lr*10)
    print(f'v: {nn.forward(data[0][0])}\nt: {data[0][1]}')
    return
    env = gym.make('CartPole-v0')
    env.reset()
    screen = env.render(mode='rgb_array')
    screen = preprocess(screen)
    screen_dim = screen.shape[:2]

    target_nn = AlexNN.DQN((screen_dim[0] * screen_dim[1], 32, 24, env.action_space.n), ('relu', 'relu', 'relu', ''), 'mse')
    policy_nn = target_nn.copy()

    episode_durations = []
    memory = ReplayMemory(memory_size)
    for episode in range(num_episodes):
        state = env.reset()
        state = env.render(mode='rgb_array')
        state = preprocess(state)
        for timestep in count():
            # choose action
            action = env.action_space.sample()
            if random.random() > gamma:
                # exploit
                pass
            new_state, reward, done, _ = env.step(action)
            new_state = env.render(mode='rgb_array')
            new_state = preprocess(new_state)

            memory.push(AlexNN.MemoryElement(state, action, reward, new_state))
            if memory.can_provide_sample(batch_size):  # has batch
                print('started training')
                sample = memory.sample(batch_size)
                states, actions, rewards, next_actions = AlexNN.to_data(sample)
                states = np.asarray([state.flatten().reshape(-1, 1)/255 for state in states])

                current_qs = np.asarray([policy_nn.forward(state) for state in states])
                next_qs = np.asarray([target_nn.forward(state) for state in states])

                target_qs = (next_qs * gamma)
                target_qs = [targ + rew for targ, rew in zip(target_qs, rewards)]

                '''print(target_qs)
                print('-' * 100)
                print(rewards)
                print('-' * 100)'''

                # print(target_qs)

                batch = [(state, targ) for state, targ in zip(states, target_qs)]
                # print(*batch)
                '''print(current_qs)
                print('-' * 100)
                print(target_qs)
                print('-' * 100)'''
                policy_nn.update_mini_batch(batch, lr)
                print('stopped training')

            if done:
                episode_durations.append(timestep)
                break

            if episode % target_update == 0:
                print('updated net')
                target_nn = policy_nn.copy()

    env.close()




    return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = CartPoleEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()
        for timestep in count():
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state
            # trains
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                print(policy_net(states).shape)
                em.close()
                return
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if em.done:
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
    em.close()

    '''EXPLORATION_COEFFICENT = 1
    env = gym.make('FrozenLake-v0')

    all_episode_rewards = []

    q_table = np.zeros([16, env.action_space.n])

    for episode in range(NUMBER_OF_EPISODES):
        state = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):

            # take action
            action = 0
            if rnd.uniform(0, 1) < EXPLORATION_COEFFICENT:
                action = env.action_space.sample()
            else:
                # look up in q table
                action = np.argmax(q_table[state, :])

            # env.render()
            new_state, reward, done, info = env.step(action)  # take a random action
            # print(reward)
            episode_reward += reward
            q_table[state, action] = q_table[state, action] * (1 - LEARNING_RATE) + \
                                     LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(q_table[new_state, :]))

            EXPLORATION_COEFFICENT = MIN_EXPLORATION_RATE + \
                               (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * episode)
            state = new_state
            if done:
                break
        all_episode_rewards.append(episode_reward)
    print(q_table)
    all_episode_rewards = np.asarray(all_episode_rewards)
    for x in range(0, NUMBER_OF_EPISODES, 1000):
        print(f"Episode {int(x/1000)}: {np.mean(all_episode_rewards[x:x+1000])}")
    # print(all_episode_rewards)
    env.close()'''


if __name__ == '__main__':
    main()
