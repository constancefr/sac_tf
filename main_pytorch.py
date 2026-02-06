import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from sac_pytorch import Agent

def plot_learning_curve(scores, filename='plot.png', window=100):
    plt.figure(figsize=(12,6))
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.plot(scores, label='Score per episode')
    
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(scores)), moving_avg, 
                label=f'{window}-episode moving average', linewidth=2, color='red')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')  # Continuous action space
    agent = Agent(env=env, input_dims=env.observation_space.shape,
                  n_actions = env.action_space.shape[0])

    score_history = []
    np.random.seed(0)
    n_episodes = 100
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        step_count = 0
        max_steps = 200
        
        while not done and step_count < max_steps:
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, new_state, int(done))
            if agent.memory.mem_counter >= agent.batch_size:
                agent.learn()
            # agent.learn()

            score += reward
            state = new_state
            step_count += 1

        score_history.append(score)

        # average score
        if len(score_history) >= 10:
            avg_score = np.mean(score_history[-10:])
        else:
            avg_score = np.mean(score_history)
            
        print(f'episode {episode+1}/{n_episodes}, score: {score:.2f}, 10-episode avg: {avg_score:.2f}')
    
    agent.save_models()

    filename = 'pt_pendulum_training.png'
    plot_learning_curve(score_history, filename, window=10)

    # final stats
    print(f"\nTraining completed for {n_episodes} episodes")
    print(f"Final 10-episode average: {np.mean(score_history[-10:]):.2f}")
    print(f"Best score: {np.max(score_history):.2f}")
    
    env.close()