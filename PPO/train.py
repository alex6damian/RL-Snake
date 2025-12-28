import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time as time_module
from datetime import datetime

from agent import PPOAgent, LR
from game import SnakeGame

def plot_final(scores, mean_scores, save_path=None):
    """
    Enhanced final plot with multiple statistics
    """
    plt.figure(figsize=(15, 10))

    # Plot 1: Scores and mean
    plt.subplot(2, 2, 1)
    plt.title('Score Evolution - PPO', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, alpha=0.4, label='Score per Game', color='blue')
    plt.plot(mean_scores, linewidth=2, label='Cumulative Mean', color='red')
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Histogram
    plt.subplot(2, 2, 2)
    plt.title('Score Distribution', fontsize=14, fontweight='bold')
    plt.hist(scores, bins=30, edgecolor='black', color='green', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.3)

    # Plot 3: Moving average
    plt.subplot(2, 2, 3)
    plt.title('Moving Average (Last 10 Games)', fontsize=14, fontweight='bold')
    window = 10
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(scores)), moving_avg, linewidth=2, color='purple')
    plt.xlabel('Number of Games')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3)

    # Plot 4: Statistics text
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Calculate values before f-string
    max_score = max(scores)
    min_score = min(scores)
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    total_games = len(scores)
    last_10_mean = np.mean(scores[-10:])
    last_10_max = max(scores[-10:])
    last_50_mean = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
    last_50_max = max(scores[-50:]) if len(scores) >= 50 else max(scores)

    stats_text = f"""
    FINAL STATISTICS - PPO
    {'='*40}

    Max Score:           {max_score}
    Min Score:           {min_score}
    Mean Score:          {mean_score:.2f}
    Median Score:        {median_score:.2f}
    Std Deviation:       {std_score:.2f}

    Total Games:         {total_games}

    Last 10 games:
    Mean:                {last_10_mean:.2f}
    Max:                 {last_10_max}

    Last 50 games:
    Mean:                {last_50_mean:.2f}
    Max:                 {last_50_max}
    """

    plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.close()


def train():
    """PPO training loop with curriculum learning"""
    NUM_GAMES_TO_TRAIN = 5000
    UPDATE_TIMESTEP = 2000  # Update policy every n timesteps

    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = PPOAgent()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Different game sizes for curriculum learning
    small_game = SnakeGame(w=200, h=200, render=False)
    medium_game = SnakeGame(w=400, h=400, render=False)
    large_game = SnakeGame(w=600, h=600, render=False)
    xlarge_game = SnakeGame(w=800, h=800, render=False)
    xxlarge_game = SnakeGame(w=1000, h=1000, render=False)

    print("\n" + "="*70)
    print(f"  PROXIMAL POLICY OPTIMIZATION (PPO) TRAINING")
    print("="*70)
    print(f"Total games:          {NUM_GAMES_TO_TRAIN}")
    print(f"Update frequency:     Every {UPDATE_TIMESTEP} timesteps")
    print(f"PPO epochs:           {agent.k_epochs}")
    print(f"Clip epsilon:         {agent.eps_clip}")
    print(f"Learning rate:        {LR}")
    print(f"Device:               {agent.device}")
    print("="*70 + "\n")

    timestep = 0
    start_time = time_module.time()
    policy_updates = 0

    with tqdm(total=NUM_GAMES_TO_TRAIN, desc="Training progress",
              unit="game", ncols=110) as pbar:

        for current_game in range(1, NUM_GAMES_TO_TRAIN + 1):
            # Curriculum learning
            if current_game <= 1000:
                game = small_game
                difficulty = "SMALL"
            elif current_game <= 2000:
                game = medium_game
                difficulty = "MEDIUM"
            elif current_game <= 3000:
                game = large_game
                difficulty = "LARGE"
            elif current_game <= 4000:
                game = xlarge_game
                difficulty = "XLARGE"
            else:
                game = xxlarge_game
                difficulty = "XXLARGE"

            done = False
            agent.start_recording()
            agent.record_frame(game)

            while not done:
                state_old = agent.get_state(game)

                # Get action from policy
                final_move, action_logprob, state_value = agent.get_action(state_old, train=True)

                # Perform action
                reward, done, score = game.step(final_move)
                state_new = agent.get_state(game)

                agent.record_frame(game)

                # Store experience
                action_idx = final_move.index(1)
                agent.remember(state_old, action_idx, action_logprob, reward, state_value, done)

                timestep += 1

                # Update policy if enough timesteps collected
                if timestep % UPDATE_TIMESTEP == 0:
                    loss = agent.update_policy()
                    policy_updates += 1

            # Update policy at end of episode if memory not empty
            if len(agent.memory) > 0:
                loss = agent.update_policy()
                policy_updates += 1

            if score > record:
                record = score
                agent.save_model()
                agent.save_best_game(score, game.w, game.h)
                tqdm.write(f"🏆 NEW RECORD: {record} on {difficulty}")

            game.reset()
            agent.n_games += 1

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            # Calculate stats
            elapsed = time_module.time() - start_time
            games_per_sec = current_game / elapsed if elapsed > 0 else 0

            pbar.set_postfix({
                'score': score,
                'record': record,
                'mean': f'{mean_score:.1f}',
                'grid': difficulty,
                'updates': policy_updates,
                'fps': f'{games_per_sec:.1f} g/s'
            })
            pbar.update(1)

            # Print periodic updates
            if current_game % 500 == 0:
                last_100_mean = np.mean(scores[-100:])
                tqdm.write(f"\n📊 Checkpoint at game {current_game}:")
                tqdm.write(f"   Last 100 games mean: {last_100_mean:.2f}")
                tqdm.write(f"   Current record: {record}")
                tqdm.write(f"   Policy updates: {policy_updates}\n")

    elapsed_total = time_module.time() - start_time

    print("\n" + "="*70)
    print("  TRAINING COMPLETED - PPO!")
    print("="*70)
    print(f"🏆 Max score reached:       {record}")
    print(f"📊 Final mean score:        {mean_scores[-1]:.2f}")
    print(f"🔄 Total policy updates:    {policy_updates}")
    print(f"🧠 Games trained:           {agent.n_games}")
    print(f"⏱️  Total training time:     {elapsed_total:.2f}s")
    print(f"⚡ Average speed:           {NUM_GAMES_TO_TRAIN/elapsed_total:.2f} games/second")
    print(f"\n📈 Performance breakdown:")
    print(f"   First 100 games mean:    {np.mean(scores[:100]):.2f}")
    if len(scores) > 200:
        print(f"   Middle 100 games mean:   {np.mean(scores[len(scores)//2:len(scores)//2+100]):.2f}")
    print(f"   Last 100 games mean:     {np.mean(scores[-100:]):.2f}")
    print("="*70 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(results_dir, f"ppo_plot_{timestamp}.png")

    print("Generating and saving plot...")
    plot_final(scores, mean_scores, save_path=plot_filename)

    # Save replay
    if len(agent.best_game_frames) > 0:
        replay_filename = os.path.join(results_dir, f"ppo_best_replay_{timestamp}.npy")

        np.save(replay_filename, {
            'frames': agent.best_game_frames,
            'score': agent.best_game_score,
            'w': agent.best_game_w,
            'h': agent.best_game_h
        })
        print(f"✅ Best game replay saved: {replay_filename}")
        print(f"   Grid size: {agent.best_game_w}x{agent.best_game_h}")
        print(f"   Total frames: {len(agent.best_game_frames)}")
        print(f"   Best score: {agent.best_game_score}")

        print("\n" + "="*70)
        response = input("Do you want to watch the best game replay? (y/n): ").strip().lower()

        if response == 'y' or response == 'yes':
            print("\nStarting replay in 2 seconds...")
            time_module.sleep(2)

            from replay_best_game import replay_best_game
            replay_best_game(replay_filename)
        else:
            print("\nReplay skipped. You can watch it later by running:")
            print(f"   python replay_best_game.py")
            print(f"\nModel saved in: ./model/ppo_model.pth")
            print(f"You can load and continue training or test it later!")

if __name__ == '__main__':
    train()
