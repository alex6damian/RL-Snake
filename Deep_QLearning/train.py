import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '. .')))

import pygame
import matplotlib
matplotlib.use('Agg')  # Backend non-interactive
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time as time_module
from datetime import datetime

from agent import DQNAgent
from game import SnakeGame

def plot_final(scores, mean_scores, save_path=None):
    """
    Enhanced final plot with multiple statistics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Scores and mean
    plt.subplot(2, 2, 1)
    plt.title('Score Evolution - DQN', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, alpha=0.4, label='Score per Game', color='blue')
    plt. plot(mean_scores, linewidth=2, label='Cumulative Mean', color='red')
    plt.ylim(ymin=0)
    plt. legend()
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
    
    # calculate values before f-string
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
    FINAL STATISTICS - DQN
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
        print(f"✅ Plot saved: {save_path}")
    
    plt.close()


def train():
    # optimized training loop
    NUM_GAMES_TO_TRAIN = 5000
    TRAIN_FREQUENCY = 4
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent(epsilon_start=100, epsilon_min=10)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # different game sizes for curriculum learning
    small_game = SnakeGame(w=200, h=200, render=False)
    medium_game = SnakeGame(w=400, h=400, render=False)
    large_game = SnakeGame(w=600, h=600, render=False)
    xlarge_game = SnakeGame(w=800, h=800, render=False)
    xxlarge_game = SnakeGame(w=1000, h=1000, render=False)

    print("\n" + "="*70)
    print(f"  DEEP Q-LEARNING - OPTIMIZED TRAINING")
    print("="*70)
    print(f"Total games:          {NUM_GAMES_TO_TRAIN}")
    print(f"Train frequency:      Every {TRAIN_FREQUENCY} steps (OPTIMIZED)")
    print(f"Device:               {agent. device}")
    print("="*70 + "\n")

    step_counter = 0
    start_time = time_module.time()
    
    with tqdm(total=NUM_GAMES_TO_TRAIN, desc="Training progress", 
              unit="game", ncols=110) as pbar:
        
        for current_game in range(1, NUM_GAMES_TO_TRAIN + 1):
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
            agent.record_frame(game) # record initial frame
            
            while not done:                
                state_old = agent.get_state(game)
                final_move = agent.get_action(state_old)
                reward, done, score = game.step(final_move)
                state_new = agent.get_state(game)

                agent.record_frame(game)

                if step_counter % TRAIN_FREQUENCY == 0:
                    agent.train_short_memory(state_old, final_move, reward, state_new, done)
                
                agent.remember(state_old, final_move, reward, state_new, done)
                step_counter += 1
            
            # agent.record_frame(game) moved inside the loop
            
            if score > record:
                record = score
                agent.save_model()
                agent.save_best_game(score, game. w, game.h)
                tqdm.write(f"🏆 NEW RECORD: {record} on {difficulty}")
            
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores. append(mean_score)
            
            # calculate FPS
            elapsed = time_module.time() - start_time
            games_per_sec = current_game / elapsed if elapsed > 0 else 0
            
            # FIX: remove spaces from format specifiers
            pbar.set_postfix({
                'score': score,
                'record': record,
                'mean': f'{mean_score:.1f}',
                'grid': difficulty,
                'ε': f'{agent.epsilon:.0f}',
                'fps': f'{games_per_sec:.1f} g/s'
            })
            pbar.update(1)
            
            # print periodic updates
            if current_game % 500 == 0:
                last_100_mean = np.mean(scores[-100:])
                tqdm.write(f"\n📊 Checkpoint at game {current_game}:")
                tqdm.write(f"   Last 100 games mean: {last_100_mean:.2f}")
                tqdm.write(f"   Current record: {record}")
                tqdm.write(f"   Memory usage: {len(agent.memory)}/100000\n")

    elapsed_total = time_module.time() - start_time
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETED!")
    print("="*70)
    print(f"🏆 Max score reached:       {record}")
    print(f"📊 Final mean score:        {mean_scores[-1]:.2f}")
    print(f"💾 Experiences stored:      {len(agent.memory)}")
    print(f"🧠 Network trained on:      {agent.n_games} games")
    print(f"⏱️  Total training time:     {elapsed_total:.2f}s")
    print(f"⚡ Average speed:           {NUM_GAMES_TO_TRAIN/elapsed_total:.2f} games/second")
    print(f"\n📈 Performance breakdown:")
    print(f"   First 100 games mean:    {np.mean(scores[:100]):.2f}")
    if len(scores) > 200:
        print(f"   Middle 100 games mean:   {np. mean(scores[len(scores)//2:len(scores)//2+100]):.2f}")
    print(f"   Last 100 games mean:     {np.mean(scores[-100:]):.2f}")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(results_dir, f"dqn_plot_{timestamp}.png")
    
    print("Generating and saving plot...")
    plot_final(scores, mean_scores, save_path=plot_filename)
    
    # save replay
    if len(agent.best_game_frames) > 0:
        replay_filename = os.path. join(results_dir, f"dqn_best_replay_{timestamp}.npy")
        
        np.save(replay_filename, {
            'frames': agent.best_game_frames,
            'score': agent. best_game_score,
            'w': agent.best_game_w,
            'h': agent.best_game_h
        })
        print(f"✅ Best game replay saved: {replay_filename}")
        print(f"   Grid size: {agent.best_game_w}x{agent.best_game_h}")
        print(f"   Total frames: {len(agent.best_game_frames)}")
        print(f"   Best score: {agent.best_game_score}")

        print("\n" + "="*70)
        response = input("Do you want to watch the best game replay? (y/n): ").strip(). lower()
        
        if response == 'y' or response == 'yes':
            print("\nStarting replay in 2 seconds...")
            time_module. sleep(2)
            
            from replay_best_game import replay_best_game
            replay_best_game(replay_filename)
        else:
            print("\nReplay skipped.  You can watch it later by running:")
            print(f"   python replay_best_game.py")
            print(f"\nModel saved in: ./model/dqn_model. pth")
            print(f"You can load and continue training or test it later!")

if __name__ == '__main__':
    train()