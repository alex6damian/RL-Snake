import sys
import os
# Adăugăm directorul părinte în path pentru a putea importa game.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import matplotlib
matplotlib.use('Agg')  # Backend non-interactive pentru salvarea graficelor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time as time_module
from datetime import datetime

# Importăm noul agent Dueling
from agent import DuelingDQNAgent
from game import SnakeGame

def plot_final(scores, mean_scores, save_path=None):
    """
    Funcție de plotare adaptată pentru Dueling DQN
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Scores and mean
    plt.subplot(2, 2, 1)
    plt.title('Score Evolution - Dueling DQN', fontsize=14, fontweight='bold')
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
    plt.hist(scores, bins=30, edgecolor='black', color='orange', alpha=0.7)
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
    
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    mean_score = np.mean(scores) if scores else 0
    median_score = np.median(scores) if scores else 0
    std_score = np.std(scores) if scores else 0
    total_games = len(scores)
    last_10_mean = np.mean(scores[-10:]) if scores else 0
    last_10_max = max(scores[-10:]) if scores else 0
    last_50_mean = np.mean(scores[-50:]) if len(scores) >= 50 else mean_score
    last_50_max = max(scores[-50:]) if len(scores) >= 50 else max_score
    
    stats_text = f"""
    FINAL STATISTICS - DUELING DQN
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
    # Setări antrenament
    NUM_GAMES_TO_TRAIN = 5000
    TRAIN_FREQUENCY = 4
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    
    # AICI INITIALIZĂM AGENTUL DUELING
    agent = DuelingDQNAgent(epsilon_start=100, epsilon_min=10)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Curriculum Learning (aceleași dimensiuni ca la DQN)
    small_game = SnakeGame(w=200, h=200, render=False)
    medium_game = SnakeGame(w=400, h=400, render=False)
    large_game = SnakeGame(w=600, h=600, render=False)
    xlarge_game = SnakeGame(w=800, h=800, render=False)
    xxlarge_game = SnakeGame(w=1000, h=1000, render=False)

    print("\n" + "="*70)
    print(f"  DUELING DQN - OPTIMIZED TRAINING")
    print("="*70)
    print(f"Total games:          {NUM_GAMES_TO_TRAIN}")
    print(f"Architecture:         Dueling Network (Value + Advantage)")
    print(f"Device:               {agent.device}")
    print("="*70 + "\n")

    step_counter = 0
    start_time = time_module.time()
    
    with tqdm(total=NUM_GAMES_TO_TRAIN, desc="Training progress", 
              unit="game", ncols=110) as pbar:
        
        for current_game in range(1, NUM_GAMES_TO_TRAIN + 1):
            # Selector dificultate (Curriculum)
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
                final_move = agent.get_action(state_old)
                
                # Execută pasul (aici se va lovi de obstacole când le implementăm în game.py)
                reward, done, score = game.step(final_move)
                
                state_new = agent.get_state(game)

                agent.record_frame(game)

                # Antrenament short memory
                if step_counter % TRAIN_FREQUENCY == 0:
                    agent.train_short_memory(state_old, final_move, reward, state_new, done)
                
                agent.remember(state_old, final_move, reward, state_new, done)
                step_counter += 1
            
            if score > record:
                record = score
                agent.save_model()
                agent.save_best_game(score, game.w, game.h)
                tqdm.write(f"🏆 NEW RECORD: {record} on {difficulty}")
            
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            
            elapsed = time_module.time() - start_time
            games_per_sec = current_game / elapsed if elapsed > 0 else 0
            
            pbar.set_postfix({
                'score': score,
                'record': record,
                'mean': f'{mean_score:.1f}',
                'grid': difficulty,
                'ε': f'{agent.epsilon:.0f}',
                'fps': f'{games_per_sec:.1f} g/s'
            })
            pbar.update(1)
            
            if current_game % 500 == 0:
                last_100_mean = np.mean(scores[-100:])
                tqdm.write(f"\n📊 Checkpoint at game {current_game}:")
                tqdm.write(f"   Last 100 games mean: {last_100_mean:.2f}")
                tqdm.write(f"   Memory usage: {len(agent.memory)}/100000\n")

    elapsed_total = time_module.time() - start_time
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETED - DUELING DQN!")
    print("="*70)
    print(f"🏆 Max score reached:       {record}")
    print(f"📊 Final mean score:        {mean_scores[-1]:.2f}")
    print(f"⏱️  Total training time:     {elapsed_total:.2f}s")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Nume fișier specific Dueling
    plot_filename = os.path.join(results_dir, f"dueling_dqn_plot_{timestamp}.png")
    
    print("Generating and saving plot...")
    plot_final(scores, mean_scores, save_path=plot_filename)
    
    if len(agent.best_game_frames) > 0:
        # Nume fișier replay specific Dueling
        replay_filename = os.path.join(results_dir, f"dueling_dqn_best_replay_{timestamp}.npy")
        
        np.save(replay_filename, {
            'frames': agent.best_game_frames,
            'score': agent.best_game_score,
            'w': agent.best_game_w,
            'h': agent.best_game_h
        })
        print(f"✅ Best game replay saved: {replay_filename}")

        print("\n" + "="*70)
        response = input("Do you want to watch the best game replay? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            print("\nStarting replay in 2 seconds...")
            time_module.sleep(2)
            
            # Folosim același script de replay, funcționează la fel
            from replay_best_game import replay_best_game
            replay_best_game(replay_filename)
        else:
            print("\nReplay skipped.")
            print(f"Model saved in: ./model/dueling_dqn_model.pth")

if __name__ == '__main__':
    train()