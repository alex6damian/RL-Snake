import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time as time_module
from datetime import datetime

from agent import Agent
from game import SnakeGame

def plot_final(scores, mean_scores, save_path=None):
    """
    enhanced final plot with multiple statistics
    """
    plt.figure(figsize=(15, 10))
    
    # plot 1: scores and mean
    plt.subplot(2, 2, 1)
    plt.title('Score Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, alpha=0.4, label='Score per Game', color='blue')
    plt.plot(mean_scores, linewidth=2, label='Cumulative Mean', color='red')
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plot 2: histogram
    plt.subplot(2, 2, 2)
    plt.title('Score Distribution', fontsize=14, fontweight='bold')
    plt.hist(scores, bins=30, edgecolor='black', color='green', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.3)
    
    # plot 3: moving average
    plt.subplot(2, 2, 3)
    plt.title('Moving Average (Last 10 Games)', fontsize=14, fontweight='bold')
    window = 10
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(scores)), moving_avg, linewidth=2, color='purple')
    plt.xlabel('Number of Games')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3)
    
    # plot 4: statistics text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
    FINAL STATISTICS
    {'='*40}
    
    Max Score:           {max(scores)}
    Min Score:           {min(scores)}
    Mean Score:          {np.mean(scores):.2f}
    Median Score:        {np.median(scores):.2f}
    Std Deviation:       {np.std(scores):.2f}
    
    Total Games:         {len(scores)}
    
    Last 10 games:
    Mean:                {np.mean(scores[-10:]):.2f}
    Max:                 {max(scores[-10:])}
    
    Last 50 games:
    Mean:                {np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores):.2f}
    Max:                 {max(scores[-50:]) if len(scores) >= 50 else max(scores)}
    """
    
    plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    # save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved in results folder.")
    
    plt.show()


def train():
    """
    training without ui - only progress bar and final plot
    """
    NUM_GAMES_TO_TRAIN = 3000
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    # create results folder in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # create both game variants
    easy_game = SnakeGame(w=120, h=120, render=False)
    normal_game = SnakeGame(w=160, h=160, render=False)
    hard_game = SnakeGame(w=200, h=200, render=False)
    ultra_hard_game = SnakeGame(w=240, h=240, render=False)

    print("\n" + "="*60)
    print(f"  Q-LEARNING SNAKE TRAINING")
    print("="*60)
    print(f"total games:          {NUM_GAMES_TO_TRAIN}")
    print(f"curriculum:           first 800 on small grid (200x200)")
    print(f"                      next 500 on normal grid (400x400)")
    print(f"                      next 700 on hard grid (600x600)")
    print(f"                      last 1000 on ultra hard grid (800x800)")
    print(f"epsilon start:        {agent.epsilon_start}")
    print(f"epsilon min:          {agent.epsilon_min}")
    print(f"gamma:                {agent.gamma}")
    print(f"ui mode:              disabled (fast training)")
    print("="*60 + "\n")

    with tqdm(total=NUM_GAMES_TO_TRAIN, desc="training progress", 
              unit="game", ncols=100) as pbar:
        
        for current_game in range(1, NUM_GAMES_TO_TRAIN + 1):
            if current_game <= 800:
                game = easy_game
                difficulty = "EASY"
            elif current_game <= 1300:
                game = normal_game
                difficulty = "NORMAL"
            elif current_game <= 2000:
                game = hard_game
                difficulty = "HARD"
            else:
                game = ultra_hard_game
                difficulty = "ULTRA HARD"
            
            done = False
            agent.start_recording()
            
            while not done:
                agent.record_frame(game)
                
                state_old = agent.get_state(game)
                final_move = agent.get_action(state_old)
                reward, done, score = game.step(final_move)
                state_new = agent.get_state(game)

                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)
            
            agent.record_frame(game)
            
            if score > record:
                record = score
                agent.save_model()
                agent.save_best_game(score, game.w, game.h)  # PASEAZĂ DIMENSIUNILE REALE
            
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            
            pbar.set_postfix({
                'score': score,
                'record': record,
                'mean': f'{mean_score:.1f}',
                'diff': difficulty,
                'states': len(agent.q_table_1)
            })
            pbar.update(1)

    print("\n" + "="*60)
    print("  TRAINING COMPLETED!")
    print("="*60)
    print(f"max score reached:    {record}")
    print(f"final mean score:     {mean_scores[-1]:.2f}")
    print(f"states learned:       {len(agent.q_table_1)}")
    print(f"experiences stored:   {len(agent.memory)}")
    print("="*60 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(results_dir, f"training_plot_{timestamp}.png")
    
    print("generating and saving plot...")
    plot_final(scores, mean_scores, save_path=plot_filename)
    
    # save replay with CORRECT dimensions
    if len(agent.best_game_frames) > 0:
        replay_filename = os.path.join(results_dir, f"best_game_replay_{timestamp}.npy")
        
        # FOLOSEȘTE DIMENSIUNILE SALVATE ÎN AGENT
        np.save(replay_filename, {
            'frames': agent.best_game_frames,
            'score': agent.best_game_score,
            'w': agent.best_game_w,  # DIMENSIUNILE CORECTE
            'h': agent.best_game_h   # DIMENSIUNILE CORECTE
        })
        print(f"Best game replay saved: {replay_filename}")
        print(f"Grid size: {agent.best_game_w}x{agent.best_game_h}")
        print(f"Total frames recorded: {len(agent.best_game_frames)}")

        if replay_filename and os.path.exists(replay_filename):
            print("\n" + "="*60)
            response = input("Do you want to watch the best game replay? (y/n): ").strip().lower()
            
            if response == 'y' or response == 'yes':
                print("\nStarting replay in 2 seconds...")
                time_module.sleep(2)
                
                from replay_best_game import replay_best_game
                replay_best_game(replay_filename)
            else:
                print("\nReplay skipped. You can watch it later by running:")
                print(f"python replay_best_game.py")

if __name__ == '__main__':
    train()
    