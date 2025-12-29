import sys
import os
# Adăugăm calea către folderul părinte pentru a importa game.py corect
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # Backend non-interactiv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time as time_module
from datetime import datetime

# Importăm Agentul tău NOU și Jocul comun
from agent import Agent
from game import SnakeGame

def plot_final(scores, mean_scores, save_path=None):
    """
    Generează graficul final cu statistici detaliate.
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Evoluția Scorului
    plt.subplot(2, 2, 1)
    plt.title('Score Evolution - Double Q-Learning (Optimized)', fontsize=12, fontweight='bold')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.plot(scores, alpha=0.3, color='blue', label='Score')
    plt.plot(mean_scores, linewidth=2, color='red', label='Mean')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Distribuția Scorurilor (Histogramă)
    plt.subplot(2, 2, 2)
    plt.title('Score Distribution', fontsize=12, fontweight='bold')
    plt.hist(scores, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    
    # 3. Media Mobilă (Moving Average - 100 jocuri)
    plt.subplot(2, 2, 3)
    plt.title('Moving Average (Last 100 Games)', fontsize=12, fontweight='bold')
    if len(scores) >= 100:
        moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        plt.plot(moving_avg, color='purple', linewidth=2)
    else:
        plt.plot(scores, color='purple')
    plt.grid(alpha=0.3)
    
    # 4. Statistici Text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    max_score = max(scores) if scores else 0
    mean_val = np.mean(scores) if scores else 0
    std_val = np.std(scores) if scores else 0
    last_100 = np.mean(scores[-100:]) if len(scores) >= 100 else mean_val
    
    stats = f"""
    FINAL STATISTICS
    ----------------
    Total Games:    {len(scores)}
    Max Score:      {max_score}
    Mean Score:     {mean_val:.2f}
    Std Deviation:  {std_val:.2f}
    
    Last 100 Avg:   {last_100:.2f}
    """
    plt.text(0.1, 0.5, stats, fontsize=12, family='monospace')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    plt.close()

def train():
    # --- SETĂRI ANTRENAMENT ---
    NUM_GAMES_TO_TRAIN = 5000
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()
    
    # Setup foldere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # --- CURRICULUM LEARNING (Hărți de la 200px la 1000px) ---
    small_game = SnakeGame(w=200, h=200, render=False)
    medium_game = SnakeGame(w=400, h=400, render=False)
    large_game = SnakeGame(w=600, h=600, render=False)
    xlarge_game = SnakeGame(w=800, h=800, render=False)
    xxlarge_game = SnakeGame(w=1000, h=1000, render=False)

    print("\n" + "="*60)
    print("  DOUBLE Q-LEARNING (OPTIMIZED) - TRAINING START")
    print("="*60)
    print(f" Target Games: {NUM_GAMES_TO_TRAIN}")
    print(f" Curriculum:   Small(200) -> XXL(1000)")
    print("="*60 + "\n")

    start_time = time_module.time()
    
    with tqdm(total=NUM_GAMES_TO_TRAIN, desc="Training", unit="game", ncols=100) as pbar:
        for i in range(1, NUM_GAMES_TO_TRAIN + 1):
            # Selectare Dificultate
            if i <= 1000:
                game = small_game
                difficulty = "SMALL"
            elif i <= 2000:
                game = medium_game
                difficulty = "MEDIUM"
            elif i <= 3000:
                game = large_game
                difficulty = "LARGE"
            elif i <= 4000:
                game = xlarge_game
                difficulty = "XLARGE"
            else:
                game = xxlarge_game
                difficulty = "XXLARGE"
            
            # Resetare Joc
            game.reset()
            agent.n_games += 1
            agent.start_recording() # Metodă specifică agentului tău nou
            agent.record_frame(game) # Înregistrează cadrul inițial
            
            done = False
            while not done:
                # 1. Obține starea (discretizată de agent)
                state_old = agent.get_state(game)
                
                # 2. Alege acțiunea
                final_move = agent.get_action(state_old)
                
                # 3. Execută pasul
                reward, done, score = game.step(final_move)
                
                # 4. Obține noua stare
                state_new = agent.get_state(game)
                
                # 5. Antrenament Pas cu Pas (Short Memory)
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                
                # 6. Salvare în memorie (Prioritized Replay)
                agent.remember(state_old, final_move, reward, state_new, done)
                
                # 7. Înregistrare cadru pentru replay
                agent.record_frame(game)

            # La finalul jocului: Antrenament din Memorie (Long Memory)
            agent.train_long_memory()
            
            # Verificare Record și Salvare
            if score > record:
                record = score
                agent.save_model() # Salvează tabelele Q
                # Salvează replay-ul (agentul tău știe să copieze current_frames în best_frames)
                is_best = agent.save_best_game(score, game.w, game.h)
            
            # Statistici
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            
            # Update bară progres
            q_size = len(agent.q_table_1) # Vedem câte stări unice a învățat
            pbar.set_postfix({
                'score': score,
                'mean': f"{mean_score:.1f}",
                'record': record,
                'grid': difficulty,
                'States': q_size
            })
            pbar.update(1)

    # --- FINALIZARE ---
    elapsed_total = time_module.time() - start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETED!")
    print(f"  Max Score: {record}")
    print(f"  Final Mean: {mean_scores[-1]:.2f}")
    print(f"  Time: {elapsed_total:.1f}s")
    print("="*60)

    # Salvare Grafic
    plot_filename = os.path.join(results_dir, f"double_q_plot_{timestamp}.png")
    plot_final(scores, mean_scores, save_path=plot_filename)
    
    # Salvare Replay (din agentul tău)
    if len(agent.best_game_frames) > 0:
        replay_filename = os.path.join(results_dir, f"double_q_best_replay_{timestamp}.npy")
        
        # Agentul tău stochează obiecte Point, trebuie să le convertim pentru salvare
        # (sau agentul le are deja ca dict-uri? Verificăm codul tău: le are ca Point în listă de dicts)
        # NPY save cere structuri serializabile. 
        # Convertim manual aici pentru siguranță:
        
        serializable_frames = []
        for frame in agent.best_game_frames:
            serializable_frames.append({
                'snake': [{'x': p.x, 'y': p.y} for p in frame['snake']], # Convert Point to dict
                'food': {'x': frame['food'].x, 'y': frame['food'].y},
                'score': frame['score'],
                'direction': frame['direction']
            })

        np.save(replay_filename, {
            'frames': serializable_frames, 
            'score': agent.best_game_score,
            'w': agent.best_game_w,
            'h': agent.best_game_h
        })
        print(f"✅ Replay saved: {replay_filename}")
        
        # Opțiune vizionare
        print("\nTo watch replay, run 'python replay_best_game.py' and select this file.")

if __name__ == '__main__':
    train()