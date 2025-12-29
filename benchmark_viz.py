import matplotlib.pyplot as plt
import numpy as np

def create_benchmark_dashboard():
    
    agents = ['Double Q-Learning', 'Dueling DQN', 'DQN', 'PPO']
    categories = ['Tabular', 'Deep RL', 'Deep RL', 'Policy-Based']
    colors = ['#FF9999', '#66B2FF', '#99CCFF', '#99FF99']
    
    max_scores = [17, 52, 56, 105]
    
    mean_scores = [2.5, 11.5, 16.1, 51.1]
    
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('FINAL BENCHMARK: Snake AI Agents Comparison', fontsize=20, fontweight='bold')
    
    
    bars1 = axs[0, 0].bar(agents, max_scores, color=colors, edgecolor='black', alpha=0.8)
    axs[0, 0].set_title('Scor Maxim Atins (Peak Performance)', fontsize=14, fontweight='bold')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].grid(axis='y', alpha=0.3)
    
    
    for bar in bars1:
        height = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    
    bars2 = axs[0, 1].bar(agents, mean_scores, color=colors, edgecolor='black', alpha=0.8)
    axs[0, 1].set_title('Scor Mediu (Consistency)', fontsize=14, fontweight='bold')
    axs[0, 1].set_ylabel('Average Score')
    axs[0, 1].grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12)

    # --- PLOT C: Eficiența Învățării (Simulare Trend) ---
    axs[1, 0].set_title('Learning Dynamics (Schematic)', fontsize=14, fontweight='bold')
    x = np.linspace(0, 5000, 100)
    
    # Double Q
    y_dq = 2.5 * (1 - np.exp(-x/500)) 
    axs[1, 0].plot(x, y_dq, color=colors[0], label='Double Q', linewidth=2, linestyle='--')
    
    # DQN 
    y_dqn = 16 * (x/5000) * 0.9 + np.random.normal(0, 0.5, 100).cumsum()/10
    axs[1, 0].plot(x, np.abs(y_dqn), color=colors[2], label='DQN', linewidth=2)
    
    # PPO
    y_ppo = 51 * (x/5000)**1.2 
    axs[1, 0].plot(x, y_ppo, color=colors[3], label='PPO (SOTA)', linewidth=3)
    
    axs[1, 0].set_xlabel('Games')
    axs[1, 0].set_ylabel('Mean Score')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].axis('off')
    axs[1, 1].set_title('Rezumat Tehnic', fontsize=14, fontweight='bold')
    
    cell_text = []
    for i in range(len(agents)):
        cell_text.append([agents[i], categories[i], max_scores[i], f"{mean_scores[i]:.1f}"])
        
    table = axs[1, 1].table(cellText=cell_text,
                           colLabels=['Agent', 'Category', 'Max Score', 'Mean Score'],
                           cellLoc='center',
                           loc='center',
                           colColours=['#eeeeee']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2) 

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = 'final_benchmark.png'
    plt.savefig(save_path, dpi=300)
    print(f"Benchmark plot saved successfully: {save_path}")
    plt.show()

if __name__ == "__main__":
    create_benchmark_dashboard()