import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    """
    Funcție helper pentru plotare simplificată.
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt. clf()
    plt.title('Training Progress - Deep Q-Learning')
    plt. xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', color='blue', alpha=0.6)
    plt.plot(mean_scores, label='Mean Score', color='red', linewidth=2)
    plt. ylim(ymin=0)
    
    # Adaugă textul pentru ultimele valori
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    plt.pause(0.1)