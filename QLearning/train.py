import sys
import os
# Adaugă directorul părinte (root) la calea de căutare a modulelor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Acum importurile vor funcționa corect
import pygame
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import numpy as np
import cv2

from agent import Agent
from game import SnakeGame, BLOCK_SIZE # <- Acest import va funcționa acum

# --- Funcție nouă pentru plotare ---
plt.ion() # Activăm modul interactiv pentru matplotlib

def plot(scores, mean_scores):
    """
    Funcție pentru a plota scorurile live.
    """
    ipythondisplay.clear_output(wait=True)
    ipythondisplay.display(plt.gcf())
    plt.clf()
    plt.title('Progresul Antrenamentului')
    plt.xlabel('Număr Jocuri')
    plt.ylabel('Scor')
    plt.plot(scores, label='Scor Joc Curent')
    plt.plot(mean_scores, label='Scor Mediu')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)


# --- Clasă nouă pentru înregistrare video ---
class VideoRecorder:
    def __init__(self, filename, frame_size, fps):
        self.filename = filename
        self.frame_size = frame_size
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec pentru .mp4
        self.writer = None
        self.is_recording = False

    def start_recording(self):
        """Începe înregistrarea unui nou fișier video."""
        # Creează un nume de fișier unic cu timestamp
        unique_filename = f"{self.filename}_{np.random.randint(1000, 9999)}.mp4"
        self.writer = cv2.VideoWriter(unique_filename, self.fourcc, self.fps, self.frame_size)
        self.is_recording = True
        print(f"A început înregistrarea: {unique_filename}")

    def add_frame(self, frame_surface):
        """Adaugă un cadru la video."""
        if self.is_recording and self.writer is not None:
            # Obține pixelii din suprafața Pygame într-un format pe care OpenCV îl înțelege
            frame_rgb = pygame.surfarray.array3d(frame_surface)
            frame_bgr = cv2.cvtColor(frame_rgb.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)

    def stop_recording(self):
        """Oprește și salvează fișierul video."""
        if self.is_recording and self.writer is not None:
            self.writer.release()
            self.is_recording = False
            self.writer = None
            print("Înregistrare finalizată.")

def train():
    """
    Bucla principală care antrenează agentul, cu plotare și înregistrare.
    """
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame() # w=640, h=480

    # Inițializează recorder-ul video
    video_recorder = VideoRecorder(
        filename='record_snake',
        frame_size=(game.w, game.h),
        fps=10  # Poți ajusta FPS-ul pentru video
    )

    while True:
        # Dacă agentul atinge un nou record, vom dori să pornim înregistrarea la următorul joc
        should_record_next_game = (game.score > record and game.score > 0)

        # 1. Obține starea curentă
        state_old = agent.get_state(game)

        # 2. Obține acțiunea de la agent
        final_move = agent.get_action(state_old)

        # *** Înregistrare video: Adaugă cadru ***
        # Trebuie să facem asta înainte ca jocul să se termine și să se reseteze
        video_recorder.add_frame(game.display)

        # 3. Execută acțiunea în joc și primește feedback
        reward, done, score = game.step(final_move)
        state_new = agent.get_state(game)

        # 4. Antrenează agentul (short memory)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Stochează experiența
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # *** Înregistrare video: Oprește înregistrarea ***
            video_recorder.stop_recording()

            # Jocul s-a terminat, resetează
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # Aici poți salva modelul (Q-table)
                # np.save('q_table.npy', agent.q_table)

            print('Joc', agent.n_games, 'Scor', score, 'Record:', record)

            # *** Plotare: Actualizează graficul ***
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

            # *** Înregistrare video: Pornește înregistrarea pentru următorul joc dacă s-a atins un record ***
            if should_record_next_game:
                video_recorder.start_recording()

if __name__ == '__main__':
    train()