import numpy as np
import gym
from collections import deque
import random

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

from pathlib import Path

display = Display(visible=0, size=(1400, 900))
display.start()

def show_video(path):
        html = []
        for mp4 in Path(path).glob("*.mp4"):
            video_b64 = base64.b64encode(mp4.read_bytes())
            html.append('''<video alt="{}" autoplay 
                        loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{}" type="video/mp4" />
                    </video>'''.format(mp4, video_b64.decode('ascii')))
        ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

class ExperienceBuffer:
    """Buffer used to train the ANN in the DRL"""
    def __init__(self, max_size=50000):
        self.buffer= deque(maxlen=max_size)
        
        
    def push(self, state, action, reward, next_state, done):
        experience= (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        
    
    def buffer_len(self):
        return len(self.buffer)

