import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity, n_steps, gamma):
        assert n_steps > 0
        self.buffer = collections.deque(maxlen=capacity)
        self.temp_buffer = collections.deque(maxlen=n_steps)
        self.gamma = gamma
        self.n_steps = n_steps

    def __len__(self):
        return len(self.buffer)
    
    def _make_n_step_experience(self):
        Rt = 0
        size = len(self.temp_buffer)
        for i in range(size):
            Rt += (self.gamma ** i) * self.temp_buffer[i]
        self.buffer.append(
            Experience(
                self.temp_buffer[0].state, 
                self.temp_buffer[0].reward,
                Rt,
                self.temp_buffer[size - 1].done,
                self.temp_buffer[size - 1].new_state
                )
            )

    def append(self, experience):
        self.temp_buffer.append(experience)
        if experience.done:
            self._make_n_step_experience()
            self.temp_buffer.clear()
        elif len(self.temp_buffer) == self.n_steps:
            self._make_n_step_experience()
            # carry on here - need to consider done states 

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)