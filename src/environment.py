import numpy as np

class GameEnvironment:
    def __init__(self):
        # Initialize the game state, etc.
        pass
    
    def reset(self):
        # Reset the environment to its initial state
        return self.state

    def step(self, action):
        # Apply the action and return the new state, reward, and done status
        return new_state, reward, done
