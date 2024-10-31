from environment import GameEnvironment
from model import create_model
from training import self_play_training

if __name__ == "__main__":
    env = GameEnvironment()
    model = create_model(input_shape, output_shape)
    self_play_training(env, model, epochs=1000, episodes_per_epoch=10)
