import argparse
from stable_baselines3 import PPO
import gymnasium as gym
import myosuite  # Ensure myosuite is installed and imported

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.zip)")
parser.add_argument("--motion", type=str, default=None, help="Reference Mtoion path")
parser.add_argument("--mode", type=str, default="walk", help="walk or stand")
args = parser.parse_args()

# Load Myosuite environment
from custom_myoleg3d import CustomEnvWrapper
if args.motion:
    motion_path = "/asset/motions/" + args.motion
else:
    motion_path = None

env = CustomEnvWrapper(motion_path=motion_path, mode=args.mode)

# Load model if specified
trained_model = PPO.load(args.model) if args.model is not None else None
obs, _ = env.reset()

# Interaction loop
while True:
    if trained_model is not None:
        action, _ = trained_model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    env.mj_render()
    if terminated or truncated:
        obs, _ = env.reset()
