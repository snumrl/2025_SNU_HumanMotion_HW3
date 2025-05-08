from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# You are allow to change the number of environments according to your cpu.
N_ENVS = 32

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
    log_std_init=-1.0 
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.zip)")
parser.add_argument("--motion", type=str, default=None, help="Reference Mtoion path")
parser.add_argument("--mode", type=str, default="walk", help="walk or stand")
args = parser.parse_args()


checkpoint_callback = CheckpointCallback(
    save_freq=10000,  
    save_path='./checkpoints/',
    name_prefix="checkpoint_" + args.mode + (args.motion if args.motion else ""),
)

if __name__ == "__main__":
    num_cpu = N_ENVS
    from custom_myoleg3d import CustomEnvWrapper
    def make_env(motion_path, mode):
        def _init():
            env = CustomEnvWrapper(motion_path = motion_path, mode = mode)
            return env
        return _init
    
    if args.motion: 
        motion_path = "/asset/motions/" + args.motion
    else:
        motion_path = None

    env = SubprocVecEnv([make_env(motion_path = motion_path, mode=args.mode) for _ in range(num_cpu)])
    env = VecMonitor(env)

    # You are allowed to change any parameters (e.g. learning rate, batch size, etc.) in the PPO algorithm.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs, device="cpu", learning_rate=0.0001, batch_size=1024, n_steps=512)
    model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
