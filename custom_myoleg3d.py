from motion import Motion
import numpy as np
import os
from myosuite.utils import gym
from myosuite.envs.myo.base_v0 import BaseV0 
from scipy.spatial.transform import Rotation as R

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, mode="walk", motion_path=None): # mode is "stand" or "walk"
        self.mode = mode
        self.dt = 0.01
        self.t = 0.0
        self.ref_motion = None if motion_path is None else Motion(os.getcwd() + motion_path, "myoleg")
        if self.ref_motion:
            self.update_ref_pose(self.t)        
        env = gym.make('myoLegWalk-v0',normalize_act=False) 
        
        self.cur_obs_dict = {}

        super().__init__(env)
        obs, _ = self.reset()   
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
        # You may define your own action space here

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self.t = np.random.uniform(0, 1.0) 
        # Reference motion time
        # Set the initial position and velocity (For 2-2)
        if self.ref_motion: # Reference State Initialization
            self.update_ref_pose(self.t)
            _, info = BaseV0.reset(self.env.unwrapped, reset_qpos = self.target_pos, reset_qvel=self.target_vel, **kwargs)
        
        if self.mode == "stand":
            stand_pos = np.zeros(self.env.unwrapped.sim.model.nq)
            stand_pos[2] += 1.0
            stand_pos[3:7] = np.array([-0.7071,0,0,0.7071])
            _, info = BaseV0.reset(self.env.unwrapped, reset_qpos = stand_pos, reset_qvel=np.zeros(self.env.unwrapped.sim.model.nv), **kwargs)


        self.cur_obs_dict = self.env.unwrapped.get_obs_dict(self.env.unwrapped.sim)
        print(self.cur_obs_dict.keys())
        custom_obs = self.custom_observation()
        return custom_obs, info

    def step(self, action):
        # ** You can define your own action scale here, but note that the action given to self.env is clipped between 0 and 1 (e.g. self.env.step(scale * action))
        obs, _, terminated, truncated, info = self.env.step(action) 
        self.cur_obs_dict = self.env.unwrapped.get_obs_dict(self.env.unwrapped.sim)
        self.t += self.dt

        custom_obs = self.custom_observation()
        custom_reward = self.custom_reward()
        custom_terminated = self.custom_terminated(terminated)
        custom_truncated = self.custom_truncated(truncated)
        
        # ** You can change the update_ref pose timing in step function
        if self.ref_motion:
            self.update_ref_pose(self.t)

        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def update_ref_pose(self, time):
        self.target_pos = self.ref_motion.get_ref_poses(time)
        self.target_vel = np.zeros(len(self.target_pos) - 1)
        _next_pos = self.ref_motion.get_ref_poses(time + self.dt)

        # A simplified velocity calculation
        self.target_vel[6:] = (_next_pos - self.target_pos)[7:] / self.dt
        self.target_vel[3:6] = (R.from_quat(_next_pos[3:7]) * R.from_quat(self.target_pos[3:7]).inv()).as_rotvec() / self.dt
        self.target_vel[:3] = (_next_pos - self.target_pos)[:3] / self.dt
        
    def custom_terminated(self, terminated):
        if self.unwrapped.sim.data.qpos[2] < 0.75:
            terminated = True
        else:
            terminated = False
        # TODO: Implement your own termination condition

        return terminated
    
    def custom_truncated(self, truncated):
        # TODO: Implement your own truncation condition

        return truncated

    def custom_observation(self):
        # Default Implementation
        obs_dict = self.cur_obs_dict
        if self.ref_motion:
            obs_dict["target"] = self.target_pos.copy()
        
        if self.mode == "walk":
            # TODO : Implement your own observation condition
            pass
        elif self.mode == "stand":
            # TODO : Implement your own observation condition
            pass
        obs = np.concatenate([v.flatten() for k, v in obs_dict.items()])[2:]
        return obs

    def custom_reward(self):
        r = 0.0
        
        # q 값에서 limitation 이 강하게 걸려있어서 의미가 없는 값들은 모두 0으로 처리하는 것을 권장합니다 (아래 주석 참고)
        # 예시) q_diff = self.target_pos - self.unwrapped.sim.data.qpos; q_diff[10:12] *= 0.0; q_diff[13:15] *= 0.0; q_diff[16:21] *= 0.0 # q_diff[24:26] *= 0.0; q_diff[27:29] *= 0.0; q_diff[30:] *= 0.0
        
        # TODO: Implement your own reward condition
        # Default Implementation
        
        if self.mode == "walk":
            if self.ref_motion:
                # TODO : Implement your own observation condition
                pass
            else:
                # TODO : Implement your own observation condition
                pass
        
        if self.mode == "stand":
            # TODO : Implement your own observation condition
            pass
        
        return r 

## Test
if __name__ == "__main__":
    env = CustomEnvWrapper(motion_path="/asset/motions/walk.bvh")
    obs, _ = env.reset() 
    idx = 0
    import time

    while True:
        env.mj_render()
        start_t = time.time()
        action = env.action_space.sample()

        # ***** Please uncomment the line below to view the reference motion.***** 
        # env.unwrapped.sim.set_state(qpos=env.target_pos, qvel=env.target_vel*0.0)
        # action *= 0.0
    
        idx += 1
        obs, reward, terminated, truncated, info = env.step(action)
        
        
        while (time.time() - start_t) < 0.01: # time sync
            pass
        if terminated or truncated:
            obs, _ = env.reset()
