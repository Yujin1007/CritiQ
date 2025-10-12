import torch
import numpy as np
import os
from collections import deque

from PIL import Image
import moviepy.video.io.ImageSequenceClip
import moviepy.config as mp_config
from envs.utils import plot_info_on_frame

def val_env(max_ep_length, model, discriminator, sv_path, sim, device, num_ep=20, seq_len=10, noisy=True, save_video=True):
    if os.path.exists(sv_path):
        os.system(f"rm -rf {sv_path}")

    os.makedirs(sv_path, exist_ok=True)

    mean_ep_rew = 0
    num_suc = 0
    video_freq = 1

    traj_obs = {o: deque(maxlen=seq_len) for o in sim.observation_space.keys()}
    traj_actions = deque(maxlen=seq_len)
    
    for i in range(num_ep):
        obs, info = sim.reset(add_noise=noisy)
        save_dict = {}

        min_prob = 1
        prob = [min_prob, 0, 0]
        screens  = []
        suc = False
        ep_reward = 0

        action = np.zeros(sim.action_space.shape[0])
        for _ in range(seq_len):  # Fill with initial observations
            for o in sim.observation_space.keys():
                traj_obs[o].append(torch.tensor(obs[o], dtype=torch.float32, device=device))  # ✅ Convert to Tensor
            traj_actions.append(torch.tensor(action, dtype=torch.float32, device=device))  # ✅ Ensure consistent dtype & device
        
        for j in range(max_ep_length):
            for o in sim.observation_space.keys():
                traj_obs[o].append(torch.tensor(obs[o], dtype=torch.float32, device=device))  # ✅ Automatically removes old ones
            
            student_obs = {k : traj_obs[k] for k in traj_obs if k not in sim.unwrapped.student_filter_list()}
            
            obs_seq_batch = torch.cat(
                [torch.stack(list(v)) if isinstance(v, deque) else v for v in student_obs.values()],
                dim=-1
            ).unsqueeze(0)
                
            with torch.no_grad():
                action = model.predict(obs_seq_batch[:,-1,:], deterministic=True)
                action = action[0]
            
            obs, r, done, trunc, info = sim.step(action.cpu())
            traj_actions.append(torch.tensor(action, dtype=torch.float32, device=device))
            
            obs_seq_tensors = {
                o: torch.stack([torch.tensor(x, dtype=torch.float32, device=device) if isinstance(x, np.ndarray) else x
                                for x in student_obs[o]], dim=0)
                for o in student_obs.keys()
            }
            obs_seq_batch = torch.cat(list(obs_seq_tensors.values()), dim=-1).unsqueeze(0) 
            
            act_seq_batch = torch.stack(list(traj_actions)).unsqueeze(0)
            seq = torch.cat((obs_seq_batch, act_seq_batch), dim=-1).flatten()
            bc_prob = discriminator(seq)
            if bc_prob.item()<0.3:
                prob[2] += 1
            if min_prob > bc_prob.item():  
                min_prob = bc_prob.item()
                prob[0] = min_prob
                prob[1] = info["step"]
            info["prob"] = bc_prob.item()
            info["min_prob"] = prob
            screens+= [plot_info_on_frame(Image.fromarray(np.uint8(sim.render())), info, font_size=10)]#info["frames"]
            if done:
                suc = info["is_success"]
                break
        
            mean_ep_rew += r
            ep_reward += r
        if save_video:
            if i % video_freq == 0 or not suc:
                # file_name = str(suc) + "_" + info["target"] + "_" + str(obs["handle_0_status"])+ str(obs["handle_1_status"])+ str(obs["handle_2_status"])+str(i)
                file_name = str(suc) + "_" + str(info["target"]) +str(i)
                mp_config.change_settings({"IMAGEMAGICK_BINARY": "auto"})  # Optional config
                mp_config.default_logging = "error"  # Suppress logs
                moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                            screens, fps=int(1/(0.005*20))
                        ).write_videofile(f"{sv_path}/{file_name}.mp4")
        num_suc += suc
       
    nf_name = f"{sv_path}_{num_suc/num_ep}"
    print(f"mean_ep_reward: {mean_ep_rew/num_ep}")
    print(f"mean_suc: {num_suc/num_ep}")
    if os.path.exists(sv_path):
        os.rename(sv_path, nf_name)
    else:
        print(f"folder '{sv_path}' doesn't exist.")
    return num_suc/num_ep
