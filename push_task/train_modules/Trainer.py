import torch
import torch.nn as nn
from model_structures.Buffer import Buffer
from collections import deque
from tqdm import tqdm
from envs.utils import get_obs_split,plot_info_on_frame
from envs import make_env
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import copy



class BCD_trainer:
    def __init__(
        self,
        student_policy: nn.Module,
        teacher_policy: nn.Module,
        discriminator: nn.Module,
        env_id: str,
        device: torch.device,
        save_path: str,
        buffer_size: int = 2000000,
        num_cpu: int = 16,
        max_episode_steps: int = 200,
        seq_len=10,
        batch_size: int = 512,
        record_video: bool = False,
    ):
        self.student_policy = student_policy.to(device)
        self.discriminator = discriminator.to(device)
        self.teacher_policy = teacher_policy
        self.env = make_env(
            env_id,
            max_episode_length=max_episode_steps,
            remove_model=False,
            student_obs=True,
        )()
        sample_env = make_env(
            env_id,
            max_episode_length=max_episode_steps,
            remove_model=False,
            student_obs=True,
        )()
        sample_env.close()

        self.device = device
        self.obs_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        self.buffer = Buffer(buffer_size, self.obs_space, self.action_space)
        self.max_episode_steps = max_episode_steps
        self.save_path = save_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.env_id = env_id
        self.buffer_size = buffer_size
        self.record_video = record_video
    def train(
        self,
        num_epoch: int = 100,
        eval_its: int = 100000,
        eval_ep: int = 10,
        model_sv_path=None,
        sv_interval=None,
    ) -> None:
        torch.autograd.set_detect_anomaly(True)
        self.student_policy.train()
        self.discriminator.train()
        self.eval_env = make_env(
            self.env_id, max_episode_length=self.max_episode_steps, remove_model=False, student_obs=True
        )()

        iteration = 0
        for epoch in tqdm(range(num_epoch), desc="CritiQ Iterations"):
            iteration += 1
            bc_loss_total = 0.0
            disc_loss_total = 0.0

            all_batches = self.buffer.sample_all_seq_batches(self.batch_size, self.device, self.seq_len)
            for state_seq_batch, action_seq_batch in all_batches:
                _, student_obs = get_obs_split(state_seq_batch, self.env)
                obs_seq_batch = torch.cat(list(student_obs.values()), dim=-1)
                bc_action_seq_batch = action_seq_batch.clone()
                for i_action in range(self.seq_len):
                    bc_action_seq_batch[:, i_action, :] = self.student_policy.forward(obs_seq_batch[:, i_action, :])

                mse_loss = self.student_policy.criterion(bc_action_seq_batch[:, -1, :], action_seq_batch[:, -1, :])

                bc_state_seq = obs_seq_batch

                expert_label_ = 0.6
                bc_label_ = 0.4
                expert_label = torch.ones((obs_seq_batch.size(0), 1), device=self.device) * expert_label_
                bc_label = torch.ones((obs_seq_batch.size(0), 1), device=self.device) * bc_label_

                obs_act_seq_batch = torch.cat([obs_seq_batch, action_seq_batch], dim=-1).view(obs_seq_batch.shape[0], -1)
                bc_obs_act_seq_batch = torch.cat([bc_state_seq.detach(), bc_action_seq_batch.detach()], dim=-1).view(
                    bc_action_seq_batch.shape[0], -1
                )

                expert_prob = self.discriminator(obs_act_seq_batch)
                bc_prob = self.discriminator(bc_obs_act_seq_batch)
                with torch.no_grad():
                    discriminator_reward = torch.mean((bc_prob - expert_label_) ** 2)

                bc_loss = mse_loss + discriminator_reward * 0.1

                self.student_policy.optimizer.zero_grad()
                bc_loss.backward()
                self.student_policy.optimizer.step()

                disc_loss = self.discriminator.criterion(expert_prob, expert_label) + self.discriminator.criterion(
                    bc_prob, bc_label
                )
                if epoch % 3 == 0:
                    self.discriminator.optimizer.zero_grad()
                    disc_loss.backward()
                    self.discriminator.optimizer.step()

                bc_loss_total += bc_loss.item()
                disc_loss_total += disc_loss.item()

            avg_bc_loss = bc_loss_total / iteration
            avg_disc_loss = disc_loss_total / iteration
            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}, Average BC Loss: {avg_bc_loss:.4f}, Averate Disc Loss: {avg_disc_loss:.4f}"
                )

            if iteration % eval_its == 0:
                success_rate, mean_reward = self.evaluate(eval_ep)
                print(f"Iteration {iteration}, Success Rate: {success_rate:.2f}, Mean Reward: {mean_reward:.2f}")

            if sv_interval is not None and iteration % sv_interval == 0:
                self.save_models(model_sv_path)

    def collect_data(self, num_traj: int, reset_bank=None, sv_dir=None):

        if os.path.exists(sv_dir):
            os.system(f"rm -rf {sv_dir}")
        os.makedirs(sv_dir, exist_ok=True)

        i = 0
        failed_traj_count = 0
        cnt_sequential_fail = 0
        desc_str = "Collecting data"
        with tqdm(total=num_traj, desc=desc_str, unit="traj") as pbar:
            while i < num_traj:
                screens = []
                if reset_bank is None:
                    obs, _ = self.env.reset()
                else:
                    data = reset_bank[np.random.randint(0, len(reset_bank))]
                    obs, info = self.env.reset(data=data)

                traj_obs = {k: [] for k in self.env.observation_space.keys()}
                traj_actions = []

                teacher_actions = np.zeros(self.env.action_space.shape[0])
                for _ in range(self.seq_len - 1):
                    for o in self.env.observation_space.keys():
                        traj_obs[o].append(obs[o])
                    traj_actions.append(teacher_actions)

                success = False
                j = 0
                done = False
                while j < self.max_episode_steps and not done:
                    teacher_obs, student_obs = get_obs_split(obs, self.env)
                    teacher_actions = self.teacher_policy.predict(teacher_obs, deterministic=True)[0]
                    for o in self.env.observation_space.keys():
                        traj_obs[o].append(obs[o])
                    traj_actions.append(teacher_actions)
                    obs, _, done, trunc, info = self.env.step(teacher_actions)
                    if done:
                        success = info["is_success"]
                    j += 1
                    if self.record_video and i < 10:
                        screens.append(plot_info_on_frame(Image.fromarray(np.uint8(self.env.render())), info, font_size=10))

                if success:
                    i += 1
                    cnt_sequential_fail = 0
                    pbar.update(1)
                    pbar.set_postfix(collected=i, failed=failed_traj_count)
                    for ob in self.env.observation_space.keys():
                        traj_obs[ob] = np.array(traj_obs[ob])
                    traj_actions = np.array(traj_actions)
                    self.buffer.add(traj_obs, traj_actions)

                    if self.record_video and i < 10 and len(screens) > 0:
                        file_name = "teacher_" + str(i) + ".mp4"
                        sv_name = os.path.join(sv_dir, file_name)
                        moviepy.video.io.ImageSequenceClip.ImageSequenceClip(screens, fps=int(1 / (0.005 * 20))).write_videofile(
                            sv_name
                        )
                else:
                    failed_traj_count += 1
                    cnt_sequential_fail += 1
                    pbar.set_postfix(collected=i, failed=failed_traj_count)
                    if cnt_sequential_fail > 20:
                        cnt_sequential_fail = 0
                        i += 1

    def evaluate(self, num_ep):
        successes = []
        total_rewards = []

        self.student_policy.eval()
        for _ in range(num_ep):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            for _ in range(self.max_episode_steps):
                with torch.no_grad():
                    teacher_obs, student_obs = get_obs_split(obs, self.env)
                    ob = {k: torch.tensor(v[np.newaxis, ...].copy(), dtype=torch.float32, device=self.device) for k, v in student_obs.items()}
                    action = self.student_policy(ob).cpu().numpy()[0]
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                if done:
                    break
            successes.append(info["is_success"])
            total_rewards.append(episode_reward)

        mean_success_rate = np.mean(successes)
        mean_reward = np.mean(total_rewards)
        return mean_success_rate, mean_reward

    def save_models(self, path: str, iter=None):
        if iter is None:
            sv_name = ["student_push.pth", "discriminator_push.pth"]
        else:
            sv_name = [f"student{iter}.pth", f"discriminator{iter}.pth"]

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.student_policy.state_dict(), os.path.join(path, sv_name[0]))
        torch.save(self.discriminator.state_dict(), os.path.join(path, sv_name[1]))

    def augment_data(self, max_ep_length, bank, model, discriminator, sim, device, num_ep=10, seq_len=10, aug_iter=0, cnt_threshold=5):
        
        prob_threshold1 = 0.35
        prob_threshold2 = 0.65
        mean_ep_rew = 0
        cnt_detected = 0

        traj_obs = {o: deque(maxlen=seq_len) for o in sim.observation_space.keys()}
        traj_actions = deque(maxlen=seq_len)

        for i in range(num_ep):
            obs, info = sim.reset()
            min_prob = 1
            prob = [min_prob, 0, 0]
            ep_reward = 0

            action = np.zeros(sim.action_space.shape[0])
            for _ in range(seq_len):
                for o in sim.observation_space.keys():
                    traj_obs[o].append(torch.tensor(obs[o], dtype=torch.float32, device=device))
                traj_actions.append(torch.tensor(action, dtype=torch.float32, device=device))

            for j in range(max_ep_length):
                for o in sim.observation_space.keys():
                    traj_obs[o].append(torch.tensor(obs[o], dtype=torch.float32, device=device))

                _, student_obs = get_obs_split(traj_obs, env=sim)
                obs_seq_batch = torch.cat([torch.stack(list(v)) if isinstance(v, deque) else v for v in student_obs.values()], dim=-1).unsqueeze(0)
                with torch.no_grad():
                    action = model.predict(obs_seq_batch[:, -1, :])
                    action = action.cpu().reshape(-1)

                obs, r, done, trunc, info = sim.step(action)
                traj_actions.append(torch.tensor(action, dtype=torch.float32, device=device))

                obs_seq_tensors = {
                    o: torch.stack([torch.tensor(x, dtype=torch.float32, device=device) if isinstance(x, np.ndarray) else x for x in student_obs[o]], dim=0)
                    for o in student_obs.keys()
                }
                obs_seq_batch = torch.cat(list(obs_seq_tensors.values()), dim=-1).unsqueeze(0)

                act_seq_batch = torch.stack(list(traj_actions)).unsqueeze(0)
                bc_obs_act_seq_batch = torch.cat([obs_seq_batch.detach(), act_seq_batch.detach()], dim=-1).view(1, -1)

                bc_prob = discriminator(bc_obs_act_seq_batch)
                if bc_prob.item() < prob_threshold1 or bc_prob.item() > prob_threshold2:
                    prob[2] += 1
                    cnt_detected += 1
                else:
                    cnt_detected = 0
                if min_prob > bc_prob.item():
                    min_prob = bc_prob.item()
                    prob[0] = min_prob
                    prob[1] = info["step"]
                info["prob"] = bc_prob.item()
                info["min_prob"] = prob
                if cnt_detected > cnt_threshold:
                    state = copy.deepcopy(sim.get_state())
                    bank.append(state)
                    break
                if done:
                    suc = info["is_success"]
                    break

                mean_ep_rew += r
                ep_reward += r

            GOAL = info["target"]
            
        # print(f"mean_ep_reward: {mean_ep_rew/num_ep}")
        # print(f"mean_suc: {num_suc/num_ep}")
        return bank

    def load_student_model(self, path_student: str, path_discriminator: str):
        self.student_policy.load_state_dict(torch.load(path_student), weights_only=True)
        self.discriminator.load_state_dict(torch.load(path_discriminator), weights_only=True)

    def weight_share(self):
        self.student_policy.load_state_dict({k: v for k, v in self.teacher_policy.state_dict().items() if "policy_mlp" in k})
