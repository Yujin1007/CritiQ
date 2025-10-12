# CritiQ/push_task/bc_algo.py
from __future__ import annotations
import os
import sys
import pickle
from typing import Optional

import torch
from stable_baselines3 import SAC

from envs import make_env
from config.config import (
    TrainConfig,
    load_config,
    save_config,
    load_config_with_raw,
    save_provided_config,
)
from model_structures.BC import BCNetwork
from model_structures.Discriminator import Discriminator
from train_modules.Trainer import BCD_trainer
from train_modules.Validation import val_env


def _select_device(cfg: TrainConfig) -> torch.device:
    if cfg.device in ("cpu", "cuda"):
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cfg_path: str) -> None:
    # --- 1) Load configuration (and keep raw user-provided dict) ---
    cfg, raw_cfg = load_config_with_raw(cfg_path)

    # --- 2) Prepare output folder & save configs ---
    os.makedirs(cfg.save_path, exist_ok=True)
    # Save the original user-provided config (so we don't overwrite or expand it with defaults)
    # save_provided_config(raw_cfg, cfg.save_path, filename_base="config_provided")
    # Save the resolved config (with defaults applied) for reproducibility
    save_config(cfg, cfg.save_path, filename_base="config_used")

    # --- 3) Prepare device/environment ---
    device = _select_device(cfg)
    sample_env = make_env(cfg.env_id, depth_rendering=False, student=False)()
    sample_env.close()

    # Load teacher policy
    teacher = SAC.load(cfg.teacher_path, sample_env, device=device)

    # Training environment (student=True)
    env = make_env(cfg.env_id, depth_rendering=False, student=True)()

    # Calculate observation size (excluding student filter)
    obs_size = sum(
        env.observation_space[k].shape[0]
        for k in env.observation_space
        if k not in env.unwrapped.student_filter_list()
    )

    student = BCNetwork(obs_size, env.action_space.shape[0], device=device)
    discriminator = Discriminator(obs_size, env.action_space.shape[0], device=device, seq_len=cfg.seq_len)

    # --- 4) Pretrained model loading check ---
    pretrain = cfg.student_path != "n"
    if pretrain:
        student.load_model(cfg.student_path)
        discriminator.load_model(cfg.discriminator_path)

    # --- 5) Build trainer ---
    trainer = BCD_trainer(
        student,
        teacher,
        discriminator,
        cfg.env_id,
        device,
        cfg.save_path,
        num_cpu=cfg.num_cpu,
        max_episode_steps=cfg.max_episode_steps,
        seq_len=cfg.seq_len,
        record_video=cfg.save_video,
    )

    # --- 6) Data collection mode ---
    if cfg.collect_data:
        if not cfg.collect_more_data:
            trainer.collect_data(cfg.num_traj)
            trainer.buffer.save(cfg.buffer_path)
        else:
            with open(cfg.buffer_path, "rb") as file:
                loaded_buffer = pickle.load(file)
            trainer.buffer = loaded_buffer
            print("Loaded buffer length:", len(trainer.buffer))
            trainer.collect_data(cfg.num_traj)
            trainer.buffer.save(cfg.new_buffer_path)
            print("New buffer length:", len(trainer.buffer))

    # --- 7) Start training ---
    print("Training BC")
    with open(cfg.buffer_path, "rb") as f:
        trainer.buffer = pickle.load(f)

    print("LOADED BUFFER LENGTH:", len(trainer.buffer))

    if not pretrain:
        # Keep original 'for buffer' setting
        print("INITIAL BC TRAINING")
        init_bc_epoch = 10#50
        trainer.train(num_epoch=init_bc_epoch)

    # Data augmentation rounds
    for aug_i in range(cfg.data_aug_ep):
        bank = []
        bank = trainer.augment_data(
            cfg.max_episode_steps,
            bank=bank,
            model=student,
            discriminator=discriminator,
            sim=env,
            device=device,
            num_ep=cfg.data_aug_ep,
            seq_len=cfg.seq_len,
            aug_iter=aug_i,
            cnt_threshold=cfg.cnt_threshold,
        )

        sv_path = os.path.join(cfg.save_path, "reset_states")
        os.makedirs(sv_path, exist_ok=True)
        with open(f"{sv_path}/student_states_start_{aug_i}.pickle", "wb") as file:
            pickle.dump(bank, file)

        if len(bank) < cfg.data_aug_ep:
            # Collect data from teacher for each failed state multiplied by aug_traj_mul
            trainer.collect_data(int(len(bank) * cfg.aug_traj_mul), bank, sv_dir=os.path.join(cfg.save_path, "videos", f"teacher_round_{aug_i}"))
        else:
            print("Failed for all turns. Skip data augmentation, train with existing dataset.")

        trainer.train(num_epoch=cfg.num_epoch)
        trainer.save_models(cfg.save_path, aug_i)
        success_rate = val_env(cfg.max_episode_steps, student, discriminator, os.path.join(cfg.save_path, "videos", f"test{aug_i}"), env, device, num_ep=cfg.eval_its, seq_len=cfg.seq_len, save_video=cfg.save_video)
        print(f"Round {aug_i} validation success rate: {success_rate}")
        if success_rate >=0.9:
            print(f"Early stopping at round {aug_i} with success rate {success_rate}")
            print(f"best model saved at {cfg.save_path}/student_push.pth, {cfg.save_path}/discriminator_push.pth")
            trainer.save_models(cfg.save_path)
            break


if __name__ == "__main__":
    # Usage: python -m push_task.bc_algo --cfg path/to/config.(json|yaml|toml)
    # If no args are provided, show usage message.
    if "--cfg" in sys.argv:
        idx = sys.argv.index("--cfg")
        try:
            cfg_path = sys.argv[idx + 1]
        except IndexError:
            raise SystemExit("Error: please provide a config file path after --cfg.")
    else:
        raise SystemExit("Usage: python -m push_task.bc_algo --cfg path/to/config.(json|yaml|toml)")

    main(cfg_path)
