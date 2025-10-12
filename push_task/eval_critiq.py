# CritiQ/push_task/bc_test.py
from __future__ import annotations
import sys
import torch
from envs import make_env
from stable_baselines3 import SAC
from model_structures.BC import BCNetwork
from model_structures.Discriminator import Discriminator
from train_modules.Validation import val_env

from config.config import TestConfig, load_test_config
def _select_device(cfg: TestConfig) -> torch.device:
    if cfg.device in ("cpu", "cuda"):
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg_path: str) -> None:
    cfg: TestConfig = load_test_config(cfg_path)
    device = _select_device(cfg)

    sample_env = make_env(cfg.env_id, depth_rendering=False, student=False)()
    sample_env.close()

    teacher = SAC.load(cfg.teacher_path, sample_env, device=device)
    env = make_env(cfg.env_id, depth_rendering=False, student=True)()

    obs_size = sum(
        env.observation_space[k].shape[0]
        for k in env.observation_space
        if k not in env.unwrapped.student_filter_list()
    )
    student = BCNetwork(obs_size, env.action_space.shape[0], device=device)
    discriminator = Discriminator(obs_size, env.action_space.shape[0], device=device, seq_len=cfg.seq_len)

    student.load_model(cfg.student_path)
    discriminator.load_model(cfg.discriminator_path)

    val_env(
        cfg.max_episode_steps,
        student,
        discriminator,
        cfg.save_path,
        env,
        device,
        num_ep=cfg.num_episodes,
        seq_len=cfg.seq_len,
        noisy=False,
    )
        

if __name__ == "__main__":
    if "--cfg" in sys.argv:
        idx = sys.argv.index("--cfg")
        try:
            cfg_path = sys.argv[idx + 1]
        except IndexError:
            raise SystemExit("Error: please provide a config file path after --cfg.")
    else:
        raise SystemExit("Usage: python bc_test.py --cfg path/to/config.(json|yaml|toml)")

    main(cfg_path)
