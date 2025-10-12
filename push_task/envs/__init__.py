import gymnasium
from envs.create_env import make_env
from envs.push.single import SinglePush

gymnasium.register(
    "singlepush",
    "envs:SinglePush",
)
