from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import wuji_left_hand_cube_rotate_env_cfg
from .rl_cfg import wuji_left_hand_cube_rotate_ppo_cfg


register_mjlab_task(
    task_id="Mjlab-Wuji-Left-HandCube-Rotate",
    env_cfg=wuji_left_hand_cube_rotate_env_cfg(),
    play_env_cfg=wuji_left_hand_cube_rotate_env_cfg(play=True),
    rl_cfg=wuji_left_hand_cube_rotate_ppo_cfg(),
)