1、训练
	注意将 vim .venv/lib/python3.13/site-packages/rsl_rl/utils/wandb_utils.py  原来的
	settings=wandb.Settings(start_method="thread"), 修改 为settings=wandb.Settings(), 然后再进行训练
	$ uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate --env.scene.num-envs 128
	训练结果保存在 wandb 中
2、验证
	预训练好的模型：uv run python scripts/play.py Mjlab-Leap-Left-Custom-HandCube-Rotate --checkpoint-file ckpts/
	leap_left_custom_model_4900.pt
	自己模型：uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate --checkpoint-file logs/rsl_rl/ 
	leap_left_hand_cube_rotate/2026-09-20_17-57-19/model_4300.pt
