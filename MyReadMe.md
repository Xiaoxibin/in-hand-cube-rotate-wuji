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
3、git 相关
	cd ~/Projects/my-project
	git init
	git add .
	git commit -m "Initial commit"

	git remote add origin git@github.com:Xiaoxibin/my-project.git
	git branch -M main
	git push -u origin main
	修改后：
	git add .
	git commit -m "xxx"
	git push
	
4、wuji_left_hand 1 迁移相关
4.1 mjcf 相关 
	下载https://github.com/wuji-technology/wuji-description/releases?page=2 里的 v2026.05.19源文件
	$ mkdir -p src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/assets
	$ touch src/in_hand_rotation_mjlab/robots/wuji_hand/__init__.py
	$ cp ~/下载/wuji-description-2026.05.19/hand/body/mjcf/left.xml src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/left_hand.xml
	$ cp ~/下载/wuji-description-2026.05.19/hand/body/meshes/left/*.STL \
   src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/assets/
   	left_hand.xml 修改路径为 meshdir="./assets/"
	$ uv run python -m mujoco.viewer   --mjcf=src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/left_hand.xml
	
	uv run python -m mujoco.viewer   --mjcf=src/in_hand_rotation_mjlab/robots/tj_hand/xmls/right.xml
	

4.2 环境验证
	uv run python scripts/play.py Mjlab-Wuji-Left-HandCube-Rotate --num-envs 1 --agent zero
	
	uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --num-envs 1 --agent zero

4.3 grasp_cache文件生成
	代码实现位于scripts/collect_hand_cube_grasp_cache_wuji.py：
	uv run python scripts/collect_hand_cube_grasp_cache_wuji.py   Mjlab-Wuji-Left-HandCube-Rotate
	生成质量与 WUJI_HOME_POS、 CUBE_SPAWN_POS、 cube_size 有关
	产出 cube_size 、joint_pos、 cube_pose_rel
	
	uv run python scripts/collect_hand_cube_grasp_cache_tj.py   Mjlab-TJ-Right-HandCube-Rotate
	
4.4 wuji训练
	uv run python scripts/train.py Mjlab-Wuji-Left-HandCube-Rotate --env.scene.num-envs 128
	uv run python scripts/train.py Mjlab-TJ-Right-HandCube-Rotate --env.scene.num-envs 2048
	
	4.4.1 中断后恢复训练
	uv run python scripts/train.py Mjlab-TJ-Right-HandCube-Rotate \
    --env.scene.num-envs 2048 \
    --agent.resume True \
    --agent.load-run 2026-10-14_11-51-31 \
    --agent.load-checkpoint model_1600.pt

	
4.5 wuji验证 
	uv run python scripts/play.py Mjlab-Wuji-Left-HandCube-Rotate --checkpoint-file logs/rsl_rl/wuji_left_hand_cube_rotate/2026-09-28_18-30-33/model_4999.pt
	uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt


LEAP Hand ： 538AE5 蓝色

tj 参数：
=============================================================================================================================
如果我 2026-10-09_19-27-11 的训练没有 cube_pose_drift_penalty 这项奖励，并且也没有 cube_pose_deviation 这个终止条件，但是有 grasp_cache_file 这个 cache 文件，可能还有些其他的训练参数修改了，现在我能 uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt 来查看效果吗？

绿色：2026-10-09_19-27-11
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt

TJ_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.08)
TJ_HOME_ROT: tuple[float, float, float, float] = (0.707107, -0.707107, 0.0, 0.0)
TJ_HOME_JOINT_POS: dict[str, float] = {
    # =========================
    # Thumb
    # =========================
    "CMC1": -0.9,
    "CMC2":  0.107,
    "CMC3": 0.26,
    "MCPT": -1.12,
    "DIPT": -0.752,

    # =========================
    # Index
    # =========================
    "MCPI2":  0.05,
    "MCPI1": -0.8,
    "PIPI":  -1.25,
    "DIPI":  -0.6,

    # =========================
    # Middle
    # =========================
    "MCPM2": 0.0616,
    "MCPM1": -0.509,
    "PIPM":  -1.34,
    "DIPM":  -1.07,

    # =========================
    # Ring
    # =========================
    "MCPR2": -0.0621,
    "MCPR1": -0.51,
    "PIPR":  -1.35,
    "DIPR":  -0.811,

    # =========================
    # Pinky
    # =========================
    "MCPS2": -0.27,
    "MCPS1": -1,
    "PIPS":  -1.22,
    "DIPS":  -0.613,
}

cube_size : 0.0272
CUBE_SPAWN_POS = (-0.006, 0.043, 0.08)

=============================================================================================================================
棕色 ：2026-10-10_14-12-59
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-10_14-12-59/model_

grasp_cache_file = "tasks/hand_cube/cache/tj_right_grasp_cache.npz"
cube_size : 0.022
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -0.9,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -1.45,     # 原 -1.12,更大对掌
    "DIPT":  -0.45,     # 原 -0.752,拇指伸直一点抬高指尖

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,     # 原 0.05
    "MCPI1": -0.95,     # 原 -0.8
    "PIPI":  -1.40,     # 原 -1.25
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,     # 原 0.0616
    "MCPM1": -0.60,     # 原 -0.509
    "PIPM":  -1.40,     # 原 -1.34
    "DIPM":  -1.07,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,     # 原 -0.0621
    "MCPR1": -0.51,
    "PIPR":  -1.35,
    "DIPR":  -0.811,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.35,
    "MCPS1": -1.00,
    "PIPS":  -1.22,
    "DIPS":  -0.613,
}

=============================================================================================================================
灰色（有中断）：2026-10-12_16-51-38 / 2026-10-12_18-37-24
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-12_18-37-24/model_700.pt 

GRASP_CACHE_FILE = (
    "tasks/hand_cube/cache/tj_right_grasp_cache_24.npz"
)
cube_size : 0.024  (0.0225~0.0255)
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -0.9,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -1.45,     # 原 -1.12,更大对掌
    "DIPT":  -0.45,     # 原 -0.752,拇指伸直一点抬高指尖

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,     # 原 0.05
    "MCPI1": -0.95,     # 原 -0.8
    "PIPI":  -1.40,     # 原 -1.25
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,     # 原 0.0616
    "MCPM1": -0.60,     # 原 -0.509
    "PIPM":  -1.40,     # 原 -1.34
    "DIPM":  -1.07,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,     # 原 -0.0621
    "MCPR1": -0.51,
    "PIPR":  -1.35,
    "DIPR":  -0.811,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.25,
    "MCPS1": -1.00,
    "PIPS":  -1.22,
    "DIPS":  -0.613,
}
=============================================================================================================================
红色：2026-10-13_14-31-44 / 2026-10-13_18-34-48
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-13_14-31-44/model_900.pt

GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_24_v1.npz")
1、修改了奖励函数：添加了 tumble_penalty 和 cube_pose_drift_penalty； 2、impratio 改成 100； 3、 rotate_finite_diff 里的 drift_mode 改成了 "exp"；

=============================================================================================================================
紫色：  2026-10-14_11-51-31 / 2026-10-14_16-36-53
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-14_16-36-53/model_1600.pt

1、cube_size 为 0.028  CUBE_SPAWN_POS = (-0.006, 0.0368, 0.08)  GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_28.npz")；
2、env_cfgs.py 中增加 cumulative_yaw 奖励，step_angle 为 1.5707963（pi/2）；

cube_size = 0.028
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -1.45,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -0.7,
    "DIPT":  -0.65,

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,
    "MCPI1": -0.95,
    "PIPI":  -1.15,
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,
    "MCPM1": -0.80,
    "PIPM":  -1.0,
    "DIPM":  -0.6,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,
    "MCPR1": -0.81,
    "PIPR":  -1.05,
    "DIPR":  -0.611,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.15,
    "MCPS1": -1.30,
    "PIPS":  -0.82,
    "DIPS":  -0.611,
}

=============================================================================================================================
青色 ： 2026-10-14_20-33-08

修改：
1、pose_diff_penalty 中的 joint_tolerance 改成了 0.15； 2、MCT_CMC3_R 中的 collision type 换成了 box；

cube_size = 0.031	CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)	 GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_31.npz")
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -1.45,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -0.7,
    "DIPT":  -0.65,

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,
    "MCPI1": -1.1,
    "PIPI":  -0.9,
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,
    "MCPM1": -0.80,
    "PIPM":  -1.0,
    "DIPM":  -0.6,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,
    "MCPR1": -0.81,
    "PIPR":  -0.7,
    "DIPR":  -0.611,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.15,
    "MCPS1": -1.30,
    "PIPS":  -0.82,
    "DIPS":  -0.611,
}
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-14_20-33-08/model_4400.pt  / 3500.pt
 


黄色 ： 2026-10-15_13-45-37

修改 ： 
1、cube_pose_drift_penalty 权重改为 -1.5 # 原-0.8； position_threshold 改为 0.005 # 原 0.002；

uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-15_13-45-37


=============================================================================================================================
水粉 ： 2026-10-15_11-14-03

cube_size = 0.033	CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)	 GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_33.npz")
1、cube_pose_drift_penalty 权重改为 -1.5 # 原-0.8； position_threshold 改为 0.005 # 原 0.002；


uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-15_11-14-03








待完成：
已知指尖相对于手腕的位姿，利用逆运动学求解手指的各关节角度。
1、https://github.com/DexRobot/dexrobot_kinematics
2、https://github.com/dexsuite/dex-retargeting
3、https://github.com/mimicrobotics/mimic_retargeter_lab
4、


需要更正下你，LEAP Hand 的 link 和 palm 的实物近似为 box ,因为它的实物 本身就是个立方体，所以 MJCF的collision 为box ,但是 wuji 和 tj hand的link 不是立方体，所以 用 box 来建立 collision 几何是否有问题？
另外，原来作者只有 tj hand 的 URDF 文件，这份 MJCF 文件是我写的（可能很多地方都有问题），不具有参考价值。
帮我检查下，然后生成新的  tj hand 的 MJCF 文件。
为什么我已经有各个link的 .STL 文件，确不能用mesh来做碰撞几何，为什么要用 box？用 box 会不会和实物手不匹配？
告诉我  4 处 Python 正则 在哪修改，如何修改？
我的collect_hand_cube_grasp_cache_tj.py 文件似乎有些问题，它只会保存更新 cube 的位姿和大小，但是不会更新手的 joint pose ,这样对吗？如果不对的话该如何修改？
另外，xml中的 armature 以及  tj hand 的 rl_cfg.py 参数该如何设定才能让训练效果更好？其他还需要注意哪里，才会让整个任务完成度更高，效果更好？
再检查下我的奖励函数的各个项，以及各个项之间的系数设定合理吗？
我的 TJ_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.08)，CUBE_SPAWN_POS = (-0.006, 0.043, 0.08)，cube_size 为 0.0272(半边长)，为什么不会导致立方体穿透灵巧手的手掌面？TJ_HOME_ROT: tuple[float, float, float, float] = (0.707107, -0.707107, 0.0, 0.0)灵巧手先绕x轴旋转90度后掌面朝上，(0.0, 0.0, 0.08) 应该是腕部的位置，那掌面的高度应该也是 0.08,cube的中心高度也为0.08,这样不会导致立方体穿过手掌吗，可以是我可视化 uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --num-envs 1 --agent zero 却没有穿过，立方体确实落在手掌上，这是什么原因？
我的电脑能跑动 env.scene.num-envs 4096 吗？
你前面提到的奖励函数的两个问题和缺失项我该如何修改？
那我手掌的实际高度在哪呢？
我可以直接拿 scripts/collect_hand_cube_grasp_cache_tj.py 来生成 tj hand 的cache 文件吗？还需要修改下其他地方了吗？
我已经通过 uv run python scripts/collect_hand_cube_grasp_cache_tj.py   Mjlab-TJ-Right-HandCube-Rotate 生成了 cache 文件，接下来需要训练了，在训练前检查下 tj hand cube rotate 这个项目代码是否有不合理的地方，如何训练？


把 reward 里负 yaw_rate 乘个 1.5~2 倍系数，或把 clip_min 收窄到 −0.1——让“打滑倒退”变得比“不转”更亏。 这个代码在哪修改？如何修改?ingertip_contact 权重调高在哪调？如何调？
帮我把  奖励不对称修正 写进代码
如果我在cube_size=0.272,grasp_cache_file = "tasks/hand_cube/cache/tj_right_grasp_cache.npz" 下训练了一个策略，后面我将cube_size 改为了0.024，换了个初始灵巧手姿态，grasp_cache_file = None再来运行这个策略：uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt，会出现什么样的结果？能这样操作吗？参数已经训练好了换一个立方体和初始姿态可以吗？
改动了cube_size=0.022 和 奖励不对称修正 后重新训练了，运行  uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/ 查看到立方体转动后会朝着大拇指、食指和中指那块，然后被那几根手指卡住动不了，立方体朝着食指那方向的掌面，其他掌面地方没有和立方体接触了，而且几根手指都太卷曲了，没有伸展开。检查下什么原因？为什么LEAP hand 的cube rotate 任务效果可以这么好？参考下 LEAP Hand 的一些参数：https://github.com/Msornerrrr/in-hand-rotation-mjlab/tree/main 我需要调整哪些参数？
如果我拿 grasp_cache_file = "tasks/hand_cube/cache/tj_right_grasp_cache.npz" 去训练，由于立方体的尺寸是一个范围，如果最终的效果比较好的 checkpoint 文件是 model_1200.pt，那我怎么知道这个文件取的 cube 的尺寸是多少？
Curriculum 在训练过程中参数也会更新吗？ Curriculum 是什么？和损失函数之间有什么区别？
CUBE_SPAWN_POS 和 TJ_HOME_POS 位置没搞懂
rotate_finite_diff 到底如何计算的？它的单位是什么？是 rad/s 还是 rad？这个 hand cube rotate 任务的评价指标似乎不涉及立方体到底转了多少，是不是持续地在旋转，而是旋转的速度，是这样的吗？我希望真真切切看到立方体转起来了，不管速度快慢，至少在 圈数的弧度上得看到旋转，旋转的圈数得增加，为什么之前那个奖励和 curriculum 对 LEAP Hand 效果这么好，到我们这效果却那么差到底差在哪了？是手的结构？手的 home joint pose？立方体的尺寸？还是奖励函数或者 curriculm 或者 域随机化 DR 的差别？


训练过程中我需要学习下这个项目，从这个项目涉及到的强化学习基本知识，再到这个项目中一些算法知识，再到这个项目中的一些部署和优化知识，给我介绍一下。
接下来如果我想 sim-to-real 实物上，该从哪里入手，需要了解哪里的代码？实物部署 tj hand 的话是参考代码中的 real_world/README.md 还是参考PIPELINE.md?这两个有什么区别？

2000轮效果不错，接下来是不是该50000个 iteration 继续训练或者增大哪个训练参数让效果更好 ？
需要添加噪声或者域随机化的注释给放开？
当前有加什么域随机化或者噪声吗？

如果到 2000 iter rot_prog 还停在 0.08 以下，那时的问题就不是“会不会转”而是“转不快”，可能的方向是提高 target_yaw_rate 的梯度效率(比如把 clip_max 提到 0.3)或检查 target 0.20 rad/s 对 TJ 手是否过于激进——但那是 1500 iter 之后的事，现在让子弹飞。

learning_time很低，一般都是在0.028附近振荡，collection_time 很高，在 10 附近振荡，mean_reward 最后在 4 附近振荡


撤 boost(rewards.py 参数回 1.0 或删)，保留 clip_max 0.35~0.5;
课程棘轮化(curriculums.py);
drift 死区 0.005→0.003;

可以考虑下 考虑换主奖励形式：用“累计正确旋转角度”(episode 内 CCW 累计弧度，倒转扣分)替代“瞬时角速度”——它天然不可钻(想拿分必须真转)，代价是需要防“甩 cube 刷角度”(用现有的 drift gate + linvel 罚压制)。这个怎么改

手内旋转任务对立方体更简单还是对圆柱体更简单？


待完成：
先 retargetting得到像人的动作，再由强化或模仿学习符合规范的动作

灵巧手的 kp 和 kd 如何计算？







