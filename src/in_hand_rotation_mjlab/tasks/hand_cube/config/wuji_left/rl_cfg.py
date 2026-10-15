from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def wuji_left_hand_cube_rotate_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=True,
            init_noise_std=0.7,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=False,
            init_noise_std=0.7,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=2.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.00,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="wuji_left_hand_cube_rotate",
        save_interval=100,
        num_steps_per_env=16,
        max_iterations=5_000,
        clip_actions=1.0,
    )


wuji_left_hand_cube_rotate_rl_cfg = wuji_left_hand_cube_rotate_ppo_cfg