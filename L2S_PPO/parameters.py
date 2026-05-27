import argparse
parser = argparse.ArgumentParser(description='DRL-LSJSP')

# env parameters
parser.add_argument('--j', type=int, default=20)
parser.add_argument('--m', type=int, default=15)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--h', type=int, default=99)
parser.add_argument('--init_type', type=str, default='spt-pdr')
parser.add_argument('--gamma', type=float, default=0.99)
# model parameters
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--embedding_layer', type=int, default=4)
parser.add_argument('--policy_layer', type=int, default=4)
parser.add_argument('--heads', type=int, default=1)  # dghan parameters
parser.add_argument('--drop_out', type=float, default=0.0)  # dghan parameters
# training parameters
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--steps_learn', type=int, default=25)
parser.add_argument('--transit', type=int, default=300)
parser.add_argument('--validation_transit', type=int, default=200)
# transit length used only during periodic validation (kept small to bound the
# 100-instance validation wall-time even when --transit is large)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--episodes', type=int, default=3)
parser.add_argument('--step_validation', type=int, default=10)

# new parameters for cp
parser.add_argument('--training_set_size', type=int, default=25600)
parser.add_argument('--window_size', type=int, default=150)
parser.add_argument('--cp_solver_time', type=float, default=1)
parser.add_argument('--cp_solver_cpu', type=int, default=1)
parser.add_argument('--cp_solver_budget', type=int, default=16)
parser.add_argument('--window_training_log_path', type=str, default=None)
parser.add_argument('--window_validation_log_path', type=str, default=None)

# ===== PPO-specific parameters =====
parser.add_argument('--ppo_epochs', type=int, default=4,
                    help='Number of optimization epochs per PPO update on a rollout chunk.')
parser.add_argument('--ppo_clip', type=float, default=0.2,
                    help='PPO ratio clipping epsilon.')
parser.add_argument('--gae_lambda', type=float, default=0.95,
                    help='GAE lambda for advantage estimation.')
parser.add_argument('--value_coef', type=float, default=0.1,
                    help='Weight on the value-function loss in the PPO total loss.')
parser.add_argument('--entropy_coef', type=float, default=0.001,
                    help='Weight on the entropy bonus in the PPO total loss.')
parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='Maximum L2 norm for gradient clipping (<=0 disables).')
parser.add_argument('--tensorboard_log_dir', type=str, default='./tensorboard_log_ppo',
                    help='Tensorboard log root for PPO runs.')
parser.add_argument('--ppo_minibatch_steps', type=int, default=8,
                    help='Number of steps in each PPO minibatch.')

args = parser.parse_args()
