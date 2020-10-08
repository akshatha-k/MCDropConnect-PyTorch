import argparse
import os

parser = argparse.ArgumentParser(description='Ensemble Training')
# Model Parameters
parser.add_argument(
  '--model', default='vgg19_bn', type=str, help='Model to use')
parser.add_argument(
  '--run_name', default=None, type=str, help='Name of this run')
parser.add_argument(
  '--model_type', default='dense', type=str, help='Dense/sparse')
parser.add_argument(
  '--seed', default=1, type=int, help='random seed')
parser.add_argument(
  '--mode', default='train', type=str, help='In eval mode or not')
parser.add_argument(
  '--load_model', default=False, type=bool, help='Load prev weights')
parser.add_argument(
  '--resume', default=False, type=bool, help='Resume training')

# Training params
parser.add_argument(
  '--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument(
  '--num_classes', default=10, type=int, help='Number of classes')
parser.add_argument(
  '--optim', default='sgd', type=str, help='Optimizer to use')
parser.add_argument(
  '--lr', default=0.05, type=float, help='Initial LR')
parser.add_argument(
  '--lr_schedule', default='linear', type=str, help='LR scheduler')
parser.add_argument(
  '--lr_cycle', default='full', type=str, help='Full or half cycle')
parser.add_argument(
  '--up_step', default=5000, type=int, help='Cyclic lr step size')
parser.add_argument(
  '--down_step', default=5000, type=int, help='Cyclic lr step size')
parser.add_argument(
  '--milestones', default=[25000, 50000, 75000, 90000], type=list, help='Multi step lr')

parser.add_argument(
  '--epochs', default=200, type=int, help='No of epochs')
parser.add_argument(
  '--clip', default=1, type=int, help='Gradient clipping')
parser.add_argument(
  '--steps', default=None, type=int, help='No of steps')
parser.add_argument(
  '--eval_step', default=1000, type=int, help='Eval every this steps')
parser.add_argument(
  '--batch_size', default=100, type=int, help='Batch size')

# MCDC args
parser.add_argument(
  '--layer_no', default=None, type=int, help='Layer to start MCDC from')
parser.add_argument(
  '--drop_prob', default=0.2, type=float, help='Drop probability')
parser.add_argument(
  '--manifold_mcdc', default=False, type=bool, help='Use manifold mcdc')
parser.add_argument(
  '--target_dataset', default='svhn', type=str, help='Target dataset for ood detection')

parser.add_argument(
  '--output_dir',
  type=str,
  help='Output directory for storing ckpts. Default is in runs/hparams')
parser.add_argument(
  '--use_colab', type=bool, default=False, help='Use Google colaboratory')


def get_args():
  args = parser.parse_args()
  if args.run_name is None:
    raise ValueError('Ernter a run name')
  args.output_dir = '{}/{}/{}/{}'.format(
    args.dataset,
    args.model,
    args.drop_prob,
    args.run_name
  )
  if args.use_colab:
    from google.colab import drive
    drive.mount('/content/gdrive')
    colab_str = '/content/gdrive/My Drive/MCDC/'
    OUTPUT_DIR = '{}/{}'.format(colab_str, args.output_dir)
    if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    args.output_dir = OUTPUT_DIR
  else:
    output_dir = 'runs/{}'.format(args.output_dir)
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    args.output_dir = ('runs/{}'.format(args.output_dir))

  return args
