import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.models.registry import get_model as model_fn
from src.utils.logger import get_logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)


def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer


def per_layer_time(model):
  inputs = torch.randn(32, 3, 32, 32)
  o = model(inputs)
  count = 0
  times = []
  for i, layer in enumerate(model.modules()):
    if hasattr(layer, 'needs_drop'):
      count += 1
      print('Layer {} | {}'.format(count, layer.exec_time))
      times.append(layer.exec_time)

  return times


def save_model(net,
               optimizer,
               directory,
               filename):
  logger.info('Saving..')
  state = {
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  if not os.path.isdir(directory):
    os.makedirs(directory)

  torch.save(state, '{}/{}.t7'.format(directory, filename))


def load_model(net,
               path,
               name):
  try:
    logger.info('Loading saved model..')
    prev_model = '{}/{}.t7'.format(path, name)
    checkpoint = torch.load(prev_model)
    net.load_state_dict(checkpoint['net'])
  except FileNotFoundError:
    logger.error('Sparse model not found')
    raise FileNotFoundError

  return net


class LrScheduler:
  def __init__(self, args, optimizer):
    self.args = args
    self.steps = args.steps
    self.init_lr = args.lr
    if args.lr_schedule == 'cyclic':
      self.cyclic_lr = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0, max_lr=args.lr,
        step_size_up=args.up_step, step_size_down=args.down_step)

  def linear_schedule(self, step):
    t = (step) / (self.steps)
    if t <= 0.5:
      factor = 1.0
    elif t <= 0.9:
      factor = 1.0 - (1.0 - 0.01) * (t - 0.5) / 0.4
    else:
      factor = 0.01
    return self.init_lr * factor

  def step(self, optimizer, step):
    if self.args.lr_schedule == 'linear':
      lr = self.linear_schedule(step)
      set_lr(optimizer, lr)
    elif self.args.lr_schedule == 'cyclic':
      self.cyclic_lr.step()
    else:
      raise NotImplementedError('Only use cyclic, linear, step')
    return optimizer


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def get_model(args):
  model = model_fn(args)
  model = model.to(device)
  params = []
  for n, m in model.named_parameters():
    if 'mask' not in n:
      params.append(m)

  if args.optim == 'sgd':
    optimizer = optim.SGD(
      params,
      lr=args.lr,
      momentum=0.9,
      weight_decay=5e-4,
      nesterov=True
    )
  elif args.optim == 'adam':
    optimizer = optim.Adam(
      params,
      lr=args.lr)

    # if args.lr_schedule == 'step':
    #   lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     args.milestones
    #   )
    # elif args.lr_schedule == 'cyclic':
    #   lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer,
    #     base_lr=0,
    #     max_lr=args.lr,
    #     step_size_up=args.up_step,
    #     step_size_down=args.down_step)

  criterion = nn.CrossEntropyLoss()
  if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

  return model, criterion, optimizer


# Mask sanity check
def mask_check(net):
  all_ones = False
  zeros_and_ones = False
  mixed_mask = False
  for module in net.modules():
    if hasattr(module, 'mask'):
      mask = module.mask.detach().cpu().numpy()
      # mask = mask.tolist()
      if set(mask.flatten()) == {1}:
        all_ones = True
      elif set(mask.flatten()) == {1, 0}:
        zeros_and_ones = True
      else:
        mixed_mask = True
  if all_ones:
    logger.warning('Mask is all 1s')
  elif zeros_and_ones:
    logger.info('Mask is 0s and 1s')
  elif mixed_mask:
    logger.warning('Mask is not binary')
