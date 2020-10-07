import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils.logger import get_logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)

EPSILON = 1e-7


def numpy_entropy(probs, axis=-1, eps=1e-6):
  return -np.sum(probs * np.log(probs + eps), axis=axis)


def numpy_classification_nll(y_true, y_pred):
  """
      Negative log-likelihood or negative log-probability loss/metric for classiication.
      Reference: Evaluating Predictive Uncertainty Challenge, Quiñonero-Candela et al, 2006.
      It sums over classes: log(y_pred) for true class and log(1.0 - pred) for not true class, and then takes average across samples.
  """
  y_pred = np.clip(y_pred, EPSILON, 1.0 - EPSILON)

  return -np.mean(np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred), axis=-1), axis=-1)


def accuracy(y_true, y_pred):
  """
      Simple categorical accuracy.
  """
  return np.mean(y_true == y_pred)


def to_categorical(y, num_classes):
  """ 1-hot encodes a tensor """
  return np.eye(num_classes, dtype='uint8')[y]


def mc_predict(model, testloader, num_samples=1):
  output_samples = []
  softmax = torch.nn.Softmax()
  for sample in range(num_samples):
    with torch.no_grad():
      logits = []
      targets = []
      for data, target in testloader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        logits.append(outputs)
        targets.append(target)
      logits = torch.cat(logits)
      targets = torch.cat(targets)
      output_samples.append(logits)
  output_samples = torch.mean(torch.stack(output_samples), 0)
  output_samples = softmax(output_samples)
  return output_samples, targets


def mcdropout_test(model, testloader, args):
  model.train()
  T = 25
  acc = []
  nll = []
  entropy = []
  total = 0
  correct = 0
  for i in range(1, T):
    outputs, targets = mc_predict(model, testloader, i)
    # class_preds = np.argmax(outputs, axis=1)
    # acc.append(accuracy(targets, class_preds))
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    acc.append((correct / total))
    outputs, targets = outputs.cpu().numpy(), targets.cpu().numpy()
    nll.append(numpy_classification_nll(to_categorical(targets, args.num_classes),
                                        outputs))
    entropy.append(np.mean(numpy_entropy(outputs)))
    logger.info("{} samples - MCDC Accuracy {:.3f} MCDC NLL {:.3f} MCDC Entropy {:.3f}".format(i, acc[-1], nll[-1],
                                                                                               entropy[-1]))


def uncertainty_test(model, testloader, args):
  T = 25
  rotation_list = range(0, 180, 10)
  with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
      if i > 10:
        break 
      data, target = data.to(device), target.to(device)
      output_list = []
      image_list = []
      unct_list = []
      for r in rotation_list:
        rotation_matrix = Variable(
          torch.Tensor([[[math.cos(r / 360.0 * 2 * math.pi), -math.sin(r / 360.0 * 2 * math.pi), 0],
                         [math.sin(r / 360.0 * 2 * math.pi), math.cos(r / 360.0 * 2 * math.pi), 0]]]).to(device))
        rotation_matrix = torch.cat(testloader.batch_size * [rotation_matrix])
        grid = F.affine_grid(rotation_matrix, data.size())
        data_rotate = F.grid_sample(data, grid)
        image_list.append(data_rotate)

        for i in range(T):
          output_list.append(torch.unsqueeze(F.softmax(model(data_rotate)).data, 0))
        output_mean = torch.cat(output_list, 0).mean(0)
        output_variance = torch.cat(output_list, 0).var(0).mean().item()
        confidence = output_mean.data.cpu().numpy().max()
        predict = output_mean.data.cpu().numpy().argmax()
        unct_list.append(output_variance)
        # print('rotation degree', str(r).ljust(3),
        #      'Uncertainty : {:.4f} Predict : {} Softmax : {:.2f}'.format(output_variance, predict, confidence))

    plt.figure()
    for i in range(len(rotation_list)):
      ax = plt.subplot(2, len(rotation_list) / 2, i + 1)
      plt.text(0.5, -0.5, "{0:.3f}".format(unct_list[i]),
               size=12, ha="center", transform=ax.transAxes)
      plt.axis('off')
      plt.gca().set_title(str(rotation_list[i]) + u'\xb0')
      plt.imshow(image_list[i][0, 0, :, :].data.cpu().numpy())
    plt.savefig('{}/uncertainty'.format(args.output_dir))
