"""
Regression tests help you track errors across model versions. Suppose 
model version A performs poorly on a small set of examples. You, as a
ML engineer, labor to patch this performance in version B. Suppose 
another member on your team releases version C. We want to make sure 
that version C did not reintroduce the same errors from version A.

The code in this file is split into two purposes. The first purpose is 
give a trained model, find which examples in the dev set it has the 
worst performance on; we save a random subset of these examples as the 
regression test. The second purpose is to implement a regression test
class that we can incorporate into our flow pipeline.
"""
from os.path import join
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.tests.base import BaseTest
from src.system import DigitClassifierSystem, MNISTDataModule


@torch.no_grad()
def build_regression_test(system, loader):
  """Given a trained PyTorch Lightning system, this function will 
  identify the elements in the dev set where the system 'got it the 
  most wrong'. We measure the 'incorrectness' using the loss function. 
  The examples with the highest loss represent those where the model's
  beliefs least match true label.

  Your task is to return the top 100 dev examples with highest loss. 

  Arguments
  ---------
  system (pytorch_lightning.Module): a PyTorch Lightning system. 
    See `system.py` for the `DigitClassifierSystem` implementation.
  loader (torch.utils.data.DataLoader): loader for return minibatches
    of MNIST data. See `MNISTDataModule`.

  Returns
  -------
  images (torch.FloatTensor): stacked image tensors 
    shape: 100 x 1 x 28 x 28
  labels (torch.LongTensor): tensor of labels
    shape: 100
  """
  losses = []
  is_correct = []

  pbar = tqdm(total=len(loader), leave=True, position=0)
  for batch in loader:
      labels = batch[1]  # these are the true labels!
      # these are unnormalized probabilities the model predicts
      logits = system.predict_step(batch[0])
      # the actual prediction is the argmax of the logits
      preds = torch.argmax(logits, dim=1)
      
      # calculate the correctness of each example in the batch
      batch_is_correct = (preds == labels).detach().cpu().numpy()
      is_correct.extend(batch_is_correct.tolist())
      
      # compute the loss for the batch
      loss = F.cross_entropy(logits, labels)
      losses.append(loss.item())
      
      pbar.update(1)

  pbar.close()

  losses = np.array(losses)
  is_correct = np.array(is_correct)

  losses_incorrect = losses[is_correct == 0]
  indices = np.argsort(losses_incorrect)[::-1][:100]

  images = []
  labels = []
  # Use `loader.dataset.__getitem__` to fetch the images and labels for 
  # each index in `indices`.
  for index in indices:
    image, label = loader.dataset.__getitem__(index)
    images.append(image)
    labels.append(label)
  images = torch.stack(images)
  labels = torch.LongTensor(labels)

  return images, labels


class MNISTRegressionTest(BaseTest):
  """A regression test includes a set of 100 known examples that 
  previous models struggled with. For this project, we include  
  examples that a linear regression model on MNIST misclassified. 
  """
  def __init__(self):
    super().__init__()

    test_dir = join(self.root, 'regression')
    data = torch.load(join(test_dir, 'test-data.pt'))
    images = data['images']
    labels = data['labels']

    dataset =  TensorDataset(images, labels)
    assert len(dataset) == 100, "Unexpected dataset size."
    self.dataset = dataset

  def get_dataloader(self, batch_size = 10):
    loader = DataLoader(self.dataset, batch_size=batch_size)
    return loader

  def test(self, trainer, system):
    loader = self.get_dataloader()
    # Pass the dataloader to the trainer and call `test`.
    # Our solution is one line of code
    trainer.test(system, dataloaders = loader)


if __name__ == "__main__":
  # Given the checkpoint file for a trained system, build a regression
  # file and save it to disk.
  import os
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('ckpt', type=str, help='path to trained checkpoint file')
  args = parser.parse_args()

  system = DigitClassifierSystem.load_from_checkpoint(args.ckpt)
  dm = MNISTDataModule(system.config)
  loader = dm.val_dataloader()

  images, labels = build_regression_test(system, loader)

  cur_dir = os.path.dirname(__file__)
  save_path = join(cur_dir, 'images/regression/test-data.pt')
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  torch.save({'images': images, 'labels': labels}, save_path)

  print(f'Saved to {save_path}.')
