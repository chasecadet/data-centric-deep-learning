"""This flow will train a neural network to perform sentiment classification 
for the beauty products reviews.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from metaflow import FlowSpec, step, Parameter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from cleanlab.filter import find_label_issues
from sklearn.model_selection import KFold

from src.system import ReviewDataModule, SentimentClassifierSystem
from src.utils import load_config, to_json
from src.consts import DATA_DIR


class TrainIdentifyReview(FlowSpec):
  r"""A MetaFlow that trains a sentiment classifier on reviews of luxury beauty
  products using PyTorch Lightning, identifies data quality issues using CleanLab, 
  and prepares them for review in LabelStudio.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./config.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    config = load_config(self.config_path)

    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.train.ckpt_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.train.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.config = config
    self.next(self.train_test)

  @step
  def train_test(self):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(self.system, self.dm)
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

    # results are saved into the system
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'pre-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.crossval)
  
  @step
  def crossval(self):
    r"""Confidence learning requires cross validation to compute 
    out-of-sample probabilities for every element. Each element
    will appear in a single cross validation split exactly once. 
    """
    # combine training and dev datasets
    X = np.concatenate([
      np.asarray(self.dm.train_dataset.embedding),
      np.asarray(self.dm.dev_dataset.embedding),
      np.asarray(self.dm.test_dataset.embedding),
    ])
    y = np.concatenate([
      np.asarray(self.dm.train_dataset.data.label),
      np.asarray(self.dm.dev_dataset.data.label),
      np.asarray(self.dm.test_dataset.data.label),
    ])
    probs = np.zeros(len(X))  # we will fill this in
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = KFold(n_splits=3)    # create kfold splits
    model = self.system
    #  create a trainerstep 
    trainer = self.trainer
    for train_index, test_index in kf.split(X):
      probs_ = None
      # Get train and test slices 
      X_train, y_train = X[train_index], y[train_index]
      X_test, y_test = X[test_index], y[test_index]
      # Convert those bad boys to tensors 
      X_test_tensor = torch.Tensor(X_test)
      y_test_tensor = torch.Tensor(y_test)
      X_train_tensor = torch.Tensor(X_train) 
      y_train_tensor = torch.Tensor(y_train)      
      # Create train/test datasets using tensors.
      train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
      test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
      # Split up the datasets 
      train_size = int(len(train_dataset) * 0.8)  # 80% of data for training
      val_size = len(train_dataset) - train_size  # remaining 20% for validation
      train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # Create DataLoader instances for training, validation, and testing sets 
      # Set batch size for your data loader
      dm = ReviewDataModule(self.config)
      dm.train_dataset = train_dataset
      dm.dev_dataset = val_dataset
      dm.test_dataset = test_dataset    
      system = SentimentClassifierSystem(self.config)
      trainer = Trainer(
      max_epochs = self.config.train.optimizer.max_epochs)
      trainer.fit(system, dm)
      trainer.test(system, dm, ckpt_path = 'best')
      results = system.test_results
      # Call `predict` on `Trainer` and the test data loader.
      logits = trainer.predict(system,dm.test_dataloader())
      logits = torch.cat(logits)
      # Convert probabilities back to numpy (make sure 1D).
      probs_np = torch.from_numpy(logits.numpy()).float()
      probs_ = probs_np.flatten()
      assert probs_ is not None, "`probs_` is not defined."
      print(f"probs shape: {probs.shape}")
      print(f"probs_np shape: {probs_np.shape}")
      print(f"probs_ shape: {probs_.shape}")
      probs[test_index] = np.squeeze(probs_np, axis=1)
      probs[test_index] = probs_
    # create a single dataframe with all input features
    all_df = pd.concat([
      self.dm.train_dataset.data,
      self.dm.dev_dataset.data,
      self.dm.test_dataset.data,
    ])
    all_df = all_df.reset_index(drop=True)
    # add out-of-sample probabilities to the dataframe
    all_df['prob'] = probs

    # save to excel file
    all_df.to_csv(join(DATA_DIR, 'prob.csv'), index=False)

    self.all_df = all_df
    self.next(self.inspect)

  @step
  def inspect(self):
    r"""Use confidence learning over examples to identify labels that 
    likely have issues with the `cleanlab` tool. """
    all_df=self.all_df
    prob = np.asarray(self.all_df.prob)
    prob = np.stack([1 - prob, prob]).T
    ranked_label_issues = find_label_issues(labels=all_df.label.values,pred_probs=prob,return_indices_ranked_by='self_confidence',)
    assert ranked_label_issues is not None, "`ranked_label_issues` not defined."
    # save this to class
    self.issues = ranked_label_issues
    print(f'{len(ranked_label_issues)} label issues found.')
    # overwrite label for all the entries in all_df
    for index in self.issues:
      label = self.all_df.loc[index, 'label']
      # we FLIP the label!
      self.all_df.loc[index, 'label'] = 1 - label

    self.next(self.review)

  @step
  def review(self):
    r"""Format the data quality issues found such that they are ready to be 
    imported into LabelStudio. We expect the following format:

    [
      {
        "data": {
          "text": <review text>
        },
        "predictions": [
          {
            "value": {
              "choices": [
                  "Positive"
              ]
            },
            "from_name": "sentiment",
            "to_name": "text",
            "type": "choices"
          }
        ]
      }
    ]

    See https://labelstud.io/guide/predictions.html#Import-pre-annotations-for-text.and

    You do not need to complete anything in this function. However, look through the 
    code and make sure the operations and output make sense.
    """
    outputs = []
    for index in self.issues:
      row = self.all_df.iloc[index]
      output = {
        'data': {
          'text': str(row.review),
        },
        'predictions': [{
          'result': [
            {
              'value': {
                'choices': [
                  'Positive' if row.label == 1 else 'Negative'
                ]
              },
              'id': f'data-{index}',
              'from_name': 'sentiment',
              'to_name': 'text',
              'type': 'choices',
            },
          ],
        }],
      }
      outputs.append(output)

    # save to file
    preanno_path = join(self.config.review.save_dir, 'pre-annotations.json')
    to_json(outputs, preanno_path)

    self.next(self.retrain_retest)

  @step
  def retrain_retest(self):
    r"""Retrain without reviewing. Let's assume all the labels that 
    confidence learning suggested to flip are indeed erroneous."""
    dm = ReviewDataModule(self.config)
    train_size = len(dm.train_dataset)
    dev_size = len(dm.dev_dataset)
    all_df=self.all_df
    dm.train_dataset.data = all_df[:train_size]
    dm.dev_dataset.data = all_df[train_size:train_size+dev_size]
    dm.test_dataset.data = all_df[train_size+dev_size:]
    system = SentimentClassifierSystem(self.config)
    trainer = Trainer(
      max_epochs = self.config.train.optimizer.max_epochs)

    trainer.fit(system, dm)
    trainer.test(system, dm, ckpt_path = 'best')
    results = system.test_results

    print(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'post-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python flow_conflearn.py`. To list
  this flow, run `python flow_conflearn.py show`. To execute
  this flow, run `python flow_conflearn.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python flow_conflearn.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python flow_conflearn.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainIdentifyReview()
