from __future__ import print_function, unicode_literals

from config import load_config
from CelebA_AlexNet import AlexNetMultiLabel

import os
import chainer
from chainer import training, datasets, dataset
from chainer.training import extensions

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


class PreprocessedDataset(dataset.DatasetMixin):
    def __init__(self, img_list, root):
        # Read
        with open(img_list) as f:
            lines = f.readlines() # convert to list
            lines = [line.split()[0] for line in lines]
            self.images = datasets.ImageDataset(lines, root)

        # Read label
        with open(img_list) as f:
            lines = f.readlines() # convert to list
            lines = [map(int,line.split()[1:]) for line in lines]
            self.labels = MultiLabelBinarizer().fit_transform(lines).astype(np.int32) # MultiLabelBinarizer() gives int64, convert to int32

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        image = self.images[i]
        # Scale to [0, 1]
        image *= (1.0/255.0)
        #print(self.labels[i])
        return image, self.labels[i]


def main():
    config = load_config('MultiLabel')
    model = AlexNetMultiLabel(config)

    if config.gpu_id >= 0:
        chainer.cuda.get_device_from_id(config.gpu_id).use()
        model.to_gpu()
        # Set up PATH for nvcc for PyCharm
        os.environ['PATH'] += ':/usr/local/cuda-8.0/bin'

    model.train = True

    train = PreprocessedDataset(config.trainList,config.rootFolder)
    valid = PreprocessedDataset(config.validList,config.rootFolder)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, config.batchSize)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, config.batchSize, repeat=False)

    # Set up a optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=config.gpu_id)
    trainer = training.Trainer(updater, (config.epoch, 'epoch'))

    # Validation
    trainer.extend(extensions.Evaluator(valid_iter, model, device=config.gpu_id))
    log_interval = (1000), 'iteration'

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
        'main/accuracy1', 'validation/main/accuracy1',
        'elapsed_time'
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=100))

    trainer.run()


if __name__ == '__main__':
    main()
