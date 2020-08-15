from __future__ import print_function, unicode_literals

from config import load_config
from CelebA_AlexNet import AlexNet

import os
import numpy as np
import chainer
from chainer import training, datasets, dataset
from chainer.training import extensions


class PreprocessedDataset(dataset.DatasetMixin):
    def __init__(self, img_list, root):
        self.base = datasets.LabeledImageDataset(img_list, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        # Scale to [0, 1]
        image *= (1.0/255.0)
        return image, label


def main():
    config = load_config('Default')
    model = AlexNet(config)

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
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time'
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=100))

    trainer.run()


if __name__ == '__main__':
    main()
