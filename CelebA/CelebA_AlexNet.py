from __future__ import print_function, unicode_literals

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable

import numpy as np


class AlexNet(Chain):
    def __init__(self, config):
        layer_param = config.layer_param
        self.layer_param = layer_param
        super(AlexNet, self).__init__(
            conv1 = L.Convolution2D(None, layer_param['conv1'], 11, stride=4),
            conv2 = L.Convolution2D(None, layer_param['conv2'], 5, pad=2),
            conv3 = L.Convolution2D(None, layer_param['conv3'], 3, pad=1),
            conv4 = L.Convolution2D(None, layer_param['conv4'], 3, pad=1),
            conv5 = L.Convolution2D(None, layer_param['conv5'], 3, pad=1),
            fc6 = L.Linear(None, layer_param['lin1']),
            fc7 = L.Linear(None, layer_param['lin2']),
            fc8 = L.Linear(None, layer_param['lin3']),
        )
        self.train = True

    def forward(self,x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        return h

    def __call__(self, x, t):
        h = self.forward(x)
        print(h.data)
        print(t.data)
        acc = F.accuracy(h, t)
        print(acc.data)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class AlexNetMultiLabel(AlexNet):
    def __init__(self,config):
        self.config = config
        super(AlexNetMultiLabel, self).__init__(config)

    def split_task(self, h, t):
        tasks = []
        for label in self.config.labels:
            h_split = Variable(h.data[:, slice(*label)])
            t_cpu = chainer.cuda.to_cpu(t.data[:, slice(*label)])
            # Get the index of nonzero element. The index represents class label.
            t_cpu = np.where(t_cpu)[1].astype(np.int32)
            t_split = Variable(chainer.cuda.to_gpu(t_cpu))
            tasks.append((h_split, t_split))
        return tasks

    def __call__(self, x, t):
        h = self.forward(x)
        loss = F.sigmoid_cross_entropy(h, t)
        tasks = self.split_task(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(*tasks[0]), 'accuracy1': F.accuracy(*tasks[1])}, self)
        return loss
