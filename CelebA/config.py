from __future__ import print_function, unicode_literals


class DefaultConfig():
    layer_param = {
        'conv1': 48,
        'conv2': 128,
        'conv3': 192,
        'conv4': 128,
        'conv5': 128,
        'lin1': 256,
        'lin2': 256,
        'lin3': 2
    }
    gpu_id = 0 # use_gpu = -1 means use CPU
    rootFolder = '/home/liao/datasets/CelebA/img_align_celeba/'
    trainList = 'data/celeba_subset_train.txt'
    validList = 'data/celeba_subset_valid.txt'
    batchSize = 10
    epoch = 20


class MultiLabelConfig(DefaultConfig):
    layer_param = {
        'conv1': 48,
        'conv2': 128,
        'conv3': 192,
        'conv4': 128,
        'conv5': 128,
        'lin1': 256,
        'lin2': 256,
        'lin3': 4
    }
    gpu_id = 0 # use_gpu = -1 means use CPU
    trainList = 'data/celeba_subset_multi_train.txt'
    validList = 'data/celeba_subset_multi_valid.txt'
    batchSize = 10
    epoch = 20
    label1 = (0,2) # Task 1: Index 0 to 1 (0 <= t < 2)
    label2 = (2,4) # Task 2: Index 2 to 3 (2 <= t < 4)
    labels = [label1, label2]


config_dict = {
    'Default': DefaultConfig,
    'MultiLabel': MultiLabelConfig,
}


def load_config(config_name):
    config = config_dict.get(config_name, DefaultConfig)()
    print('Using config:', config.__class__.__name__)
    return config
