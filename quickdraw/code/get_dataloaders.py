from tqdm import tqdm
import torch
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize

from flags import FLAGS
from load_tensor_data import QuickData, QuickTestData
from draw_functions import Sketch, Draw


def get_data(training, test):
    """
    :param training: list of training csv file paths
    :param test: the test csv file path
    :return: the training, validation, and test data loaders
    """

    kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if torch.cuda.is_available() else {}

    # initialize empty lists for the individual csv data sets
    train_data_sets = [None] * FLAGS.num_classes
    val_data_sets = [None] * FLAGS.num_classes

    # initialize weights for uniform sampling during training
    weights = np.zeros(FLAGS.num_classes, dtype=np.float32)

    # define your transform
    transform = Compose([Draw(), ToTensor(), Normalize([0], [255])])

    # populate the lists and weights
    for i in tqdm(range(FLAGS.num_classes)):
        ds = QuickData(training[i], transform)
        weights[i] = ds.__len__() - FLAGS.val_samples_per_class
        train_data_sets[i], val_data_sets[i] = \
            random_split(ds, [ds.__len__() - FLAGS.val_samples_per_class, FLAGS.val_samples_per_class])

    weights = np.sum(weights) / weights
    weights = torch.as_tensor(weights, dtype=torch.float)

    # get the training, validation, and test tensor data set
    train_data = ConcatDataset(train_data_sets)
    val_data = ConcatDataset(val_data_sets)
    test_data = QuickTestData(test, transform)

    # get training, validation, and test data loaders
    train_loader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=FLAGS.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, test_data.__len__(), weights


