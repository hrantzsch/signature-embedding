"""This script serves as a simple example and test for the DNN"""
import configparser
import sys

from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, report, training
from chainer.datasets import ImageDataset
from chainer.training import extensions

import triplet
from triplet_iterator import TripletIterator


class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),    # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x_a, x_p, x_n):
        y_a, y_p, y_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        loss = F.triplet(y_a, y_p, y_n)
        report({'loss': loss}, self)
        return loss


def usage():
    print("python3 {} <config_file.ini>".format(sys.argv[0]))


def get_trainer(updater, evaluator, epochs):
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='result')
    trainer.extend(evaluator)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(
        (epochs, 'epoch'), update_interval=10))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    return trainer


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        usage()
        exit(1)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    batch_size = int(config['TRAINING']['batch_size'])
    epochs = int(config['TRAINING']['epochs'])
    train_index = config['DATA']['train']
    test_index = config['DATA']['test']

    train_iter = TripletIterator(ImageDataset(train_index),
                                 batch_size=batch_size,
                                 repeat=True)
    test_iter = TripletIterator(ImageDataset(test_index),
                                batch_size=batch_size)

    model = Classifier(MLP(100, 10))

    optimizer = optimizers.SGD(lr=0.000001)
    optimizer.setup(model)
    updater = triplet.Updater(train_iter, optimizer)

    evaluator = triplet.Evaluator(test_iter, model)

    trainer = get_trainer(updater, evaluator, epochs)
    trainer.run()
