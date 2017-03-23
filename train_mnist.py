"""This script serves as a simple example and test for the DNN"""
import configparser
import sys

from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, report, training
from chainer.datasets import ImageDataset

from triplet_iterator import TripletIterator
from triplet_updater import TripletUpdater


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
        print(loss.data)
        report({'loss': loss}, self)
        return loss


def usage():
    print("python3 {} <config_file.ini>".format(sys.argv[0]))


def log(msg):
    print("# {}".format(msg))


def setup(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    test_index = config['DATA']['test']
    train_index = config['DATA']['train']
    log("reading train data from {}".format(train_index))
    log("reading test data from {}".format(test_index))
    log("loading data...")
    data_train = ImageDataset(train_index)
    train_iter = TripletIterator(data_train, batch_size=300, shuffle=False)
    data_test = ImageDataset(test_index)
    test_iter = TripletIterator(data_test, batch_size=300, shuffle=False)

    model = Classifier(MLP(100, 10))
    optimizer = optimizers.SGD(lr=0.000001)
    optimizer.setup(model)

    return TripletUpdater(train_iter, optimizer)


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        usage()
        exit(1)
    updater = setup(sys.argv[1])
    training.Trainer(updater, (20, 'epoch'), out='result').run()
