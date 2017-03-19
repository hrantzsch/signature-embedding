"""This script serves as a simple example and test for the DNN"""
import configparser
import sys

from chainer.datasets import ImageDataset


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
    data_test = ImageDataset(test_index)


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        usage()
        exit(1)
    setup(sys.argv[1])
