"""
This script is used to shuffle input images to an order that can easily be
accessed using chainer.datasets.ImageDataset.
"""

import os
import random
import sys


def get_triplets(source_dir):
    paths = os.listdir(source_dir)
    for (path, _, files) in os.walk(source_dir):
        for anc in files:
            pos = anc
            while (pos == anc):
                pos = random.choice(files)
            neg_path = path
            while (neg_path == path):
                neg_path = os.path.join(source_dir, random.choice(paths))
            neg = random.choice(os.listdir(neg_path))
            yield(os.path.join(path, anc))
            yield(os.path.join(path, pos))
            yield(os.path.join(neg_path, neg))


def print_usage():
    print("usage:\tpython3 {} input_data [output_file]\n"
          "e.g.\tpython3 {} /home/data/mnist/training triplets.txt"
          .format(sys.argv[0], sys.argv[0]))


if __name__ == '__main__':
    if not len(sys.argv) >= 2:
        print_usage()
        exit(1)
    triplets = get_triplets(sys.argv[1])

    if len(sys.argv) == 3:
        with open(sys.argv[2], 'w+') as out:
            for t in triplets:
                out.write("{}\n".format(t))
    else:
        for t in triplets:
            print(t)
