from __future__ import division

import numpy as np

from chainer.dataset import iterator


class TripletIterator(iterator.Iterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat

        if shuffle:
            print("warning: shuffling or TripletIterator NYI")

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + 3 * self.batch_size
        N = len(self.dataset)

        batches = np.array(self.dataset[i:i_end])
        batch_anc = batches[np.arange(0, len(batches), 3)]
        batch_pos = batches[np.arange(1, len(batches), 3)]
        batch_neg = batches[np.arange(2, len(batches), 3)]

        self.is_new_epoch = i_end >= N
        if self.is_new_epoch:
            self.current_position = 0
            self.epoch += 1
        else:
            self.current_position = i_end

        return batch_anc, batch_pos, batch_neg

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
