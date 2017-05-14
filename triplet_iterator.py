from __future__ import division

import numpy as np
import queue
import threading

from chainer.dataset.iterator import Iterator


def queue_worker(index_queue, batch_queue, dataset, xp):
    while True:
        batch_begin, batch_end = index_queue.get()
        batches = xp.array(dataset[batch_begin:batch_end])

        batch_anc = batches[xp.arange(0, len(batches), 3)]
        batch_pos = batches[xp.arange(1, len(batches), 3)]
        batch_neg = batches[xp.arange(2, len(batches), 3)]

        batch_queue.put((batch_anc, batch_pos, batch_neg))


class TripletIterator(Iterator):
    def __init__(self, dataset, batch_size, repeat=False, xp=np):
        self.dataset = dataset
        self.len_data = len(dataset)
        self.batch_size = batch_size
        self.repeat = repeat
        self.xp = xp

        self.indices = queue.Queue()
        self.batches = queue.Queue(maxsize=6)

        self.current_position = 0
        self.epoch = 0
        self.fill_queue()

        self.queue_worker = threading.Thread(target=queue_worker, kwargs={
            "index_queue": self.indices,
            "batch_queue": self.batches,
            "dataset": self.dataset,
            "xp": self.xp
        })
        self.queue_worker.start()

    def fill_queue(self):
        for i in range(0, self.len_data, 3*self.batch_size):
            i_end = i + 3 * self.batch_size
            self.indices.put((i, i_end))

    def __next__(self):
        if self.indices.empty() and self.batches.empty():
            self.current_position = 0
            self.epoch += 1
            self.fill_queue()

            if not self.repeat:
                raise StopIteration

        # simulate progress for ProgressBar extension
        self.current_position += 3 * self.batch_size
        return self.batches.get(timeout=2)

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.len_data

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
