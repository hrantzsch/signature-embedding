from chainer import variable
from chainer.training.updater import StandardUpdater


class TripletUpdater(StandardUpdater):
    """
    Just a StandardUpdater, with it's update_core function adjusted to work
    with the triplet data I feed it.
    """
    def update_core(self):
        batches = self._iterators['main'].next()
        in_vars = (variable.Variable(self.converter(batch, self.device))
                   for batch in batches)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        optimizer.update(loss_func, *in_vars)
