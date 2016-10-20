# coding: UTF-8
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import training
from chainer import reporter
from chainer import cuda
from chainer.training import extensions
import six

import numpy as np
import glob
import re
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import datetime
import os
import shutil


# Network definition
class discriminator(chainer.Chain):

    def __init__(self, n_units, n_canvas):
        super(discriminator, self).__init__(
            l1=L.Linear(784, n_units),
            l2=L.Linear(n_units, n_units*5),
            l3=L.Linear(n_units, n_units*5),
            l4=L.Linear(n_units, 1)
        )

    def __call__(self, x, train=True):
        h1 = F.dropout(F.relu(self.l1(x)), train=train)
        h2 = F.dropout(F.maxout(self.l2(h1), 5), train=train)
        h3 = F.dropout(F.maxout(self.l3(h2), 5), train=train)
        return self.l4(h3)


class generator(chainer.Chain):

    def __init__(self, n_z, n_units, n_canvas, batchsize, xp, sigmma):
        super(generator, self).__init__(
            l1=L.Linear(n_z, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_units),
            l4=L.Linear(n_units, n_canvas * n_canvas)
        )
        self.n_canvas = n_canvas
        self.batchsize = batchsize
        self.n_z = n_z
        self._xp = xp
        self.n_saved = 1
        self.sigmma = sigmma

    def __call__(self, train=True, n_image=None):
        if n_image:
            z = chainer.Variable(
                self._xp.random.normal(0, self.sigmma,
                                       (n_image * n_image, self.n_z)).astype(np.float32))

        else:
            z = chainer.Variable(
                self._xp.random.normal(0, self.sigmma, (self.batchsize,
                                                self.n_z)).astype(np.float32))
        h1 = F.dropout(F.relu(self.l1(z)), train=train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=train)
        h3 = F.dropout(F.relu(self.l3(h2)), train=train)
        y = F.relu(self.l4(h3))
        if n_image:
            return cuda.to_cpu(y.data)
        else:
            return y

    def save_images(self, n_image, result):
        pic = np.zeros((n_image * self.n_canvas, n_image * self.n_canvas))
        images = np.squeeze(self.__call__(False, n_image)).reshape((-1, self.n_canvas, self.n_canvas))
        for i in six.moves.range(n_image):
            for j in six.moves.range(n_image):
                pic[self.n_canvas * i:self.n_canvas * i + self.n_canvas,
                    self.n_canvas * j:self.n_canvas * j + self.n_canvas] = \
                    images[i * n_image + j]
        plt.imshow(pic, vmin=0, vmax=1, interpolation='none')
        plt.gray()
        plt.savefig('{}/image_{}.png'.format(result, self.n_saved))
        self.n_saved += 1


class gan_updater(training.StandardUpdater):

    def __init__(self, iterator, discriminator, generator,
                 optimizer_d, optimizer_g, device,
                 batchsize, xp, converter=convert.concat_examples):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.discriminator = discriminator
        self.generator = generator
        self._optimizers = {'discriminator': optimizer_d,
                            'generator': optimizer_g}
        self.device = device
        self.converter = converter
        self.iteration = 0
        self._xp = xp
        self.ones = chainer.Variable(self._xp.ones((batchsize, 1),
                                                   dtype=np.int32))
        self.zeros = chainer.Variable(self._xp.zeros((batchsize, 1),
                                                     dtype=np.int32))

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        in_var = chainer.Variable(in_arrays)
        generated = self.generator(False)

        label_data = self.discriminator(in_var)
        loss_dis = F.sigmoid_cross_entropy(label_data, self.zeros)

        label_generated = self.discriminator(generated)
        loss_dis += F.sigmoid_cross_entropy(label_generated, self.ones)
        loss_gen = F.sigmoid_cross_entropy(label_generated, self.zeros)

        for optimizer in self._optimizers:
            self._optimizers[optimizer].target.cleargrads()

        reporter.report({'dis/loss': loss_dis})
        reporter.report({'gen/loss': loss_gen})

        loss_dis.backward()
        self._optimizers['discriminator'].update()

        loss_gen.backward()
        self._optimizers['generator'].update()


def main():
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unitd', '-ud', type=int, default=240,
                        help='Number of units of discriminator')
    parser.add_argument('--unitg', '-ug', type=int, default=1200,
                        help='Number of units of generator')
    parser.add_argument('--canvas', '-c', type=int, default=28,
                        help='Size of canvas')
    parser.add_argument('--dimension', '-d', type=int, default=100,
                        help='Dimension of rand')
    parser.add_argument('--image', '-i', type=int, default=5,
                        help='Number of output images')
    args = parser.parse_args()
    xp = cuda.cupy if args.gpu >= 0 else np
    result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    os.mkdir(result)
    shutil.copy(__file__, result + '/' + __file__)

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(False, 3)
    # 0 to 1
    dis = discriminator(args.unitd, args.canvas)
    gen = generator(args.dimension, args.unitg, args.canvas,
                    args.batchsize, xp, np.std(train))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        dis.to_gpu()  # Copy the model to the GPU
        gen.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizers = {'dis': chainer.optimizers.Adam(),
                  'gen': chainer.optimizers.Adam()}
    optimizers['dis'].setup(dis)
    optimizers['gen'].setup(gen)

    data_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # Set up a trainer
    updater = gan_updater(data_iter, dis, gen,
                          optimizers['dis'], optimizers['gen'],
                          args.gpu, args.batchsize, xp)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('dis/loss'))
    trainer.extend(extensions.dump_graph('gen/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'dis/loss', 'gen/loss']))

    @training.make_extension(trigger=(1, 'epoch'))
    def save_images(trainer):
        gen.save_images(args.image, result)

    trainer.extend(save_images)
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
