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

import os
import datetime
import shutil


# Network definition
class discriminator(chainer.Chain):

    def __init__(self, n_units, n_canvas):
        super(discriminator, self).__init__(
            fc1=L.Linear(784, 1024, wscale=0.002, bias=0),
            bn1=L.BatchNormalization(1024),
            fc2=L.Linear(1024, 1024, wscale=0.002, bias=0),
            bn2=L.BatchNormalization(1024),
            fc3=L.Linear(1024, 1024, wscale=0.002, bias=0),
            bn3=L.BatchNormalization(1024),
            fc4=L.Linear(1024, 1024, wscale=0.002, bias=0),
            bn4=L.BatchNormalization(1024),
            fc5=L.Linear(1024, n_canvas * n_canvas, wscale=0.002, bias=0),
        )
        self.n_canvas = n_canvas
        self.n_saved = 1

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(F.dropout(self.bn1(self.fc1(x)), train=train))
        h2 = F.leaky_relu(F.dropout(self.bn2(self.fc2(h1)), train=train))
        h3 = F.leaky_relu(F.dropout(self.bn3(self.fc3(h2)), train=train))
        h4 = F.leaky_relu(F.dropout(self.bn4(self.fc4(h3)), train=train))
        y = self.fc5(h4)
        if train:
            return y
        else:
            return cuda.to_cpu(y.data)

    def encode(self, x):
        h1 = F.leaky_relu(F.dropout(self.bn1(self.fc1(x)), train=False))
        h2 = F.leaky_relu(F.dropout(self.bn2(self.fc2(h1)), train=False))
        h3 = F.leaky_relu(F.dropout(self.bn3(self.fc3(h2)), train=False))
        h4 = F.leaky_relu(F.dropout(self.bn4(self.fc4(h3)), train=False))
        return h4

    def save_images(self, n_image, result, test):
        pic = np.zeros((n_image * self.n_canvas, n_image * self.n_canvas))
        test = chainer.Variable(test)
        images = np.squeeze(self.__call__(test, train=False))
        images = images.reshape((-1, self.n_canvas, self.n_canvas))
        for i in six.moves.range(n_image):
            for j in six.moves.range(n_image):
                pic[self.n_canvas * i:self.n_canvas * i + self.n_canvas,
                    self.n_canvas * j:self.n_canvas * j + self.n_canvas] = \
                    images[i * n_image + j]
        plt.imshow(pic, vmin=-1, vmax=1, interpolation='none')
        plt.gray()
        plt.savefig('{}/encode_decode_{}.png'.format(result, self.n_saved))
        self.n_saved += 1


class generator(chainer.Chain):

    def __init__(self, n_z, n_units, n_canvas, batchsize, xp):
        super(generator, self).__init__(
            fc1=L.Linear(n_z, 3200, wscale=0.02, bias=0),
            bn1=L.BatchNormalization(3200),
            fc2=L.Linear(3200, 3200, wscale=0.02, bias=0),
            bn2=L.BatchNormalization(3200),
            fc3=L.Linear(3200, 3200, wscale=0.02, bias=0),
            bn3=L.BatchNormalization(3200),
            fc4=L.Linear(3200, 3200, wscale=0.02, bias=0),
            bn4=L.BatchNormalization(3200),
            fc5=L.Linear(3200, n_canvas * n_canvas, wscale=0.02, bias=0),
        )
        self.n_canvas = n_canvas
        self.batchsize = batchsize
        self.n_z = n_z
        self._xp = xp
        self.n_saved = 1

    def __call__(self, train=True, n_image=None):
        if train:
            z = chainer.Variable(
                self._xp.random.uniform(-1, 1, (self.batchsize,
                                                self.n_z)).astype(np.float32))
        else:
            z = chainer.Variable(
                self._xp.random.uniform(-1, 1, (n_image * n_image,
                                                self.n_z)).astype(np.float32))
        h1 = F.relu(F.dropout(self.bn1(self.fc1(z)), train=train))
        h2 = F.relu(F.dropout(self.bn2(self.fc2(h1)), train=train))
        h3 = F.relu(F.dropout(self.bn3(self.fc3(h2)), train=train))
        h4 = F.relu(F.dropout(self.bn4(self.fc4(h3)), train=train))
        y = self.fc5(h4)
        if train:
            return y
        else:
            return cuda.to_cpu(y.data)

    def save_images(self, n_image, result):
        pic = np.zeros((n_image * self.n_canvas, n_image * self.n_canvas))
        images = np.squeeze(self.__call__(False, n_image))
        images = images.reshape((-1, self.n_canvas, self.n_canvas))
        for i in six.moves.range(n_image):
            for j in six.moves.range(n_image):
                pic[self.n_canvas * i:self.n_canvas * i + self.n_canvas,
                    self.n_canvas * j:self.n_canvas * j + self.n_canvas] = \
                    images[i * n_image + j]
        plt.imshow(pic, vmin=-1, vmax=1, interpolation='none')
        plt.gray()
        plt.savefig('{}/generated_{}.png'.format(result, self.n_saved))
        self.n_saved += 1


class gan_updater(training.StandardUpdater):

    def __init__(self, iterator, discriminator, generator,
                 optimizer_d, optimizer_g, margin,
                 device, batchsize,
                 converter=convert.concat_examples, pt=True):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.discriminator = discriminator
        self.generator = generator
        self._optimizers = {'discriminator': optimizer_d,
                            'generator': optimizer_g}
        self.margin = margin
        self.pt = pt
        self.device = device
        self.converter = converter
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        in_var = chainer.Variable(in_arrays)
        generated = self.generator()

        y_data = self.discriminator(in_var)
        loss_dis = F.mean_squared_error(y_data, in_var)

        y_generated = self.discriminator(generated)
        loss_gen = F.mean_squared_error(y_generated, generated)
        loss_dis += F.relu(self.margin - loss_gen)

        if self.pt:
            s = self.discriminator.encode(generated)
            normalized_s = F.normalize(s)
            cosine_similarity = F.matmul(normalized_s, normalized_s,
                                         transb=True)
            ptterm = F.sum(cosine_similarity)
            ptterm /= s.shape[0] * s.shape[0]
            loss_gen += 0.1 * ptterm

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
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--canvas', '-c', type=int, default=28,
                        help='Size of canvas')
    parser.add_argument('--dimension', '-d', type=int, default=3,
                        help='Dimension of rand')
    parser.add_argument('--image', '-i', type=int, default=5,
                        help='Number of output images')
    parser.add_argument('--margin', '-m', type=int, default=10,
                        help='Margin of loss function')
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
    dis = discriminator(args.unit, args.canvas)
    gen = generator(args.dimension, args.unit, args.canvas, args.batchsize, xp)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        dis.to_gpu()  # Copy the model to the GPU
        gen.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizers = {'dis': chainer.optimizers.Adam(alpha=1e-3, beta1=0.5),
                  'gen': chainer.optimizers.Adam(alpha=1e-3, beta1=0.5)}
    optimizers['dis'].setup(dis)
    optimizers['gen'].setup(gen)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(False, 1)
    train *= 2
    train -= 1
    test *= 2
    test -= 1
    test = xp.asarray(test[:args.image * args.image])
    data_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # Set up a trainer
    updater = gan_updater(data_iter, dis, gen,
                          optimizers['dis'], optimizers['gen'], args.margin,
                          args.gpu, args.batchsize)
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
    def generate_images(trainer):
        gen.save_images(args.image, result)

    @training.make_extension(trigger=(1, 'epoch'))
    def encode_decode_images(trainer):
        dis.save_images(args.image, result, test)

    trainer.extend(generate_images)
    trainer.extend(encode_decode_images)
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()