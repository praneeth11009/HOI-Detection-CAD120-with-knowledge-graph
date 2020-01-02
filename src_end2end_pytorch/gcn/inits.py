import tensorflow as tf
import torch
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# ##################################################################



def uniform2(shape, scale=0.05):
    """Uniform init."""
    initial = (-1.000 * scale) + (2.000 * scale * torch.rand(shape))
    initial = initial.to(torch.float32)
    initial.requires_grad = True
    return initial


def glorot2(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = (-1.000 * init_range) + (2.000 * init_range * torch.rand(shape))
    initial = initial.to(torch.float32)
    initial.requires_grad = True
    return initial

def zeros2(shape):
    """All zeros."""
    initial = torch.zeros(shape)
    initial = initial.to(torch.float32)
    initial.requires_grad = True


def ones2(shape):
    """All ones."""
    initial = torch.ones(shape)
    initial = initial.to(torch.float32)
    initial.requires_grad = True