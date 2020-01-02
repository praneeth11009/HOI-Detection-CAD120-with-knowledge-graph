from inits import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.reuse = kwargs.get('reuse')
        self.vars = {}        
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs
    
    def __call__(self, inputs):        
        outputs = self._call(inputs)

        return outputs
    
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.5,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, name='gcn_layer', **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = 0.5
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.name = name

        # helper variable for sparse dropout and initialize w, b
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.variable_scope(self.name) as scope:
            
            if self.reuse:
                scope.reuse_variables()
                
            for i in range(len(self.support)):                  
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim]) # initialize one matrix
            if self.bias:
                self.vars['bias'] = zeros([output_dim])

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)): 
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

def fc(input, output_shape, activation_fn=tf.nn.relu, name='fc', reuse=False):
    with tf.variable_scope(name) as scope:
        
        if reuse:
            scope.reuse_variables()

        output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn, scope=name)

    return output


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., act=tf.nn.relu, bias=False, name='fc_layer', **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.name = name
        
        if dropout:
            self.dropout = 0.5
        else:
            self.dropout = 0.

        self.act = act
        self.bias = bias

        with tf.variable_scope(self.name) as scope:
            if self.reuse:
                scope.reuse_variables()
            
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)