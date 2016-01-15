import lasagne as nn
import theano
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import ReshapeLayer, ConcatLayer, SliceLayer, LSTMLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

# theano's built-in softmax and binary cross-entropy can be numerically unstable
# so it is necessary to use equivalent log-domain functions
def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)

def build_rnn(conv_input_var, seq_input_var, conv_shape, word_dims, n_hid, lstm_layers):
    ret = {}
    ret['seq_input'] = seq_layer = InputLayer((None, None, word_dims), input_var=seq_input_var)
    batchsize, seqlen, _ = seq_layer.input_var.shape
    ret['seq_resh'] = seq_layer = ReshapeLayer(seq_layer, shape=(-1, word_dims))
    ret['seq_proj'] = seq_layer = DenseLayer(seq_layer, num_units=n_hid)
    ret['seq_resh2'] = seq_layer = ReshapeLayer(seq_layer, shape=(batchsize, seqlen, n_hid))
    ret['conv_input'] = conv_layer = InputLayer(conv_shape, input_var = conv_input_var)
    ret['conv_proj'] = conv_layer = DenseLayer(conv_layer, num_units=n_hid)
    ret['conv_resh'] = conv_layer = ReshapeLayer(conv_layer, shape=([0], 1, -1))
    ret['input_concat'] = layer = ConcatLayer([conv_layer, seq_layer], axis=1)
    for lstm_layer_idx in xrange(lstm_layers):
        ret['lstm_{}'.format(lstm_layer_idx)] = layer = LSTMLayer(layer, n_hid)
    ret['out_resh'] = layer = ReshapeLayer(layer, shape=(-1, n_hid))
    ret['output_proj'] = layer = DenseLayer(layer, num_units=word_dims, nonlinearity=log_softmax)
    ret['output'] = layer = ReshapeLayer(layer, shape=(batchsize, seqlen+1, word_dims))
    ret['output'] = layer = SliceLayer(layer, indices=slice(None, -1), axis=1)
    return ret

# originally from
# https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
def build_vgg(shape, input_var):
    net = {}
    w,h = shape
    net['input'] = InputLayer((None, 3, w, h), input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')
    return net
