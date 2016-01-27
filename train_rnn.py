import pickle
import numpy as np
import os
import sys
from PIL import Image

import lasagne as nn
import theano
import theano.tensor as T

import utils as u
import config as c
import models as m

def main(n_hid=256, lstm_layers=2, num_epochs=100,
        batch_size=32, save_to='output', max_per_epoch=-1):

    # load current set of words used
    words = open(c.words_used_file, 'r').readlines()
    idx_to_words = dict((i+1,w.strip()) for i,w in enumerate(words))
    idx_to_words[0] = '<e>'
    word_dim=len(words)+1

    # normalization expected by vgg-net
    mean_values = np.array([104, 117, 123]).reshape((3,1,1)).astype(theano.config.floatX)

    # build function for extraction convolutional features
    img_var = T.tensor4('images')
    net = m.build_vgg(shape=(c.img_size, c.img_size), input_var=img_var)
    values = pickle.load(open(c.vgg_weights))['param values']
    nn.layers.set_all_param_values(net['pool5'], values)
    conv_feats = theano.function([img_var], nn.layers.get_output(net['pool5']))
    conv_shape = nn.layers.get_output_shape(net['pool5'])

    # helper function for converting word vector to one-hot
    raw_word_var = T.matrix('seq_raw')
    one_hot = theano.function([raw_word_var], nn.utils.one_hot(raw_word_var, m=word_dim))

    # build expressions for lstm
    conv_feats_var = T.tensor4('conv')
    seq_var = T.tensor3('seq')
    lstm = m.build_rnn(conv_feats_var, seq_var, conv_shape, word_dim, n_hid, lstm_layers)
    output = nn.layers.get_output(lstm['output'])
    output_det = nn.layers.get_output(lstm['output'], deterministic=True)
    loss = m.categorical_crossentropy_logdomain(output, seq_var).mean()
    te_loss = m.categorical_crossentropy_logdomain(output_det, seq_var).mean()

    # compile training functions
    params = nn.layers.get_all_params(lstm['output'], trainable=True)
    lr = theano.shared(nn.utils.floatX(1e-3))
    updates = nn.updates.adam(loss, params, learning_rate=lr)
    train_fn = theano.function([conv_feats_var, seq_var], loss, updates=updates)
    test_fn = theano.function([conv_feats_var, seq_var], te_loss)
    predict_fn = theano.function([conv_feats_var, seq_var], T.exp(output_det[:,-1:]))

    zeros = np.zeros((batch_size, 1, word_dim), dtype=theano.config.floatX)
    def transform_data(imb):
        y,x = imb
        # data augmentation: flip = -1 if we do flip over y-axis, 1 if not
        flip = -2*np.random.binomial(1, p=0.5) + 1
        # this vgg-net expects image values that are normalized by mean but not magnitude
        x = (u.raw_to_floatX(x[:,:,::flip], pixel_shift=0.)\
                .transpose(0,1,3,2)[:,::-1] * 255. - mean_values)
        return conv_feats(x), np.concatenate([zeros, one_hot(y)], axis=1)

    data = u.DataH5PyStreamer(c.twimg_hdf5_file, batch_size=batch_size)

    hist = u.train_with_hdf5(data, num_epochs=num_epochs, train_fn=train_fn, test_fn=test_fn,
                      max_per_epoch=max_per_epoch,
                      tr_transform=transform_data,
                      te_transform=transform_data)
    np.savetxt('lstm_train_hist.csv', np.asarray(hist), delimiter=',', fmt='%.5f')
    u.save_params(lstm['output'], os.path.join(save_to,
        'lstm_{}.npz'.format(np.asarray(hist)[-1, -1])))


    # generate some example captions for one batch of images
    streamer = data.streamer(training=False, shuffled=True)
    y_raw, x_raw = next(streamer.get_epoch_iterator())
    x, _ = transform_data((y_raw, x_raw))

    y = zeros
    captions = []
    for idx in xrange(y.shape[0]):
        captions.append([])
    idx_to_words[0] = '<e>'
    for sample_num in xrange(c.max_caption_len):
        pred = predict_fn(x, y)
        new_y = []
        for idx in xrange(pred.shape[0]):
            # reduce size by a small factor to prevent numerical imprecision from
            # making it sum to > 1.
            # reverse it so that <e> gets the additional probability, not a word
            sample = np.random.multinomial(1, pred[idx,0,::-1]*.999999)[::-1]
            captions[idx].append(idx_to_words[np.argmax(sample)])
            new_y.append(sample)
        new_y = np.vstack(new_y).reshape(-1,1,word_dim).astype(theano.config.floatX)
        y = np.concatenate([y, new_y], axis=1)
    captions = ['{},{}\n'.format(i, ' '.join(cap)) for i,cap in enumerate(captions)]
    with open(os.path.join(save_to, 'captions_sample.csv'), 'w') as wr:
        wr.writelines(captions)

    for idx in xrange(x_raw.shape[0]):
        Image.fromarray(x_raw[idx].transpose(2,1,0)).save(os.path.join(save_to,
            'ex_{}.jpg'.format(idx)))

if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name,arg_type in zip(ma.args[-len(ma.defaults):],[type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
