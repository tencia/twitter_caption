# Twitter Captioning using Show and Tell
Uses a variant of the model presented by [1] Vinyals 2014 ( http://arxiv.org/abs/1411.4555 ) to learn to generate tweets that accompany pictures from Twitter.

### Model

As detailed in [1], the model extracts convolutional features and uses them as the initial input to an LSTM, with each word of the accompanying tweet as the next input in the sequence. Both the convolutional features and the one-hot word vector are projected into the LSTM's hidden state.

This model uses a [Lasagne implementation of VGG-net](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb) [2] Simonyan 2014 ( http://arxiv.org/abs/1409.1556 ) for feature extraction.

### Results

Some of the captions almost make sense.

### Dependencies
- lasagne
- theano
- fuel
- hdf5
- tqdm
