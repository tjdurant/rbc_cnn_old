
from __future__ import print_function

import lasagne
import theano
import theano.tensor as T
import time
import plac
import numpy as np
import pickle
import hickle
import sklearn.cross_validation
import skimage.transform

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer

DATA_DIR = './data'
OUTPUT_DIR = './output/'

RANDOM_SEED = 42

NUM_CLASSES = 12
IMAGE_W = 80
NUM_CHANNELS = 3

NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MOMENTUM = 0.9

np.random.seed(RANDOM_SEED)


def chunks(iterable, chunk_size):
    """Generate sequences of `chunk_size` elements from `iterable`."""
    iterable = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(iterable.next())
            yield chunk
        except StopIteration:
            if chunk:
                yield chunk
            break


def patch(x, rng=None, seed=0):
    if rng is None:
        rng = np.random.RandomState(seed)

    ang = rng.uniform(-20, 20)
    x = skimage.transform.rotate(x, ang, order=0)

    if rng.randint(2):
        x = np.fliplr(x)
    if rng.randint(2):
        x = np.flipud(x)
    if rng.randint(2):
        x = np.rot90(x)

    cropx, cropy = rng.randint(32, 64, 2)
    x = x[cropy:cropy + 160:2, cropx:cropx + 160:2]

    x = x.swapaxes(0, 2)
    x = x[::-1]

    return x


def get_data_sampler(train):
    d = hickle.load('{}/dataset.hkl'.format(DATA_DIR))
    sss = sklearn.cross_validation.StratifiedShuffleSplit(
        d['y'],
        n_iter=1,
        test_size=0.2,
        random_state=RANDOM_SEED,
        )
    if train:
        ix, _ = tuple(sss)[0]
    else:
        _, ix = tuple(sss)[0]

    images = d['X'][ix]
    labels = d['y'][ix]

    IMAGE_MEAN = images.mean(0).mean(0).mean(0)
    images = (images - IMAGE_MEAN)

    def sample(seed, N, p=0.5):
        rng = np.random.RandomState(seed)
        X = np.zeros((N, NUM_CHANNELS, IMAGE_W, IMAGE_W))
        Y = np.zeros(N)
        idx = range(len(images))
        rng.shuffle(idx)
        for i in range(N):
            X[i] = patch(images[idx[i]], rng=rng)
            Y[i] = labels[idx[i]]
        return X, Y

    return sample


def load_data(epoch_label, samplers):
    X_train, y_train = samplers['train'](epoch_label, BATCH_SIZE * 128)
    X_val, y_val = samplers['val'](0, BATCH_SIZE * 64)

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        y_train=y_train.astype('int32'),
        X_valid=lasagne.utils.floatX(X_val),
        y_valid=y_val.astype('int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_val.shape[0],
        output_dim=NUM_CLASSES,
    )


def build_model(input_width, input_height, output_dim,
                batch_size=BATCH_SIZE):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, NUM_CHANNELS, input_width, input_height),
    )

    l_conv1 = ConvLayer(
        l_in,
        num_filters=32,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )

    l_conv1b = ConvLayer(
        l_conv1,
        num_filters=32,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )
    l_conv1c = ConvLayer(
        l_conv1b,
        num_filters=32,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )

    l_pool1 = MaxPoolLayer(
        l_conv1c, pool_size=(3, 3), stride=(2, 2))
    l_dropout1 = lasagne.layers.DropoutLayer(l_pool1, p=0.25)

    l_conv2 = ConvLayer(
        l_dropout1,
        num_filters=64,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )
    l_conv2b = ConvLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )
    l_conv2c = ConvLayer(
        l_conv2b,
        num_filters=64,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )

    l_pool2 = MaxPoolLayer(
        l_conv2c, pool_size=(2, 2), stride=(2, 2))
    l_dropout2 = lasagne.layers.DropoutLayer(l_pool2, p=0.25)

    l_conv3 = ConvLayer(
        l_dropout2,
        num_filters=128,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )
    l_conv3b = ConvLayer(
        l_conv3,
        num_filters=128,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )
    l_conv3c = ConvLayer(
        l_conv3b,
        num_filters=128,
        filter_size=(3, 3),
        pad=0,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
        W=lasagne.init.Orthogonal(),
    )

    l_pool3 = lasagne.layers.GlobalPoolLayer(
        l_conv3c, pool_function=T.max)

    l_dropout3 = lasagne.layers.DropoutLayer(l_pool3, p=0.25)

    l_out = lasagne.layers.DenseLayer(
        l_dropout3,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.Orthogonal(),
    )
    return l_out


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')

    def loss(output):
        return -T.mean(T.log(output)[T.arange(y_batch.shape[0]), y_batch])

    loss_train = loss(lasagne.layers.get_output(output_layer, X_batch))
    loss_eval = loss(lasagne.layers.get_output(output_layer,
                                               X_batch,
                                               deterministic=True))

    pred = T.argmax(
        lasagne.layers.get_output(output_layer,
                                  X_batch,
                                  deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)

    updates = lasagne.updates.adam(
        loss_train, all_params, learning_rate=LEARNING_RATE)

    iter_train = theano.function(
        [X_batch, y_batch], loss_train,
        updates=updates,
    )

    iter_valid = theano.function(
        [X_batch, y_batch], [loss_eval, accuracy],
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
    )


def get_predictor(output_layer):
    X_batch = T.tensor4()
    pred = lasagne.layers.get_output(output_layer, X_batch, deterministic=True)
    predfunc = theano.function([X_batch], pred)

    def predict(X, batch_size=BATCH_SIZE):
        p = []
        for chunk in chunks(X, batch_size):
            p.append(predfunc(chunk))
        return np.concatenate(p)

    return predict


def train_epoch(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    batch_train_losses = []
    for batch_index in range(num_batches_train):
        batch_slice = slice(
            batch_index * batch_size, (batch_index + 1) * batch_size)
        batch_train_loss = iter_funcs['train'](
            dataset['X_train'][batch_slice], dataset['y_train'][batch_slice])
        batch_train_losses.append(batch_train_loss)

    avg_train_loss = np.mean(batch_train_losses)

    batch_valid_losses = []
    batch_valid_accuracies = []
    for batch_index in range(num_batches_valid):
        batch_slice = slice(
            batch_index * batch_size, (batch_index + 1) * batch_size)
        batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](
            dataset['X_valid'][batch_slice], dataset['y_valid'][batch_slice])
        batch_valid_losses.append(batch_valid_loss)
        batch_valid_accuracies.append(batch_valid_accuracy)

    avg_valid_loss = np.mean(batch_valid_losses)
    avg_valid_accuracy = np.mean(batch_valid_accuracies)

    return {
        'train_loss': avg_train_loss,
        'valid_loss': avg_valid_loss,
        'valid_accuracy': avg_valid_accuracy,
    }


def save_parameters(output_layer, meta):
    filename = '{0}models/{1} epoch {2}.pickle'.format(
        OUTPUT_DIR,
        meta['basename'],
        meta['epoch'])

    all_params = lasagne.layers.get_all_params(output_layer)
    all_param_values = [p.get_value() for p in all_params]

    pickle.dump(
        {'params': all_param_values, 'meta': meta}, open(filename, 'w'))

    print('Saved network parameters to {0}'.format(filename))


def load_parameters(output_layer, filename):
    model = pickle.load(open(filename, 'r'))
    all_param_values = model['params']
    all_params = lasagne.layers.get_all_params(output_layer)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)
    print('Loaded network parameters from {0}'.format(filename))
    return model['meta']


def setup():
    output_layer = build_model(IMAGE_W, IMAGE_W, NUM_CLASSES)

    layers = lasagne.layers.get_all_layers(output_layer)
    for i, layer in enumerate(layers):
        print('Layer {0} type: {1} output: {2}'.format(
            i, layer.__class__, layer.output_shape))

    samplers = {'train': get_data_sampler(train=True),
                'val': get_data_sampler(train=False),
                }

    return output_layer, samplers


def train(output_layer, samplers, num_epochs, basename):
    warmstart_fn = None
    warmstart_epoch = 0

    metrics = {'train': [99], 'val': [99], 'val_acc': [0]}

    learning_rate = LEARNING_RATE
    print('Learning rate = {0}'.format(learning_rate))

    print('Starting training...')
    t0 = time.time()
    t1 = time.time()
    i = 0
    meta = {'basename': basename,
            'start time': t0,
            'epoch': -1,
            'metrics': metrics,
            'random_state': np.random.get_state(),
            }
    save_parameters(output_layer, meta)

    stuck_count = 0

    while i < range(NUM_EPOCHS):
        if warmstart_fn:
            i = warmstart_epoch
            load_parameters(output_layer, warmstart_fn)
            warmstart_fn = None

        try:
            dataset = load_data(i, samplers)
        except IOError:
            print('Missing dataset for epoch {0}'.format(i))
            i += 1
            continue
        iter_funcs = create_iter_functions(
            dataset,
            output_layer,
            X_tensor_type=T.tensor4,
            learning_rate=learning_rate,
        )
        epoch = train_epoch(iter_funcs, dataset)
        if epoch['valid_loss'] < min(metrics['val']):
            stuck_count = 0
            meta['epoch'] = i
            meta['random_state'] = np.random.get_state()
            meta['metrics'] = metrics
            save_parameters(output_layer, meta)
        else:
            stuck_count += 1
            if stuck_count == 30:
                stuck_count = 0
                learning_rate *= 0.1
                print('Decreased learning rate to {0}'.format(learning_rate))

        metrics['train'].append(epoch['train_loss'])
        metrics['val'].append(epoch['valid_loss'])
        metrics['val_acc'].append(epoch['valid_accuracy'])

        t2 = time.time()
        print(('Epoch {0:3d}/{1:3d} duration {5:3.1f}s total {6:3.1f}min'
               '\t train {2:2.3f} \t val {3:2.3f} \t val acc {4:2.2f}%'
               ).format(i,
                        num_epochs,
                        epoch['train_loss'],
                        epoch['valid_loss'],
                        epoch['valid_accuracy'] * 100,
                        t2 - t1, (t2 - t0) / 60
                        ))
        t1 = t2
        i += 1


@plac.annotations(
    basename=plac.Annotation("Base name for output files", 'option', 'b'),
    num_epochs=plac.Annotation("Maximum training epochs", 'option', 'e'),
)
def main(basename='testrun', num_epochs=NUM_EPOCHS):
    print('Logging base name: {0}'.format(basename))

    output_layer, samplers = setup()

    train(output_layer, samplers, num_epochs, basename)

if __name__ == '__main__':
    import plac
    plac.call(main)
