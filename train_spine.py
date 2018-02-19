#!/usr/bin/env python
# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 2/2018
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Training of a SSD model to predict the whole spine bounding boxes
# ------------------------------------------------------------------------------

# --
import os
import keras
import time as t
import datetime
import cPickle
import argparse
import keras.callbacks
from math import sqrt
# --
from net import Residual_SSD
from multibox_loss import MultiboxLoss
from data import OnlineSpineDataGenerator
from weightnorm import SGDWithWeightnorm, data_based_init
from anchor_generator_layer import AnchorGenerator
# --

# ------------------------------------------------------------------------------
## Parameters
# ------------------------------------------------------------------------------
## Number of epochs to train
num_epochs = 150

## Save snapshot every n-th epoch. If <= 0, no snapshots will be saved
snapshot_epoch = 50

## Projection type ('mean' or 'max')
aggregation_method = 'mean'

## Axis (coronal/sagittal)
aggregation_plane = 'sagittal'

## Use weight normalizaton training? If False, use Adam
use_weightnorm = False

## Use batch normalization
use_batchnorm = True

## Folder for tensorboard logs
tensorboard_folder = 'logs_spine'

## Use CPU only?
cpu_only = False
# ------------------------------------------------------------------------------

def main():

    # ------------------------------------------------------------------------------
    #  Run CPU only
    if cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # ------------------------------------------------------------------------------
    #  Limit memory usage
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=config))
    # ------------------------------------------------------------------------------

    ## Output folder for trained model
    output_folder = 'output/residual_ssd/cts_{}_{}_train_spine/'.format(
    aggregation_plane, aggregation_method)

    def ensure_dirs(path):
        import errno

        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    ## Make output directories
    ensure_dirs(output_folder)

    ## Prepare data generatrs
    def get_augmenter_settings(mean, std, aggregation_plane, aggregation_method):
        settings = {
            'use_crop': True,
            'max_crop': 0.7,
            'zmuv_mean': mean,
            'zmuv_std': std,
            'aggregation_plane': aggregation_plane,
            'aggregation_method': aggregation_method
        }
        if aggregation_plane == 'coronal' and aggregation_method == 'mean':
            settings['aggregation_scale'] = 0.01
        return settings


    aug_settings_train = get_augmenter_settings(209.350884188, 353.816477769,
                                                aggregation_plane, aggregation_method)
    aug_settings_val = get_augmenter_settings(-103.361759224, 363.301491674,
                                              aggregation_plane, aggregation_method)
    if aggregation_plane == 'coronal':
        aspect_ratios = [sqrt(.2), sqrt(.4)]
        scales = (5,6.5)
        min_wh_ratio=.05
    else:
        aspect_ratios = [sqrt(2.5), sqrt(3.5)]
        scales = (5,7.5)
        min_wh_ratio=.3

    ag = AnchorGenerator(feature_stride=32,
                         offset=0,
                         aspect_ratios=aspect_ratios,
                         scale=scales)
    gen_train = OnlineSpineDataGenerator(batch_size=2, imageset_name='train_large',
                                    cts_root_path='/media/Data/Datasets/ct-spine',
                                    settings=aug_settings_train,
                                    overlap_threshold=.5,
                                    anchor_generator=ag,
                                    min_wh_ratio=min_wh_ratio)
    gen_val = OnlineSpineDataGenerator(batch_size=30, imageset_name='valid_large',
                                    cts_root_path='/media/Data/Datasets/ct-spine',
                                    settings=aug_settings_val,
                                    overlap_threshold=.5,
                                    anchor_generator=ag,
                                    min_wh_ratio=min_wh_ratio)
    data_val = next(gen_val.Generate())

    ## Prepare model
    model = Residual_SSD(2, use_bn=use_batchnorm, num_anchors=len(aspect_ratios)*len(scales))

    ## Use Weightnorm with data-based initialization
    if use_weightnorm:
        opt = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        data_based_init(model, [next(gen_train.Generate()) for _ in range(10)]) # 100

    ## Use Adam
    else:
        opt = keras.optimizers.Adam(lr=.0001)

    ## Use pre-trained weights
    #with open('output/simple_ssd/cts_sagittal_train/epoch_{:d}.pkl'.format(11)) as f:
    #    w = cPickle.load(f)
    #model.set_weights(w)

    model.compile(loss=lambda y_true, y_pred: MultiboxLoss(y_true, y_pred, num_classes=2),
                  optimizer=opt)

    ## Generate tensorboard output folder
    run_name = tensorboard_folder
    if run_name[-1] != '/': run_name += '/'
    run_name += 'weightnorm' if use_weightnorm else 'adam'
    run_name += '_'
    if use_batchnorm: run_name += 'bn_'
    ## timestamp, e.g. 180216-152735 (year month day - hour minute second)
    run_name += datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    tb_callback = keras.callbacks.TensorBoard(histogram_freq=1,
                                              batch_size=2,
                                              #write_graph=True,
                                              write_grads=True,
                                              #write_images=True,
                                              log_dir='./' + run_name)
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1, patience=20, mode='min', verbose='1')

    def snapshot(model, epoch):
        epoch += 1 # 1-indexed epoch
        if snapshot_epoch > 0 and epoch % snapshot_epoch == 0:
            w = model.get_weights()
            with open(output_folder + 'epoch_{}.pkl'.format(epoch), 'wb') as f:
                cPickle.dump(w, f)

    snapshot_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda e,l: snapshot(model, e))

    results = {}
    tic = t.clock()
    print 'starting...'
    hist = model.fit_generator(gen_train.Generate(),
                     steps_per_epoch=gen_train.steps_per_epoch,
                     epochs=num_epochs, verbose=1,
                     validation_data=data_val,#gen_val.Generate(),
                     #validation_steps=gen_val.steps_per_epoch,
                     shuffle=False,
                     callbacks=[tb_callback, reduce_lr_callback, snapshot_callback])
    toc = t.clock()
    # log time
    hist.history['time'] = [toc-tic]

    # append history to results
    for key in hist.history:
        if key in results:
            results[key].extend(hist.history[key])
        else:
            results[key] = hist.history[key]

    print results
    with open(output_folder + 'report.txt', 'w') as f:
        f.write(str(results))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a SSD network',
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--aggregation_method', '-m', dest='aggregation_method',
                        help='Which aggregation method to use: max or mean',
                        type=str)
    parser.add_argument('--aggregation_plane','-p', dest='aggregation_plane',
                        help='Which plane to use: coronal or sagittal',
                        type=str)
    parser.add_argument('--epochs', '-e', dest='num_epochs',
                        help='Number of epochs to train. Default ' + str(num_epochs),
                        type=int)
    parser.add_argument('--no_batchnorm', '-nobn', dest='use_batchnorm',
                        help='Do not use batchnorm.',
                        default=True, action='store_false')
    parser.add_argument('--weightnorm','-wn', dest='use_weightnorm',
                        help='Use weightnorm+SGD instead of Adam.',
                        default=False, action='store_true')
    parser.add_argument('--tensorboard_folder', '-tb', dest='tensorboard_folder',
                        help='TensorBoard logs folder. Default ' + tensorboard_folder,
                        type=str)
    parser.add_argument('--snapshot_epoch', '-s', dest='snapshot_epoch',
                        help='Snapshot every n-th epoch. Default ' + str(snapshot_epoch),
                        type=int)
    parser.add_argument('--cpu', '-c', dest='cpu_only',
                        help='Use CPU only.',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    args = vars(args)

    if 'aggregation_method' in args:
        aggregation_method = args['aggregation_method']
    if 'aggregation_plane' in args:
        aggregation_plane = args['aggregation_plane']
    if 'num_epochs' in args:
        num_epochs = args['num_epochs']
    if 'use_batchnorm' in args:
        use_batchnorm = args['use_batchnorm']
    if 'use_weightnorm' in args:
        use_weightnorm = args['use_weightnorm']
    if 'tensorboard_folder' in args:
        tensorboard_folder = args['tensorboard_folder']
    if 'snapshot_epoch' in args:
        snapshot_epoch = args['snapshot_epoch']
    if 'cpu_only' in args:
        cpu_only = args['cpu_only']

    main()
