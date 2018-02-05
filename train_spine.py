# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Training of a SSD model to predict the whole spine bounding boxes
# ------------------------------------------------------------------------------

# --
import os
import keras
import time as t
import cPickle
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

## Option to save weight snapshots
save_snapshots = True

## Output folder for trained model
output_folder = 'output/residual_ssd/cts_sagittal_train_spine/'

## Name of the run
run_name = 'logs_spine/adam_bn_3'

## Use weight normalizaton training? If False, use Adam
use_weightnorm = False

## Use CPU only?
cpu_only = False
# ------------------------------------------------------------------------------

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
aug_settings_train = {
    'use_crop': True,
    'max_crop': 0.7,
    'zmuv_mean': 209.350884188,
    'zmuv_std': 353.816477769
}
aug_settings_val = {
    'use_crop': True,
    'max_crop': 0.7,
    'zmuv_mean': -103.361759224,
    'zmuv_std': 363.301491674
}
aspect_ratios = [sqrt(2.5), sqrt(3.5)]
scales = (5,7.5)
ag = AnchorGenerator(feature_stride=32,
                     offset=0,
                     aspect_ratios=aspect_ratios,
                     scale=scales)
gen_train = OnlineSpineDataGenerator(batch_size=2, imageset_name='train_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_train,
                                overlap_threshold=.5,
                                anchor_generator=ag)
gen_val = OnlineSpineDataGenerator(batch_size=30, imageset_name='valid_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_val,
                                overlap_threshold=.5,
                                anchor_generator=ag)
data_val = next(gen_val.Generate())

## Prepare model
model = Residual_SSD(2, use_bn=True, num_anchors=len(aspect_ratios)*len(scales))

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
tb_callback = keras.callbacks.TensorBoard(histogram_freq=1,
                                          batch_size=2,
                                          #write_graph=True,
                                          write_grads=True,
                                          #write_images=True,
                                          log_dir='./' + run_name)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1, patience=20, mode='min', verbose='1')

def snapshot(model, epoch):
    if epoch > 0 and epoch % 50 == 0:
        w = model.get_weights()
        with open(output_folder + 'epoch_{}.pkl'.format(epoch), 'wb') as f:
            cPickle.dump(w, f)

snapshot_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda e,l: snapshot(model, e))

epochs = 1
best_accuracy = 0.0
best_epoch = 0
results = {}
t_start = t.clock()
for e in xrange(epochs):
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
    w = model.get_weights()
    print w[0][:,:,0,0]

    # if save_snapshots and e % 5 == 0:
    #     with open(output_folder + 'epoch_{}.pkl'.format(e), 'wb') as f:
    #         cPickle.dump(w, f)

print results
with open(output_folder + 'report.txt', 'w') as f:
    f.write(str(results))
