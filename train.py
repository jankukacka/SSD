# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Training of a SSD model
# ------------------------------------------------------------------------------

# --
import keras
import time as t
import cPickle
import keras.callbacks
# --
from net import Simple_SSD, Residual_SSD
from multibox_loss import MultiboxLoss
from data import OnlineDataGenerator
from weightnorm import SGDWithWeightnorm, data_based_init
# --

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# ------------------------------------------------------------------------------
#  Limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
# ------------------------------------------------------------------------------

# gen_train = DataGenerator(batch_size=2, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_train_large')
# gen_val = DataGenerator(batch_size=2, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_valid_large')

## Option to save weight snapshots
save_snapshots = True
## Option to use simpler, two-class prediction
use_two_classes = True
# --
num_classes = 4
if use_two_classes:
    num_classes = 2
# --

aug_settings_train = {
    'use_crop': True,
    'zmuv_mean': 209.350884188,
    'zmuv_std': 353.816477769
}
aug_settings_val = {
    'use_crop': True,
    'zmuv_mean': -103.361759224,
    'zmuv_std': 363.301491674
}
gen_train = OnlineDataGenerator(batch_size=2, imageset_name='train_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_train,
                                use_two_classes=use_two_classes)
gen_val = OnlineDataGenerator(batch_size=30, imageset_name='valid_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_val,
                                use_two_classes=use_two_classes)
data_val = next(gen_val.Generate())

model = Residual_SSD(num_classes)
#model.summary()
## Use Weightnorm with data-based initialization
opt = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
data_based_init(model, [next(gen_train.Generate()) for _ in range(100)])

## Use Adam
# opt = keras.optimizers.Adam(lr=.0001)
## Use pre-trained weights
#with open('output/simple_ssd/cts_sagittal_train/epoch_{:d}.pkl'.format(11)) as f:
#    w = cPickle.load(f)
#model.set_weights(w)

model.compile(loss=lambda y_true, y_pred: MultiboxLoss(y_true, y_pred, num_classes=num_classes),
              optimizer=opt)

callback = keras.callbacks.TensorBoard(histogram_freq=1,
                                       batch_size=2,
                                       write_graph=True,
                                       write_grads=True,
                                       write_images=True,
                                       log_dir='./logs/weightnorm_3')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=20, mode='min', verbose='1')

epochs = 1
best_accuracy = 0.0
best_epoch = 0
results = {}
t_start = t.clock()
for e in xrange(epochs):
    tic = t.clock()
    hist = model.fit_generator(gen_train.Generate(),
                     steps_per_epoch=gen_train.steps_per_epoch,
                     epochs=150, verbose=1,
                     validation_data=data_val,#gen_val.Generate(),
                     #validation_steps=gen_val.steps_per_epoch,
                     shuffle=False,
                     callbacks=[callback, reduce_lr])
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

    if save_snapshots and e % 5 == 0:
        with open('output/simple_ssd/cts_sagittal_train/epoch_{}.pkl'.format(e), 'wb') as f:
            cPickle.dump(w, f)

print results
with open('output/simple_ssd/cts_sagittal_train/report.txt', 'w') as f:
    f.write(str(results))
#return history
