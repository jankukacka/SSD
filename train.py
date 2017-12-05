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
# --
from net import Simple_SSD
from multibox_loss import MultiboxLoss
from data import DataGenerator
# --

model = Simple_SSD()
#model.summary()
model.compile(loss=MultiboxLoss, optimizer='sgd')

gen_train = DataGenerator(batch_size=1, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_train_large')
gen_val = DataGenerator(batch_size=1, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_valid_large')

epochs = 30
best_accuracy = 0.0
best_epoch = 0
results = {}
t_start = t.clock()
for e in xrange(epochs):
    tic = t.clock()
    hist = model.fit_generator(gen_train.Generate(),
                     steps_per_epoch=gen_train.steps_per_epoch,
                     epochs=1, verbose=1,
                     validation_data=gen_val.Generate(),
                     validation_steps=gen_val.steps_per_epoch,
                     shuffle=False)
    toc = t.clock()
    # log time
    hist.history['time'] = [toc-tic]

    # append history to results
    for key in hist.history:
        if key in results:
            results[key].extend(hist.history[key])
        else:
            results[key] = hist.history[key]

    val_acc = hist.history['val_acc'][0]
    if val_acc > best_accuracy:
        best_weights = model.get_weights()
        best_epoch = e + 1 # e is 0-indexed
best_accuracy = val_acc
