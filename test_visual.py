# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Testing of a trained SSD model with visualization of results
# ------------------------------------------------------------------------------


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --
import numpy as np
import cPickle
import matplotlib.pyplot as plt
# --
from net import Residual_SSD
from data import DataGenerator
# --

model = Residual_SSD(num_classes=2)

with open('output/simple_ssd/cts_sagittal_train/epoch_0.pkl', 'rb') as f:
    w = cPickle.load(f)
model.set_weights(w)

gen = DataGenerator(5, '/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_train_large/', max_images=10, use_two_classes=True)
x_test, y_test = gen.Generate(shuffle=False).next()

pred = model.predict(x_test, batch_size=5)
print np.sum(pred[:,:,5:-4]>.7, axis=(1,2))

pred[-1,0,5:-4]

#%%
def display_img_and_boxes(img, pred, gt):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    # plot GT boxes
    for i in xrange(gt.shape[0]):
        ax.add_patch(
            plt.Rectangle((gt[i][0]-.5*gt[i][2], gt[i][1]-.5*gt[i][3]),
                          gt[i][2], gt[i][3], fill=False, edgecolor='lightblue',
                          linewidth=1))
    for i in xrange(pred.shape[0]):
        ax.add_patch(
            plt.Rectangle((pred[i][0]-.5*pred[i][2], pred[i][1]-.5*pred[i][3]),
                          pred[i][2], pred[i][3], fill=False, edgecolor='lightgreen',
                          linewidth=1))

    plt.show()

def pred2bbox(anchor, pred):
    #result = np.zeros_like(anchor)
    centers = anchor[:,:,2:]*pred[:,:,:2] + anchor[:,:,:2]
    sizes = anchor[:,:,2:]*np.exp(pred[:,:,2:])
    return np.concatenate((centers, sizes), axis=-1)

img_index = 0
img = x_test[img_index,:,:,0]
single_image = pred2bbox(pred[:,:,-4:],pred[:,:,:4])[img_index]
top_k = np.argsort(-np.max(pred[img_index,:,5:-4], axis=-1))[:20] # k = 10
plt.hist(pred[img_index,:,5])
plt.show()
print pred[img_index, top_k]
display_img_and_boxes(img, single_image[top_k], y_test[img_index,:,:4])



#%%
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from multibox_loss import _assign_boxes, _get_hard_negatives, _ignore_boundary_boxes, _l1_smooth_loss, _cross_entropy, _get_boundary_mask
def inspect_loss(y_pred, y_true, num_classes=4, overlap_threshold=0.5, alpha=1):
    anchors = y_pred[:,:,-4:]

    gt = _assign_boxes(anchors=anchors, gt_boxes=y_true,
                       overlap_threshold=overlap_threshold,
                       num_classes=num_classes)
    gt = tf.Print(gt, [tf.count_nonzero(gt[:,:,4], axis=-1)], message='A.# of samples with indicator on:')
    boundary_mask = _get_boundary_mask(anchors)
    boundary_mask = tf.Print(boundary_mask, [K.sum(boundary_mask)], message='Boundary mask:', summarize=544)
    gt = _get_hard_negatives(gt=gt, conf=(y_pred[:,:,4:5]*boundary_mask)[:,:,0])
    gt = tf.Print(gt, [tf.count_nonzero(gt[:,:,4], axis=-1)], message='B.# of samples with indicator on:')
    gt = _ignore_boundary_boxes(gt, boundary_mask)
    gt = tf.Print(gt, [tf.count_nonzero(gt[:,:,4], axis=-1)], message='C.# of samples with indicator on:')

    targets = tf.concat(((gt[:,:,:2] - anchors[:,:,:2]) / anchors[:,:,-2:],
                          tf.log(gt[:,:,2:4]/anchors[:,:,-2:])), axis=-1)
    loc_error = _l1_smooth_loss(y_true=targets, y_pred=y_pred[:,:,:4])
    conf_error = _cross_entropy(y_true=gt[:,:,5:], y_pred=y_pred[:,:,4:-4])

    # Mask out loss of invalid anchors (have indicator == 0)
    loc_error = tf.where(tf.equal(gt[:,:,4], 1.), loc_error, tf.zeros_like(loc_error))
    conf_error = tf.where(tf.equal(gt[:,:,4], 1.), conf_error, tf.zeros_like(conf_error))

    # Mask out localization loss of negative samples
    loc_error = tf.where(tf.equal(gt[:,:,5], 1.), loc_error, tf.zeros_like(loc_error))

    loss = (K.sum(loc_error+alpha*conf_error, axis=-1, keepdims=True)
            / (K.sum(gt[:,:,4], axis=-1, keepdims=True) + K.epsilon()))
    #loss = tf.Print(loss, [K.shape(loss)], message='loss.shape', summarize=100)
    return loss, gt, targets

sess = K.get_session()
input_tensor = model.input
output_tensor = model.output
gt = K.placeholder(shape=(None,None,12), name='gt')
loss, out_gt, out_targets = inspect_loss(output_tensor, gt)
res = out_gt.eval(feed_dict={input_tensor:x_test, gt:y_test}, session=sess)
tar = out_targets.eval(feed_dict={input_tensor:x_test, gt:y_test}, session=sess)
pred = output_tensor.eval(feed_dict={input_tensor:x_test, gt:y_test}, session=sess)

#%%
print np.sum(res[0,:,4]>=1)
print res[0,101,4]

valid_anchors = res[0,:,4]==1
img = x_test[0,:,:,0]
display_img_and_boxes(img, pred[0,valid_anchors,-4:],  pred2bbox(pred[:,valid_anchors,-4:],pred[:,valid_anchors, :4])[0])
#display_img_and_boxes(img, res[0,valid_anchors,:4],  pred2bbox(pred[:,valid_anchors,-4:],tar[:,valid_anchors])[0])

#%%
with open('output/simple_ssd/cts_sagittal_train/epoch_0.pkl') as f:
    w = cPickle.load(f)
print w[0].shape
print w[0][:,:,0,0]
#%%
# Visualize 1st layer filters
for k in xrange(30):
    with open('output/simple_ssd/cts_sagittal_train/epoch_{:d}.pkl'.format(k)) as f:
        w = cPickle.load(f)
    for i in xrange(8):
        for j in xrange(8):
            plt.subplot(8,8,i*8+j+1)
            plt.imshow(w[0][:,:,0,i*8+j], cmap='gray')
            plt.axis('off')
    plt.show()
# %%
import keras
## Get activations in 6th layers
intermediate_layer_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer('conv6_2').output)
intermediate_output = intermediate_layer_model.predict(x_test)
print intermediate_output.shape

plt.imshow(intermediate_output[0,:,:,4], cmap='gray')
plt.show()
# %%
## Print gradients
from net import Simple_SSD
from multibox_loss import MultiboxLoss
from data import DataGenerator
import keras
import keras.backend as K
import numpy as np
gen_train = DataGenerator(batch_size=5, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_train_large', max_images=100)
gen_val = DataGenerator(batch_size=10, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_valid_large', max_images=20)

model = Simple_SSD()
#model.summary()
#sgd_wn = SGDWithWeightnorm(lr=0, decay=1e-6, momentum=0.9, nesterov=True)
#data_based_init(model, [next(gen_train.Generate()) for _ in range(100)])
print model.get_weights()[0][:,:,0,0]
sgd = keras.optimizers.SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=MultiboxLoss, optimizer=sgd)
x_data, y_data = next(gen_train.Generate())
#model.train_on_batch(np.ones_like(x_data), y_data)
print model.get_weights()[0][:,:,0,0]

grads = model.optimizer.get_gradients(model.total_loss, model.get_layer('conv6_2_bbox_loc').weights)
get_grads = K.function([model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase()], grads)


G = get_grads([x_data, [1], y_data, 0])

print np.count_nonzero(~np.isnan(G[0][:,:,:,2])), np.size(G[0][:,:,:,2])
print G[1]
