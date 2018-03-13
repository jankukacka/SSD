# Object detection using Single Shot multibox Detector (SSD)

This repository contains the code for detection of objects (spine, vertebrae) in medical images. Its most notable contribution is a working implementation of the SSD network for Keras+Tensor Flow. This code is using our private dataset which is not part of the repository. For usage with your own (medical) images it needs to be adjusted.

## Repository structure

### `./`

Contains two main scripts:

* `train_spine.py`

    Can be run from the command line to train the net for detection of the whole spine. Uses following command line arguments:
    * `--aggregation_method` Aggregtation of 3D scans to 2D images. Can be `mean` or `max`
    * `--aggregation_plane` Which view should be used. Can be `coronal` or `sagittal`
    * `--epochs` For how many epochs should the training proceed.
    * `--no_batchnorm` Flag to indicate that network architecture w/o batch normalizaton should be used
    * `--weightnorm` Flag to indicate that SGD + Weight normalizaton should be used for training instead of Adam.
    * `--tensorboard_folder` Path for storing TensorBoard logs
    * `--snapshot_epoch` How often should the snapshots be saved
    * `--cpu` Flag indicating that the net should run on CPU only.


* `predict_spine.py`

     Inference code to predict the spine bounding box using trained nets (coronal + sagittal) for a 3D scan. Currently has no command line interface and can be called using the `predict` function.


### `./lib`

Contains the library code:

* `anchor_assignment.py`

    Funtion `Match` takes care for assignment of objects' ground truth bounding boxes to the network's "anchor" bounding boxes and computing desired offsets which the network should predict.

* `anchor_generator.py`

    Contains class `AnchorGenerator` whose function `Generate` produces a list of anchor bounding boxes for a specified size of the input data. Each object detector needs its own `AnchorGenerator`.

* `data_augmenter.py`

    Contains class `DataAugmenter` which performs data processing, data augmentation and currently also data loading using my private `dataset_sdk` and data caching.

* `data.py`

    Contains data generators `OnlineDataGenerator` and its subclass `OnlineSpineDataGenerator`, which generate data for vertebra and spine detection respectively.

* `multibox_loss.py`

    Implements `MultiboxLoss` loss function for training the SSD network.

* `net.py`

    Contains implementation of our Residual SSD neural net (function `Residual_SSD`).

* `utils.py`

    Contains implementation of the following utilities:
    * Fast Non-maximum suppression (function `nms`)
    * Bounding box aggregation routines for bounding boxes in format x1y1x2y2 (`aggregate_bboxes`) and format ccwh (`aggregate_bboxes_ccwh`). These aggregators convert a group of bounding boxes to a single one using either mean or hull.


* `weightnorm.py`

    Contains implementation of SGD with Weight normalization regularization. See https://arxiv.org/abs/1602.07868 for more info.

### `./visual_tools`

Contains various tools for data visualization and debugging.
