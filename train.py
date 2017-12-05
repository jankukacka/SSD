# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Training of a SSD model
# ------------------------------------------------------------------------------

from net import Simple_SSD
from multibox_loss import MultiboxLoss
import keras

model = Simple_SSD()
model.summary()
model.compile(loss=MultiboxLoss, optimizer='sgd')
