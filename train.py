# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Training of a SSD model
# ------------------------------------------------------------------------------

from net import Simple_SSD
import keras

model = Simple_SSD()
model.summary()
