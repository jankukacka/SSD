import ast
import os
path = os.path.expanduser('output/simple_ssd/cts_sagittal_train/report.txt')
with open(path) as f:
    s = f.read()

d = ast.literal_eval(s)
print d
#%%
import matplotlib.pyplot as plt
import numpy as np
plt.semilogy(d['loss'], label='training loss')
plt.semilogy(d['val_loss'], label='validation loss')
plt.title('Residual SSD training')
plt.gca().set_xlabel('epoch')
plt.gca().set_ylabel('loss')
plt.legend()
plt.show()
plt.savefig('output/simple_ssd/cts_sagittal_train/report.png', bbox_inches='tight')
