import torch
import numpy as np

from evluate import Acc

golden = np.array([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
golden=torch.from_numpy(golden)
predict=(0, 1, 1, 1, 2, 2, 2, 0, 0, 2, 2)
predict=np.array(predict)
predict=torch.from_numpy(predict)
acc=Acc()
acc.update(predict,golden)
print(acc.acc)