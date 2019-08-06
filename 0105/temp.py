# -*- coding: utf-8 -*-
import numpy as np
std = np.array([0, -1])
if(np.any(std < 0)):
    std = std + 0.01
    print(std)
