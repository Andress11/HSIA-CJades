import numpy as np


RGB_LAB_MATRIX_D65 = np.array([[0.4124, 0.3576, 0.1805],
                               [0.2126, 0.7152, 0.0722],
                               [0.0193, 0.1192, 0.9505]])

XYZ_D65_STANDAR_ILUMINATION = np.array([0.950456, 1.0, 1.088754])
EPSILON = 216 / 24389
KAPPA = 24389/27