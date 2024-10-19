import numpy as np

def unfolding(img:np.ndarray):
    rows, columns, bands = img.shape
    img_unfold = np.reshape(img,(rows*columns,bands))
    return img_unfold

def folding(img:np.ndarray, rows, columns, bands):
    img_fold = np.reshape(img,(rows,columns,bands))
    return img_fold