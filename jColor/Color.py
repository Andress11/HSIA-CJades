import numpy as np
import cv2
from .const import (RGB_LAB_MATRIX_D65, XYZ_D65_STANDAR_ILUMINATION, EPSILON,KAPPA)
from .utils import (folding, unfolding)


class Color:

    def __init__(self,Image):

        iRGB = np.copy(Image)
        self.Lab = self._RGB2Lab(iRGB, fold = True)
        #self.Lch = self._Lab2Lch(self.Lab,fold = True)
    

    def _companding_sRGB(self,rgb_values: np.ndarray):
        
        idx = rgb_values <= 0.04045

        rgb_values[idx] =  rgb_values[idx]/(12.92)
        rgb_values[~idx] = ((rgb_values[~idx]+0.055)/1.055)**2.4

        return rgb_values
    
    def _inverse_companding_sRGB(self, rgb_values):

        idx = rgb_values <= 0.0031308
        rgb_values[idx] = rgb_values[idx]*(12.92)
        rgb_values[~idx] = (rgb_values[~idx]**(1/2.4))*1.055 - 0.055

        return rgb_values
    
    def _RGB2XYZ(self,img_RGB, fold: bool = True):
        rows, columns, bands = img_RGB.shape
        RGB = unfolding(np.copy(img_RGB))
        rgb = self._companding_sRGB(RGB)
        XYZ = np.dot(rgb, RGB_LAB_MATRIX_D65.T)
        if fold is True:
            XYZ = folding(XYZ,rows,columns, bands)    
        return XYZ 
    
    def _XYZ2Lab(self, img_XYZ, fold: bool = True):
        rows, columns, bands = img_XYZ.shape
        XYZ = unfolding(np.copy(img_XYZ))
        nXYZ = XYZ/ XYZ_D65_STANDAR_ILUMINATION
        condition = nXYZ > EPSILON
        cXYZ = np.where(condition, nXYZ ** (1 / 3), (nXYZ * 903.3 + 16) / 116)
        L = 116 * cXYZ[:, 1] - 16
        L = np.clip(L, 0, 100)
        a = 500 * (cXYZ[:, 0] - cXYZ[:, 1])
        b = 200 * (cXYZ[:, 1] - cXYZ[:, 2])
        Lab = np.column_stack((L, a, b))
        if fold is True:
            Lab = folding(Lab,rows,columns,bands)
        return Lab
    
    def _Lab2Lch(self, img_Lab: np.ndarray,fold: bool = False):

        rows, columns, bands = img_Lab.shape
        Lab = unfolding(img_Lab)

        L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
        
        c = np.sqrt(a**2 + b**2)
        h = np.arctan2(b, a) * (180 / np.pi)
        h[h < 0] += 360

        Lch = np.column_stack((L, c, h))
        if fold is True:
            Lch = folding(Lch,rows, columns, bands)
    
        return Lch

    def _RGB2Lab(self,img_RGB, fold: bool = True):
        XYZ = self._RGB2XYZ(img_RGB, fold = True)
        Lab = self._XYZ2Lab(XYZ, fold = fold)
        return Lab

    def _Lab2XYZ(self, img_Lab, fold: bool = True):
        rows, columns, bands = img_Lab.shape
        Lab = unfolding(np.copy(img_Lab))
        L, a, b =  Lab[:, 0], Lab[:, 1], Lab[:, 2]
        fy = (L + 16)/116
        fx = a/500 + fy
        fz = fy - b/200  
        X = np.where(fx**3 > EPSILON, fx**3, (116*fx - 16)/KAPPA)
        Y = np.where(L > KAPPA*EPSILON, fy**3, L/KAPPA)
        Z = np.where(fz**3 > EPSILON, fz**3, (116*fx - 16)/KAPPA)
        nXYZ = np.column_stack((X,Y,Z))
        XYZ = nXYZ*XYZ_D65_STANDAR_ILUMINATION
        if fold is True:
            XYZ = folding(XYZ,rows,columns,bands)
        return XYZ
    
    def _XYZ2RGB(self, img_XYZ, fold: bool = True):
        rows, columns, bands = img_XYZ.shape
        XYZ = unfolding(np.copy(img_XYZ))
        rgb = np.dot(XYZ, np.linalg.inv(RGB_LAB_MATRIX_D65).T)
        RGB = self._inverse_companding_sRGB(rgb)
        if fold is True:
            RGB = folding(RGB,rows,columns, bands)    
        return RGB
    
    def _Lab2RGB(self,img_lab,fold: bool = True):
        XYZ = self._Lab2XYZ(img_lab)
        RGB = self._XYZ2RGB(XYZ)
        return RGB