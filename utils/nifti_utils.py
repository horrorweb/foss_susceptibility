import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io
import cmath                
from PIL import Image

import sys
sys.path.append('/home/milab/SSD_8TB/LeeSooHyung/b0inhomo/dephase_simulator/epi_feasibility_training/epi_20block/utils')

import nibabel as nib
import pandas as pd
import nilearn

# from common import *

def copy_nii_header(image_source, header_source): #source is image source, target is header source
    if type(image_source)==np.ndarray:
        temp_header=header_source.header
        temp_affine=header_source.affine
        return nib.Nifti1Image(image_source, temp_affine, temp_header)
    else:
        temp_header=header_source.header
        temp_affine=header_source.affine
        return nib.Nifti1Image(image_source.get_fdata(), temp_affine, temp_header)


def apply_mask(image_nii,mask_nii):
    masked_np=image_nii.get_fdata()*mask_nii.get_fdata()
    return copy_nii_header(masked_np,image_nii)

def zero_below_percentile(arr, percentile):
    element_num=torch.numel(arr)
    element_num_true=torch.count_nonzero(arr)
    percentile_of_true=element_num_true/element_num
    true_percentile=100-(100-percentile)*percentile_of_true
    threshold = np.percentile(arr, true_percentile)
    
    arr[arr < threshold] = 0
    return arr

def cutoff_nii(niifile,percentage):
    image=niifile.get_fdata()
    image_cutted=zero_below_percentile(torch.Tensor(image),percentage).numpy()
    return copy_nii_header(image_cutted, niifile)

    