import pandas as pd
import numpy as np

import cv2

import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
from nilearn.image import mean_img

import matplotlib.pyplot as plt

from utils.nifti_utils import *

def create_hrf(measurement_num=80,plot=False):
    tr = 1  # repetition time is 1 second
    n_scans = measurement_num  # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times

    # these are the types of the different trials
    conditions = ["on", "on"]
    duration = [20, 20]
    onsets = [20, 60]

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    hrf_model = "glover"
    X1 = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        hrf_model=hrf_model,
    )
    X1=pd.DataFrame.to_numpy(X1)
    output=np.squeeze(X1[:,0])
    # plot_design_matrix(X1)
    if plot==True:
        plt.plot(output)
        plt.show()
    return output

# def add_noise(input,constant,mean=0,std=1,positive=False,check=False,device=None):
#     signal_power=torch.norm(input)
    
#     if device!=None:
#         noise = constant*signal_power*torch.tensor(np.random.normal(mean, std, input.size()), dtype=torch.float).to(device)
#     else:    
#         noise = constant*signal_power*torch.tensor(np.random.normal(mean, std, input.size()), dtype=torch.float)
#     result=noise+input
#     if check==True:
#         print(torch.max(noise))
#     if positive==True:
#         result=torch.clamp(result,min=0)
#     return result

def signal_synthesize(image,var=80,wave_form='hrf'):
    BOLD = cv2.imread('BOLD.png', cv2.IMREAD_GRAYSCALE)
    BOLD=cv2.resize(BOLD, dsize=(128,128), interpolation=cv2.INTER_LINEAR)/255*20
    BOLD = cv2.rotate(BOLD, cv2.ROTATE_90_CLOCKWISE)
    mask_shape=BOLD
    if wave_form=='hrf':
        mag=create_hrf()
    elif wave_form=='linear':
        mag=np.array(list(range(int(-var/2),int(var/2))))
        mag=mag/var
    test_input=[]
    for i in range(var):
        test_input.append(image)
    test_input=np.stack(test_input,2)

    for j in range(var):
        mask=mag[j]*mask_shape #dev=6
        test_input[:,:,j]=test_input[:,:,j]+mask
        
    # if noise!=None:
    #     test_input=add_noise(test_input,noise,mean,std,True,True)
    index_0=np.where(BOLD==20)[0][10]
    index_1=np.where(BOLD==20)[1][10]
    BOLD[index_0,index_1]

    plt.plot(test_input[index_0,index_1,:])
    return test_input

def calculate_glm_synthesized(data, percentage=None, threshold=5,cut_coords=3,mod=80, plot=True,annotate=False,colorbar=True,fig=None,ax=None):
    
    tr = 1  # repetition time is 1 second
    n_scans = mod  # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times

    # these are the types of the different trials
    conditions = ["on", "on"]
    duration = [20, 20]
    onsets = [20, 60]

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    hrf_model = "glover"
    X1 = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        hrf_model=hrf_model,
    )

    # duration = 7.0 * np.ones(len(conditions))
    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    fmri_glm = FirstLevelModel(mask_img=False,high_pass=0,drift_order=3)
    fmri_glm = fmri_glm.fit(data, design_matrices=X1)

    contrast_matrix = np.eye(X1.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(X1.columns)
    }
    
    z_map = fmri_glm.compute_contrast(X1.columns[0], output_type="stat")
    if percentage!=None:
        z_map=cutoff_nii(z_map,percentage)
        # print(z_map)
        threshold=0
    plt.style.use('dark_background')
    if fig==None:
        fig=plt.figure(figsize=(6,6))
    # plt.figure(figsize=(6,6))
    # print(nilearn.masking.compute_epi_mask(mean_img(data),mask_slice).shape)

    background_img=mean_img(data)
    if plot==True:
        plotting.plot_stat_map(z_map,
                                bg_img=background_img,
                                threshold=threshold,
                                display_mode="z",
                                cut_coords=cut_coords,
                                black_bg=True,
                                figure=fig,
                                annotate=annotate,
                                colorbar=colorbar
                                )
    plt.show()
    plt.pause(0.005)
    
    return z_map,data
