import sys
sys.path.append('/home/milab/SSD_8TB/LeeSooHyung/b0inhomo/dephase_simulator/epi_feasibility_training/epi_20block/utils')

import os
import torch
import numpy as np
import torch.nn.functional as F
import math
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import  TensorDataset, DataLoader
import scipy.io
import cmath                

from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d,gaussian_filter
from PIL import Image
import torchvision.transforms as transforms

import nibabel as nib
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
from nilearn.image import concat_imgs, mean_img, resample_img
import nilearn
import pandas
from nifti_utils import *
# from common import *


def check_img(img,cmap='gray',cbar=True,vmin=None,vmax=None,norm=False,rotate=False):
    fig, axes = plt.subplots(figsize=(13,8))
    if rotate==True:
        img=np.flip(np.swapaxes(img,0,1),0)
    if norm==True:
        image=img/np.max(img)
    else:
        image=img
    if vmin != None:
        imgplot=plt.imshow(abs(image), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        imgplot=plt.imshow(abs(image), cmap=cmap)
    
    plt.axis('off')
    if cbar==True:
        plt.colorbar(imgplot)
    plt.tight_layout
    plt.show()
    
def check_signal(model,coordinate,testdataset,device,norm_scale,gaussian_blur_2d=None,gaussian_blur_1d=None):
    list=[]
    for idx, (x,y) in enumerate(testdataset):
        model.eval()
        input_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(x))/norm_scale)
        target_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(y))/norm_scale)
        
        input_i_2ch=x/norm_scale
        target_i_2ch=y/norm_scale

        input=input_k_2ch.to(device)
        target=target_k_2ch.to(device)
        
        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=two_ch_to_complex(predict_test[0,:,:,:])
        
        image_plot=np.squeeze(np.fft.ifft2(predict_img))

        orig_plot=y[0,:,:,:]
        if gaussian_blur_2d!=None:
            image_plot=gaussian_filter(image_plot,gaussian_blur_2d)
        list.append(image_plot[coordinate[1],coordinate[0]]*norm_scale)
    list=list-list[int(len(testdataset)/2)]
    if gaussian_blur_1d!=None:
        list=gaussian_filter1d(np.array(list),gaussian_blur_1d)
    plt.plot(list)
    
def check_signal_image(model,coordinate,testdataset,device,norm_scale,gaussian_blur_2d=None,gaussian_blur_1d=None):
    list=[]
    for idx, (x,y) in enumerate(testdataset):
        model.eval()
        
        input=(x/norm_scale).to(device)
        target=(y/norm_scale).to(device)

        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=predict_test[0,:,:,:]
        
        image_plot=np.squeeze(predict_img)
        if gaussian_blur_2d!=None:
            image_plot=gaussian_filter(image_plot,gaussian_blur_2d)
        list.append(image_plot[coordinate[1],coordinate[0]]*norm_scale)
    list=list-list[int(len(testdataset)/2)]
    if gaussian_blur_1d!=None:
        list=gaussian_filter1d(np.array(list),gaussian_blur_1d)
    plt.plot(list)
    
def add_noise(input,constant,mean=0,std=1,positive=False,check=False,device=None):
    signal_power=torch.norm(input)
    
    if device!=None:
        noise = constant*signal_power*torch.tensor(np.random.normal(mean, std, input.size()), dtype=torch.float).to(device)
    else:    
        noise = constant*signal_power*torch.tensor(np.random.normal(mean, std, input.size()), dtype=torch.float)
    result=noise+input
    if check==True:
        print(torch.max(noise))
    if positive==True:
        result=torch.clamp(result,min=0)
    return result

def testset_output(model,testdataset,device,norm_scale,noise=None):
    list=[]
    model.to(device)
    for idx, (x) in enumerate(testdataset):
        model.eval()
        input_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(x[0]))/norm_scale)
        if noise!=None:
            input_k_2ch=add_noise(input_k_2ch,0,3,noise,device=device)
        input_i_2ch=x[0]/norm_scale

        input=input_k_2ch.to(device)

        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=two_ch_to_complex(predict_test[0,:,:,:])
        
        image_plot=np.squeeze(np.fft.ifft2(predict_img))

        list.append(torch.tensor(image_plot*norm_scale))
    return torch.stack(list)

def testset_output_2(model,testdataset,device,norm_scale,noise=None):
    list=[]
    model.to(device)
    for idx, (x) in enumerate(testdataset):
        model.eval()
        input_k_2ch=complex_to_two_ch(torch.fft.fft2(x[0])/norm_scale)
        if noise!=None:
            input_k_2ch=add_noise(input_k_2ch,0,3,noise,device=device)
        input_i_2ch=x[0]/norm_scale

        input=input_k_2ch.to(device)

        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=two_ch_to_complex(predict_test[0,:,:,:])
        
        image_plot=np.squeeze(np.fft.ifft2(predict_img))

        list.append(torch.tensor(image_plot*norm_scale))
    return torch.stack(list)

def testset_output_new(model1,model2,testdataset,device,norm_scale,noise=None):
    list=[]
    model1.to(device)
    model2.to(device)
    for idx, (x,y,z) in enumerate(testdataset):
        model1.eval()
        model2.eval()
        input_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(x[0]))/norm_scale)
        if noise!=None:
            input_k_2ch=add_noise(input_k_2ch,0,3,noise,device=device)
        input_i_2ch=x[0]/norm_scale

        input=input_k_2ch.to(device)

        predict_test1=model1(input)
        predict_test2=model2(predict_test1)
        predict_test=predict_test2.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=two_ch_to_complex(predict_test[0,:,:,:])
        
        image_plot=np.squeeze(np.fft.ifft2(predict_img))

        list.append(torch.tensor(image_plot*norm_scale))
    return torch.stack(list)


def testset_output_image(model,testdataset,device,norm_scale):
    list=[]
    for idx, (x) in enumerate(testdataset):
        model.eval()
        input=(x/norm_scale).to(device)

        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=predict_test[0,:,:,:]
        
        image_plot=np.squeeze(predict_img)

        list.append(torch.tensor(image_plot*norm_scale))
    return torch.stack(list)

def testset_output_image(model,testdataset,device,norm_scale):
    list=[]
    for idx, (x) in enumerate(testdataset):
        model.eval()
        input=(x[0]/norm_scale).to(device)

        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=predict_test[0,:,:,:]
        image_plot=np.squeeze(predict_img)

        list.append(torch.tensor(image_plot*norm_scale))
    return torch.stack(list)

def cut_topvalue(input,percent):
    
    if torch.is_tensor(input)==True:
        vector=torch.flatten(input)
        shape=vector.shape
        sorted,indices=vector.sort()
        threshold_value=sorted[int(percent*shape[0])]
        clamped=torch.clamp(input, min=threshold_value)
        
    else:
        vector=np.ndarray.flatten(input)
        shape=vector.shape
        sorted=np.sort(vector)
        threshold_value=sorted[int(percent*shape[0])]
        clamped=np.clip(input, a_min=threshold_value,a_max=None)
        clamped[clamped == threshold_value]=0
        
    return clamped


def calculate_linear_regression(bold):
    array=abs(bold).float()
    output=torch.zeros((array.shape[1],array.shape[2]))
    x=range(1,bold.shape[0]+1)
    for i in range(array.shape[2]):
        for j in range(array.shape[1]):
            y=array[:,j,i]
            linearmodel=LinearRegression()
            value=linearmodel.fit(np.array(x).reshape(-1,1),y).coef_
            output[j,i]=value[0]
    return output
    

def complex_to_two_ch(input):
    ###non batch(3d)###
    if input.ndim==2:
        real_layer=input.real
        imag_layer=input.imag

        
        if torch.is_tensor(input):
            output=torch.stack((real_layer,imag_layer),axis=0)
        else:
            output=np.stack((real_layer,imag_layer),axis=0)
    
    elif input.ndim==3:
        real_layer=input.real
        imag_layer=input.imag
        
        if torch.is_tensor(input):
            output=torch.stack((real_layer,imag_layer),axis=1)    
        else:
            output=np.stack((real_layer,imag_layer),axis=1)
            
    elif input.ndim==4:
        real_layer=input.real
        imag_layer=input.imag
        
        if torch.is_tensor(input):
            output=torch.cat((real_layer,imag_layer),axis=1)    
        else:
            output=np.cat((real_layer,imag_layer),axis=1)
    return output

def two_ch_to_complex(input,squeeze=True):
    if input.ndim==3:
        complex_var=input[0,:,:]+1j*input[1,:,:]
    elif input.ndim==4:
        complex_var=input[:,0,:,:]+1j*input[:,1,:,:]
    if squeeze==True:
        if torch.is_tensor(input):
            complex_var=torch.squeeze(complex_var)
        else:
            complex_var=np.squeeze(complex_var)    
    return complex_var

def make_mask_sliced(mask,slice_number):
    if mask.get_fdata()[:,:,slice_number].any()!=0:
        return nib.Nifti1Image(np.expand_dims(mask.get_fdata()[:,:,slice_number], 2),affine=np.eye(4))
    else:
        return nib.Nifti1Image(np.expand_dims(np.ones(mask.get_fdata()[:,:,slice_number].shape), 2),affine=np.eye(4))

def make_mask_sliced_direct(data,slice_number):
    mask=nilearn.masking.compute_epi_mask(data)
    
    return make_mask_sliced(mask,slice_number)


def make_nii_data_sliced(data,slice_number):
    data_1=subject_data.get_fdata()[:,:,26,:]
    data=nib.Nifti1Image(np.expand_dims(data_1, 2),affine=np.eye(4))    
    return data

def make_mask_by_threshold(nii, slice_num, threshold=100):
    from nilearn.image import math_img
    mask_img = math_img(f'img > {threshold}', img=nii)
    mask_img=make_mask_sliced(mask_img,slice_num)
    return mask_img

def calculate_glm(data,mask_slice, contrast_id = 'incong-off',tr = 3, output=None, percentage=None, threshold=5,cut_coords=3,mod=80,highpass=0.01, masking=False, plot=True,annotate=False,colorbar=True,fig=None,ax=None,plot_design=False):
    
    tr = 3  # repetition time is 1 second
    n_scans = mod  # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times

    # these are the types of the different trials
    # conditions = ["off", "cong", "off", "incong", "off", "neutral", "off", "cong", "off", "incong", "off", "neutral", "off", "cong", "off", "incong", "off", "neutral","off"]
    # duration = [20, 30, 20, 30, 20, 30, 20, 30, 20, 30, 20, 30, 20, 30, 20, 30, 20, 30, 30]
    # onsets = [0, 20, 50, 70, 100, 120, 150, 170, 200, 220, 250, 270, 300, 320, 350, 370, 400, 420, 450]
    conditions = ["on", "on"]
    duration = [60, 60]
    onsets = [60, 180]

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
    if plot_design==True:
        nilearn.plotting.plot_design_matrix(X1, rescale=True, ax=None, output_file=None)
    # duration = 7.0 * np.ones(len(conditions))
    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    fmri_glm = FirstLevelModel(mask_img=mask_slice,high_pass=highpass,drift_order=3)
    fmri_glm = fmri_glm.fit(data, design_matrices=X1)

    contrast_matrix = np.eye(X1.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(X1.columns)
    }
    # basic_contrasts['cong-off'] = (basic_contrasts['cong'] - basic_contrasts['off'])
    # basic_contrasts['incong-off'] = (basic_contrasts['incong'] - basic_contrasts['off'])
    # basic_contrasts['neutral-off'] = (basic_contrasts['neutral'] - basic_contrasts['off'])
    # basic_contrasts['cong-neutral'] = (basic_contrasts['cong'] - basic_contrasts['neutral'])
    # basic_contrasts['incong-neutral'] = (basic_contrasts['incong'] - basic_contrasts['neutral'])
    # basic_contrasts['color-neutral'] = (basic_contrasts['cong'] + basic_contrasts['incong']-basic_contrasts['neutral'])

    z_map = fmri_glm.compute_contrast(X1.columns[0], output_type="stat")

    # z_map = fmri_glm.compute_contrast(basic_contrasts[contrast_id], output_type="z_score")
    if percentage!=None:
        z_map=cutoff_nii(z_map,percentage)
        # print(z_map)
        threshold=0
    plt.style.use('dark_background')
    if fig==None:
        fig=plt.figure(figsize=(6,6))
    # plt.figure(figsize=(6,6))
    # print(nilearn.masking.compute_epi_mask(mean_img(data),mask_slice).shape)
    if masking==True:
        background_img=apply_mask(mean_img(data),mask_slice)
    else: 
        background_img=mean_img(data)
    if plot==True:
        plotting.plot_stat_map(z_map,
                               output_file=output,
                                bg_img=background_img,
                                threshold=threshold,
                                display_mode="z",
                                cut_coords=cut_coords,
                                black_bg=True,
                                cmap='cold_hot',
                                figure=fig,
                                annotate=annotate,
                                colorbar=colorbar
                                )
    if plot=='glass':
        plotting.plot_glass_brain(z_map,
                                    title="plot_glass_brain",
                                    black_bg=True,
                                    display_mode="xz",
                                    threshold=threshold,
                                )
    plotting.show()
    # # z_map = fmri_glm.compute_contrast(X1.columns[0], output_type="z_score")
    # if percentage!=None:
    #     z_map=cutoff_nii(z_map,percentage)
    #     # print(z_map)
    #     threshold=0
    # plt.style.use('dark_background')
    # if fig==None:
    #     fig=plt.figure(figsize=(6,6))
    # # plt.figure(figsize=(6,6))
    # # print(nilearn.masking.compute_epi_mask(mean_img(data),mask_slice).shape)
    # if masking==True:
    #     background_img=apply_mask(mean_img(data),mask_slice)
    # else: 
    #     background_img=mean_img(data)
    # if plot==True:
        
    #     plotting.plot_stat_map(z_map,
    #                             bg_img=background_img,
    #                             threshold=threshold,
    #                             display_mode="z",
    #                             cut_coords=cut_coords,
    #                             black_bg=True,
    #                             figure=fig,
    #                             annotate=annotate,
    #                             colorbar=colorbar,
    #                             title=contrast_id
    #                             )
    # plt.show()
    plt.pause(0.005)
    
    return z_map,data
    
def create_design_matrix(tr, n_scans, a=0, csv=None):
    tr = 2  # repetition time is 1 second
    n_scans = n_scans  # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
    if csv==None:
        # these are the types of the different trials
        conditions = ["c0", "c0", "c0", "c0", "c0", "c0", "c0", "c0"]
        duration = [22, 22, 22, 22, 22, 22, 22, 36]
        onsets = [18 + a, 58 + a, 98 + a, 138 + a, 178 + a, 218 + a, 258 + a, 298]
            

        events = pd.DataFrame(
            {"trial_type": conditions, "onset": onsets, "duration": duration}
        )
    else:
        selected_columns = ["emotion", "correct_stim_onset_time", "duration"]
        selected_data = csv[selected_columns].copy()
        selected_data['emotion'] = selected_data['emotion'].astype(str).replace({'rest': 'off', '0': 'on', '1': 'on'})

    hrf_model = "glover"
    X1 = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        hrf_model=hrf_model,
    )
    return X1


def calculate_glm_emotion(data,mask_slice, bg_image = None,fwhm=None, csv=None,a=0, percentage=None, threshold=5,cut_coords=3,mod=178, masking=False, plot=True,annotate=False,colorbar=True,fig=None,axes=None,output=None):
    
    X1=create_design_matrix(tr=2, n_scans=mod, a=0, csv=csv)
    

    # # duration = 7.0 * np.ones(len(conditions))
    # events = pd.DataFrame(
    #     {"trial_type": conditions, "onset": onsets, "duration": duration}
    # )

    fmri_glm = FirstLevelModel(mask_img=mask_slice,high_pass=0,drift_order=3)
    if fwhm==None:
        fmri_glm = fmri_glm.fit(data, design_matrices=X1)
    else:
        smooth_data = nilearn.image.smooth_img(data,fwhm)
        fmri_glm = fmri_glm.fit(smooth_data, design_matrices=X1)

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
    if masking==True:
        background_img=apply_mask(mean_img(data),mask_slice)
    else: 
        background_img=mean_img(data)

    if bg_image != None:
        background_img = bg_image
    if plot==True:
        plotting.plot_stat_map(z_map,
                               output_file=output,
                                bg_img=background_img,
                                threshold=threshold,
                                display_mode="z",
                                cut_coords=1,
                                black_bg=True,
                                cmap='cold_hot',
                                figure=fig,
                                axes=axes,
                                annotate=annotate,
                                colorbar=colorbar
                                )

    # plt.show()
    # plt.pause(0.005)
    
    return z_map,data


def create_design_matrix_gamble(tr, n_scans, a=0, csv=None):
    tr = 2  # repetition time is 1 second
    n_scans = n_scans  # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
    if csv==None:
        # these are the types of the different trials
        conditions = ["c0", "c0"]
        duration = [60,60]
        onsets = [0,120]
            

        events = pd.DataFrame(
            {"trial_type": conditions, "onset": onsets, "duration": duration}
        )
    else:
        selected_columns = ["emotion", "correct_stim_onset_time", "duration"]
        selected_data = csv[selected_columns].copy()
        selected_data['emotion'] = selected_data['emotion'].astype(str).replace({'rest': 'off', '0': 'on', '1': 'on'})

    hrf_model = "glover"
    X1 = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        hrf_model=hrf_model,
    )
    return X1


def calculate_glm_gamble(data,mask_slice,fwhm=None,csv=None,a=0, percentage=None, threshold=5,cut_coords=3,mod=178, masking=False, plot=True,annotate=False,colorbar=True,fig=None,axes=None,output=None):
    
    X1=create_design_matrix_gamble(tr=2, n_scans=mod, a=0, csv=csv)
    

    # # duration = 7.0 * np.ones(len(conditions))
    # events = pd.DataFrame(
    #     {"trial_type": conditions, "onset": onsets, "duration": duration}
    # )

    fmri_glm = FirstLevelModel(mask_img=mask_slice,high_pass=0,drift_order=3)
    if fwhm==None:
        fmri_glm = fmri_glm.fit(data, design_matrices=X1)
    else:
        smooth_data = nilearn.image.smooth_img(data,fwhm)
        fmri_glm = fmri_glm.fit(smooth_data, design_matrices=X1)
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
    if masking==True:
        background_img=apply_mask(mean_img(data),mask_slice)
    else: 
        background_img=mean_img(data)
    if plot==True:
        plotting.plot_stat_map(z_map,
                               output_file=output,
                                bg_img=background_img,
                                threshold=threshold,
                                display_mode="z",
                                cut_coords=cut_coords,
                                black_bg=True,
                                cmap='cold_hot',
                                figure=fig,
                                axes=axes,
                                annotate=annotate,
                                colorbar=colorbar
                                )
    plt.show()
    plt.pause(0.005)
    
    return z_map,data

    
    