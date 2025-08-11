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
# from fmri_utils_true import *
from skimage.metrics import structural_similarity as ssim

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

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def transform_kspace_to_image(k, dim=None, img_shape=None):
    if k.ndim==3: #real or imag, h, w
        k_complex=k[0,:,:]+1j*k[1,:,:]
        img=np.fft.ifft2(k_complex)
    elif k.ndim==4: #batch, real or imag, h, w
        img=torch.zeros((k.shape[0],k.shape[2],k.shape[3]))
        for i in range(k.shape[0]):
            k_complex=k[i,0,:,:]+1j*k[i,1,:,:]
            img[i,:,:]=np.fft.ifft2(torch.squeeze(k_complex))
    return img

def transform_two_ch_image_to_kspace(img, dim=None, img_shape=None):
    if img.ndim==3:
        img_complex=two_ch_to_complex(img)
        k_complex=torch.fft.fft2(img_complex)
        
    elif img.ndim==4:
        batch_size=img.shape[0]
        variable_list=[]
        for i in range(batch_size):
            img_complex=two_ch_to_complex(img[i])
            var=torch.fft.fft2(img_complex)
            variable_list.append(var)
        k_complex=torch.stack(variable_list)
    
    return k_complex

def transform_image_to_kspace(img, dim=None, k_shape=None):
    if img.ndim==3:
        k=np.fft.fftshift(np.fft.fft2(img))
    elif img.ndim==4:
        batch_size=img.shape[0]
        variable_list=[]
        for i in range(batch_size):
            if torch.is_tensor(img[i])==True:
                var=torch.fft.fftshift(torch.fft.fft2(img[i]))
            else:
                var=np.fft.fftshift(np.fft.fft2(img[i]))
            variable_list.append(var)
        k=torch.stack(variable_list)
    return k

def check_img(img,cmap='gray',cbar=True,vmin=None,vmax=None,norm=False):
    fig, axes = plt.subplots(figsize=(13,8))
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

def make_path_list(dir_path):
    filenames = os.listdir(dir_path)
    file_list=[]
    for filename in filenames:
        if 'bold' in filename:
            full_filename = os.path.join(dir_path, filename)
            file_list.append(full_filename)
    return file_list

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

def data_aug_flip(input,dimension):
    input_f=torch.flip(input,dimension)
    output=torch.cat((input,input_f),-3)
    return output

def delete_slice(input,a,b):
    input_a=input[0:a]
    input_b=input[b:]
    return torch.cat((input_a,input_b),0)
    
def gaussian2d(array_size,center_x,center_y,amplitude,deviation,radius):
    gaussian2d_tensor=torch.zeros(array_size)

    for i in range(array_size[1]):
        for j in range(array_size[0]):
            index_distance=math.sqrt((center_x-i)*(center_x-i)+(center_y-j)*(center_y-j))
            
            if index_distance<radius:
                gaussian2d_tensor[j,i]=amplitude*math.exp(-index_distance**2 / deviation**2 / 2)
    return gaussian2d_tensor

def rect(array_size,center_x,center_y,amplitude,width,height):
    rect_tensor=torch.zeros(array_size)

    for i in range(array_size[1]):
        for j in range(array_size[0]):
    
            
            if abs(i-center_x)<width/2 and abs(j-center_y)<height/2:
                rect_tensor[j,i]=amplitude
    return rect_tensor

def points(array_size,coordinate_list,amplitude):
    point_tensor=torch.zeros(array_size)
    for coordinate in coordinate_list:
        point_tensor[coordinate[1],coordinate[0]]=amplitude
    return point_tensor

def testset_for_bold(GE_EPI,SE_EPI,x,y,n,abs,mod,mean=0,std=1,noise=None,mask=None,wave_form='hrf',coordinate_list=[],Tensor=False):
    var=n*10
    if wave_form=='hrf':
        mag=torch.squeeze(create_hrf(n))
    elif wave_form=='linear':
        mag=np.array(list(range(int(-var/2),int(var/2))))
        mag=mag/var
    if mod=='point':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)
        
        for j in range(var):
            test_input[j,y,x]=test_input[j,y,x]+abs*mag[j]
        if noise!=None:
            test_input=add_noise(test_input,noise,True)
        plt.plot(test_input[:,y,x])
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)
    elif mod=='gaussian2d':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        
        print(feas_input.shape)
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)

        for j in range(var):
            gaussian_mask=abs*gaussian2d(GE_EPI.shape,x,y,amplitude=mag[j],deviation=10,radius=4) #dev=6
            test_input[j,:,:]=test_input[j,:,:]+gaussian_mask
        if noise!=None:
            test_input=add_noise(test_input,noise,mean,std,True,True)
        plt.plot(test_input[:,y,x])
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        check_img(gaussian2d(GE_EPI.shape,x,y,amplitude=1,deviation=6,radius=4))
        print(gaussian_mask[y,x])
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)

    elif mod=='rect':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)

        for j in range(var):
            rect_mask=abs*rect(GE_EPI.shape,x,y,amplitude=mag[j],width=10,height=10)
            test_input[j,:,:]=test_input[j,:,:]+rect_mask
        if noise!=None:
            test_input=add_noise(test_input,noise,mean,std,True,True)
        plt.plot(test_input[:,y,x])
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        check_img(rect_mask)
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)

    
    elif mod=='stick':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)

        for j in range(var):
            rect_mask=abs*rect(GE_EPI.shape,x,y,amplitude=mag[j],width=2,height=32)
            test_input[j,:,:]=test_input[j,:,:]+rect_mask
        if noise!=None:
            test_input=add_noise(test_input,noise,mean,std,True,True)
        plt.plot(test_input[:,y,x])
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        check_img(rect_mask)
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)

    
    elif mod=='horizontal':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)

        for j in range(var):
            rect_mask=abs*rect(GE_EPI.shape,x,y,amplitude=mag[j],width=64,height=3)
            test_input[j,:,:]=test_input[j,:,:]+rect_mask
        if noise!=None:
            test_input=add_noise(test_input,noise,mean,std,True,True)
        plt.plot(test_input[:,y,x])
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        check_img(rect_mask)
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)

    
    elif mod=='full':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)
        for j in range(var):
            full_mask=abs*mag[j]*torch.ones(GE_EPI.shape)
            test_input[j,:,:]=test_input[j,:,:]+full_mask
        if noise!=None:
            test_input=add_noise(test_input,noise,mean,std,True)
        plt.plot(test_input[:,y,x])
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        check_img(full_mask)
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)

    
    elif mod=='points':
        feas_input=torch.tensor(GE_EPI.copy()) #34, 48, 4, value=26
        SE_EPI=torch.tensor(SE_EPI)
        test_input=[]
        test_label=[]
        for i in range(var):
            test_input.append(feas_input)
            test_label.append(SE_EPI)
        test_input=torch.stack(test_input)
        test_label=torch.stack(test_label)

        for j in range(var):
            point_mask=abs*points(GE_EPI.shape,coordinate_list,amplitude=mag[j])
            test_input[j,:,:]=test_input[j,:,:]+point_mask
        if noise!=None:
            test_input=add_noise(test_input,noise,mean,std,True,True)
        print(coordinate_list[0])
        plt.plot((test_input[:,coordinate_list[0][1],coordinate_list[0][0]]-test_input[40,coordinate_list[0][1],coordinate_list[0][0]])/torch.max(test_input))
        plt.xlabel('Measurement')
        plt.ylabel('Magnitude')
        if isinstance(mask, type(None))==False:
            for slices in range(test_input.shape[0]):
                test_input[slices,:,:]=test_input[slices,:,:]*mask
        check_img(point_mask,cbar=False)
        test_input=torch.unsqueeze(test_input,dim=1).float()
        test_label=torch.unsqueeze(test_label,dim=1).float()
        if Tensor==True:
            return (test_input,test_label)
        return TensorDataset(test_input,test_label)


    
def weight_init_xavier_uniform(submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(submodule.weight)
        
def total_variation_loss(img, weight):

    if img.dim()==2:
        tv_h = torch.pow(img[1:,:]-img[:-1,:], 2).sum()
        tv_w = torch.pow(img[:,1:]-img[:,:-1], 2).sum()
    else:
        tv_h = torch.pow(img[:,1:,:]-img[:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,1:]-img[:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(math.prod(img.size()))

def dataset_to_tensor(dataset):
    list=[]
    for idx, (x) in enumerate(dataset):
        list.append(x[0])
    return torch.stack(list)

def train_model(model, traindataset, optimizer, loss_function_k, loss_function_img, device, norm_scale, writer=None, epoch_number=None,noise=None,frequency_loss=True, image_loss=False,l1norm=None,tv_loss=None,tv_loss_f=None):
    total=0
    total_loss=0
    total_loss2=0
    for idx, (x,y) in enumerate(traindataset):
        model.train()

        # input_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(x))/torch.max(x)).to(device)
        input_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(x))/norm_scale).to(device)
        if noise!=None:
            input_k_2ch=add_noise(input_k_2ch,0,3,noise,device=device)
        # target_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(y))/torch.max(y)).to(device)
        target_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(y))/norm_scale).to(device)
        input=input_k_2ch.to(device)
        target=target_k_2ch.to(device)
               
        predict=model(input)
        if image_loss==True:
            input_i_2ch=(x/norm_scale).to(device)
            # target_i_2ch=(y/norm_scale).to(device)
            target_img=y.to(device)/norm_scale
            predict_i_2ch=torch.fft.ifft2(two_ch_to_complex(predict))
            loss2=loss_function_img(torch.abs(predict_i_2ch), target_img)
            total_loss2+=loss2.item()
        
        loss=0
        if frequency_loss==True:
            loss1=loss_function_k(predict, target)
            total_loss+=loss1.item()
            loss+=loss1
            
        if image_loss==True:
            loss+=loss2
        if l1norm!=None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l1_loss = l1norm * l1_norm
            loss+=l1_loss
        if tv_loss!=None:
            predict_i_2ch=torch.fft.ifft2(two_ch_to_complex(predict))
            loss+=total_variation_loss(torch.sub(torch.abs(predict_i_2ch),y.to(device)/norm_scale,alpha=1), tv_loss)
        if tv_loss_f!=None:
            loss+=total_variation_loss(torch.sub(predict,torch.fft.fftshift(torch.fft.fft2(y.to(device)))/norm_scale,alpha=1), tv_loss_f)
            
        total+=x.shape[0]
        loss.backward(retain_graph=True)
        
        optimizer.step()
        optimizer.zero_grad()
        
    return total, total_loss

def eval_model(model, valdataset, optimizer, loss_function_k, loss_function_img, device, norm_scale, writer=None, epoch_number=None, k_plot=False, img_plot=False, print_ssim=False):
    total=0
    total_ssim=0
    for idx, (x,y) in enumerate(valdataset):
        # print(idx)
        model.eval()
        input_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(x))/norm_scale)
        target_k_2ch=complex_to_two_ch(torch.fft.fftshift(torch.fft.fft2(y))/norm_scale)
        
        input_i_2ch=x/norm_scale
        target_i_2ch=y/norm_scale

        input=input_k_2ch.to(device)
        target=target_k_2ch.to(device)
        model.to(device)
        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=two_ch_to_complex(predict_test[0,:,:,:])
        
        image_plot=np.squeeze(np.fft.ifft2(predict_img))

        orig_plot=y[0,:,:,:]
        img1=np.squeeze(image_plot*norm_scale)
        img2=np.squeeze(orig_plot.numpy())
        (score, diff) = ssim(abs(img1/np.max(img1)), abs(img2/np.max(img2)), full=True)
        total+=x.shape[0]
        total_ssim+=score
        
        if k_plot==True:
            fig, axes = plt.subplots(1,3)
            axes[0].set_title('input k-space')
            axes[0].imshow(torch.log(torch.abs(torch.squeeze(torch.fft.fftshift(torch.fft.fft2(x/norm_scale))))), cmap='gray')
            axes[1].set_title('output k-space')
            axes[1].imshow(np.log(np.abs(np.squeeze(two_ch_to_complex(predict_test)))), cmap='gray')
            axes[2].set_title('label k-space')
            axes[2].imshow(torch.log(torch.abs(torch.squeeze(torch.fft.fftshift(torch.fft.fft2(y/norm_scale))))), cmap='gray')
            plt.pause(0.05)
            fig.show()
        if img_plot==True:
            fig, axes = plt.subplots(2,2)
            axes[0,0].set_title('input image')
            axes[0,0].axis('off')
            img0=axes[0,0].imshow(torch.abs(torch.squeeze(x)), cmap='gray')
            fig.colorbar(img0,ax=axes[0,0])
            axes[0,1].set_title('output image')
            axes[0,1].axis('off')
            img1=axes[0,1].imshow(np.abs(image_plot)*norm_scale, cmap='gray')
            fig.colorbar(img1,ax=axes[0,1])
            axes[1,0].set_title('label image')
            axes[1,0].axis('off')
            img2=axes[1,0].imshow(torch.abs(torch.squeeze((y))), cmap='gray')
            fig.colorbar(img2,ax=axes[1,0])
            axes[1,1].set_title('Difference map')
            axes[1,1].axis('off')
            img3=axes[1,1].imshow((torch.tensor(np.abs(image_plot)*norm_scale)-torch.abs(torch.squeeze((y)))), cmap='cool')

            fig.colorbar(img3,ax=axes[1,1])
            plt.pause(0.05)
            fig.show()
        if print_ssim==True:
            print(f'SSIM Accuracy of the network on the images: {score} %')
        
        return total, total_ssim
    
def eval_model_test2(model, valdataset, optimizer, loss_function_k, loss_function_img, device, norm_scale, writer, epoch_number, k_plot, img_plot, print_ssim):
    total=0
    total_ssim=0
    for idx, (x,y) in enumerate(valdataset):
        # print(idx)
        model.eval()
        input_k_2ch=complex_to_two_ch(torch.fft.fft2(x)/norm_scale)
        target_k_2ch=complex_to_two_ch(torch.fft.fft2(y)/norm_scale)
        
        input_i_2ch=x/norm_scale
        target_i_2ch=y/norm_scale

        input=input_k_2ch.to(device)
        target=target_k_2ch.to(device)
        model.to(device)
        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=two_ch_to_complex(predict_test[0,:,:,:])
        
        image_plot=np.squeeze(np.fft.ifft2(predict_img))

        orig_plot=y[0,:,:,:]

        (score, diff) = ssim(abs(np.squeeze(image_plot*norm_scale)), abs(np.squeeze(orig_plot.numpy())), full=True)
        total+=x.shape[0]
        total_ssim+=score
        
        if k_plot==True:
            fig, axes = plt.subplots(1,3)
            axes[0].set_title('input k-space')
            axes[0].imshow(torch.log(torch.abs(torch.squeeze(torch.fft.fft2(x/norm_scale)))), cmap='gray')
            axes[1].set_title('output k-space')
            axes[1].imshow(torch.log(np.abs(np.squeeze(two_ch_to_complex(predict_test)))), cmap='gray')
            axes[2].set_title('label k-space')
            axes[2].imshow(torch.log(torch.abs(torch.squeeze(torch.fft.fft2(y/norm_scale)))), cmap='gray')
            
            # axes[0].imshow(torch.abs(torch.squeeze(torch.fft.fft2(x/norm_scale))), cmap='gray',vmin=0,vmax=5000/norm_scale)
            # axes[1].set_title('output k-space')
            # axes[1].imshow(np.abs(np.squeeze(two_ch_to_complex(predict_test))), cmap='gray',vmin=0,vmax=5000/norm_scale)
            # axes[2].set_title('label k-space')
            # axes[2].imshow(torch.abs(torch.squeeze(torch.fft.fft2(y/norm_scale))), cmap='gray',vmin=0,vmax=5000/norm_scale)
            
            plt.pause(0.05)
            fig.show()
        if img_plot==True:
            fig, axes = plt.subplots(2,2)
            axes[0,0].set_title('input image')
            axes[0,0].axis('off')
            img0=axes[0,0].imshow(torch.abs(torch.squeeze(x)), cmap='gray')
            fig.colorbar(img0,ax=axes[0,0])
            axes[0,1].set_title('output image')
            axes[0,1].axis('off')
            img1=axes[0,1].imshow(np.abs(image_plot)*norm_scale, cmap='gray')
            fig.colorbar(img1,ax=axes[0,1])
            axes[1,0].set_title('label image')
            axes[1,0].axis('off')
            img2=axes[1,0].imshow(torch.abs(torch.squeeze((y))), cmap='gray')
            fig.colorbar(img2,ax=axes[1,0])
            axes[1,1].set_title('Difference map')
            axes[1,1].axis('off')
            img3=axes[1,1].imshow((torch.tensor(np.abs(image_plot)*norm_scale)-torch.abs(torch.squeeze((y)))), cmap='cool')

            fig.colorbar(img3,ax=axes[1,1])
            plt.pause(0.05)
            fig.show()
        if print_ssim==True:
            print(f'SSIM Accuracy of the network on the images: {score} %')
        
        return total, total_ssim

def train_model_image(model, traindataset, optimizer, loss_function, device, norm_scale):
    total=0
    total_loss=0
    model.train()
    
    for idx, (x,y) in enumerate(traindataset):
        # print(model.parameters().get_device())
        model.to(device)
        input=x.to(device)/norm_scale
        target=y.to(device)/norm_scale
        predict=model(input)
        
        loss1=loss_function(predict, target)
        total_loss+=loss1.item()
        loss=loss1
        
        total+=x.shape[0]
        loss.backward(retain_graph=True)
        
        optimizer.step()
        optimizer.zero_grad()
        
    return total, total_loss


def eval_model_image(model, valdataset, device, norm_scale, k_plot, img_plot, print_ssim):
    total=0
    total_ssim=0
    for idx, (x,y) in enumerate(valdataset):
        model.eval()
        input=(x/norm_scale).to(device)
        target=(y/norm_scale).to(device)

        predict_test=model(input)
        predict_test=predict_test.cpu()
        predict_test=predict_test.detach().numpy()

        predict_img=predict_test[0,:,:,:]
        
        image_plot=np.squeeze(predict_img)

        orig_plot=y[0,:,:,:]
        print(image_plot.shape)
        print(orig_plot.shape)
        (score, diff) = ssim(abs(np.squeeze(image_plot*norm_scale)), abs(np.squeeze(orig_plot.numpy())), full=True)
        total+=x.shape[0]
        total_ssim+=score
        
        if k_plot==True:
            fig, axes = plt.subplots(1,3)
            axes[0].set_title('input k-space')
            axes[0].imshow(torch.abs(torch.squeeze(torch.fft.fftshift(torch.fft.fft2(x/norm_scale)))), cmap='gray')
            axes[1].set_title('output k-space')
            axes[1].imshow(np.abs(np.squeeze(np.fft.fftshift(np.fft.fft2(predict_test)))), cmap='gray')
            axes[2].set_title('label k-space')
            axes[2].imshow(torch.abs(torch.squeeze(torch.fft.fftshift(torch.fft.fft2(y/norm_scale)))), cmap='gray')
            plt.pause(0.05)
            fig.show()
        if img_plot==True:
            fig, axes = plt.subplots(2,2)
            axes[0,0].set_title('input image')
            axes[0,0].axis('off')
            img0=axes[0,0].imshow(torch.abs(torch.squeeze(x)), cmap='gray')
            fig.colorbar(img0,ax=axes[0,0])
            axes[0,1].set_title('output image')
            axes[0,1].axis('off')
            img1=axes[0,1].imshow(np.abs(image_plot)*norm_scale, cmap='gray')
            fig.colorbar(img1,ax=axes[0,1])
            axes[1,0].set_title('label image')
            axes[1,0].axis('off')
            img2=axes[1,0].imshow(torch.abs(torch.squeeze((y))), cmap='gray')
            fig.colorbar(img2,ax=axes[1,0])
            axes[1,1].set_title('Difference map')
            axes[1,1].axis('off')
            img3=axes[1,1].imshow((torch.tensor(np.abs(image_plot)*norm_scale)-torch.abs(torch.squeeze((y)))), cmap='cool')

            fig.colorbar(img3,ax=axes[1,1])
            plt.pause(0.05)
            fig.show()
        if print_ssim==True:
            print(f'SSIM Accuracy of the network on the images: {score} %')
        
        return total, total_ssim
    
def simple_normalize(input,scale,mean=None):
    if mean!=None:
        input_zero_mean=input-mean
    else:
        input_zero_mean=input-torch.mean(input)
    input_zero_mean_scaled=input_zero_mean/scale
    return input_zero_mean_scaled
        