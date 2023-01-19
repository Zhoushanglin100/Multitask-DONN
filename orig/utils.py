import os
import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def phase_func(phase_file, i_k=256):
    # i_k = 256 # interpolate step; larger number means for fine-grained interpolate function
    phase = pd.read_csv(phase_file, sep=',', header=None)
    #print('phase file shape', phase.values.shape)
    n_shape = phase.values.shape[0]
    t_n_sampling = phase.values[:,2].reshape(1, 1, n_shape)
    t_n_sampling = torch.Tensor(t_n_sampling)
    t_f1 = torch.nn.functional.interpolate(t_n_sampling, size=i_k, mode='linear', align_corners=True).view(-1)
    return t_f1

def intensity_func(amp_file, i_k=256):
    amp = pd.read_csv(amp_file, sep=',', header=None)
    #print(amp.values.shape)
    n_shape = amp.values.shape[0]
    a_n_sampling = amp.values[:,2].reshape(1, 1, n_shape)
    a_n_sampling = torch.Tensor(a_n_sampling) # sqrt the normlized one ?
    a_f1 = torch.nn.functional.interpolate(a_n_sampling, size=i_k, mode='linear', align_corners=True).view(-1)
    return a_f1


def get_phase(model,args):
    if args.get_phase:
        if not os.path.exists(args.model_save_path):
            assert(0), "folder (%s) of the saved model(s) does not exist" % args.model_save_path
        else:
            if not os.path.exists((args.model_save_path + str(args.start_epoch) + args.model_name)):
                assert(0), "model file (%s) of the saved model does not exist" % (args.model_save_path + str(args.start_epoch) + args.model_name)
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))
        for name, param_ in model.named_parameters():
            if "voltage_" in name:
                param = torch.nn.functional.gumbel_softmax(param_,tau=2, hard=True)
                print(name, param.shape, param)
                print("phase at %s with max = %.4f, min = %.4f, mean = %.4f" % (name, torch.max(param).item(), torch.min(param).item(), torch.mean(param).item()))
                #print(param.cpu().detach().numpy()) # needed if you want to convert phase tensor into numpy
        print('Model : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
        exit(0)
    else:
        assert(args.get_phase), "get_phase is False something very wrong"
        exit(0)

def WeightClipper(model):
    # filter the variables to get the ones you want
    count = 0
    for p in model.phase:
        mask = p % (2*np.pi)
        phase_bound = torch.empty(p.shape).fill_(np.pi)
        count += torch.gt(mask.cpu().float(), np.pi).float().sum()
    print(count)
    return model

def data_to_cplex_seg(batch, device, input_padding=0):
    images = batch[0].to(device)  # (batch, 1, args.sys_size, args.sys_size) float32 1. 0.
    labels = batch[1].to(device).reshape(batch[1].shape[0],batch[1].shape[-1],batch[1].shape[-1])  # (batch, args.sys_size, args.sys_size) same dim as input
    images = torch.nn.functional.pad(images, pad=(input_padding,input_padding,input_padding,input_padding))
    images = torch.squeeze(torch.cat((images.unsqueeze(-1),
                    torch.zeros_like(images.unsqueeze(-1))), dim=-1), dim=1)
    return torch.view_as_complex(images), labels

def data_to_cplex_seg_rgb(batch, device, input_padding=0):
    assert(device)
    images = batch[0].to(device)
    print(images.shape)
    assert(images.shape[1]==3)
    images_r = images[:,0]
    images_g = images[:,1]
    images_b = images[:,2]

    images_r = torch.nn.functional.pad(images_r, pad=(input_padding,input_padding,input_padding,input_padding))
    images_g = torch.nn.functional.pad(images_g, pad=(input_padding,input_padding,input_padding,input_padding))
    images_b = torch.nn.functional.pad(images_b, pad=(input_padding,input_padding,input_padding,input_padding))

    images_r = torch.squeeze(torch.cat((images_r.unsqueeze(-1),torch.zeros_like(images_r.unsqueeze(-1))), dim=-1), dim=1)
    images_g = torch.squeeze(torch.cat((images_g.unsqueeze(-1),torch.zeros_like(images_g.unsqueeze(-1))), dim=-1), dim=1)
    images_b = torch.squeeze(torch.cat((images_b.unsqueeze(-1),torch.zeros_like(images_b.unsqueeze(-1))), dim=-1), dim=1)

    labels = batch[1].to(device).reshape(batch[1].shape[0],batch[1].shape[-1],batch[1].shape[-1])  # (batch, args.sys_size, args.sys_size) same dim as input 
    return torch.view_as_complex(images_r), torch.view_as_complex(images_g), torch.view_as_complex(images_b), labels



def data_to_cplex(batch, device, num_classes=10, input_padding=0, reverse_onehot=False, save_inputs=False):
    #print("testing data mapping")
    assert(device)
    images = batch[0].to(device)  # (64, 1, args.sys_size, args.sys_size) float32 1. 0.
    labels_ = batch[1].to(device)  # int64 9 0
    images = torch.nn.functional.pad(images, pad=(input_padding,input_padding,input_padding,input_padding))
    labels = torch.nn.functional.one_hot(labels_, num_classes=10).float()

    if reverse_onehot:
        labels = (labels + 1) % 2
    images = torch.squeeze(torch.cat((images.unsqueeze(-1),
                    torch.zeros_like(images.unsqueeze(-1))), dim=-1), dim=1)
    if save_inputs:
        for i in range(images.shape[0]):
            filename = "input_" + str(i)+"_"+str(labels_[i].cpu().item())  + ".npy"
            print(images[i].shape)
            with open(filename, 'wb') as f:
                np.save(filename, images[i].cpu())
        assert(0), "currently only save the first batch"
    return torch.view_as_complex(images), labels

def data_to_cplex_rgb(batch, device, num_classes=10, input_padding=0, reverse_onehot=False, save_inputs=False):
    #print("testing data mapping")
    assert(device)
    images = batch[0].to(device)  # (64, 1, args.sys_size, args.sys_size) float32 1. 0.
    assert(images.shape[1]==3)
    images_r = images[:,0]
    images_g = images[:,1]
    images_b = images[:,2]
    print(images_r.shape, images_g.shape, images_b.shape)
    labels_ = batch[1].to(device)  # int64 9 0
    images_r = torch.nn.functional.pad(images_r, pad=(input_padding,input_padding,input_padding,input_padding))
    images_g = torch.nn.functional.pad(images_g, pad=(input_padding,input_padding,input_padding,input_padding))
    images_b = torch.nn.functional.pad(images_b, pad=(input_padding,input_padding,input_padding,input_padding))
    labels = torch.nn.functional.one_hot(labels_, num_classes=10).float()
    if reverse_onehot:
        labels = (labels + 1) % 2
    images_r = torch.squeeze(torch.cat((images_r.unsqueeze(-1),torch.zeros_like(images_r.unsqueeze(-1))), dim=-1), dim=1)
    images_g = torch.squeeze(torch.cat((images_g.unsqueeze(-1),torch.zeros_like(images_g.unsqueeze(-1))), dim=-1), dim=1)
    images_b = torch.squeeze(torch.cat((images_b.unsqueeze(-1),torch.zeros_like(images_b.unsqueeze(-1))), dim=-1), dim=1)
    if save_inputs:
        assert(0), "not implemented yet"
        for i in range(images.shape[0]):
            filename = "input_" + str(i)+"_"+str(labels_[i].cpu().item())  + ".npy"
            print(images[i].shape)
            with open(filename, 'wb') as f:
                np.save(filename, images[i].cpu())
        assert(0), "currently only save the first batch"
    return torch.view_as_complex(images_r), torch.view_as_complex(images_g), torch.view_as_complex(images_b), labels



def phase_visualization(phase_list, size, cmap='Greys', idx=0, fname=""):
    n = len(phase_list)
    size = size
    fig, axs = plt.subplots(1, n)
    fig.set_figheight(22)
    fig.set_figwidth(22)
    for i in range(n):
        ax = axs[i]
        ax.imshow(phase_list[i].cpu().detach().numpy(),cmap=cmap,vmax=phase_list[i].cpu().detach().numpy().max()*1,
                vmin=phase_list[i][idx].abs().cpu().detach().numpy().min()*1)
        ax.set_axis_off()

    plt.savefig(fname,bbox_inches = 'tight', pad_inches = 0)

def forward_func_visualization2(intensity_list, size, cmap='Greys', idx=0, fname="forward_func_visualization.pdf",
            save_input=False, det_box=False):
    n = len(intensity_list)
    for i in range(n):
        size = size
        fig,axs = plt.subplots(1, 1)
        #fig.set_figheight(22)
        #fig.set_figwidth(22)

        if i==0:
            ax = axs
            print(n, intensity_list[0][idx].real.cpu().shape)
            if save_input:
                with open(fname+"_img.npy", 'wb') as f:
                    np.save(f, intensity_list[0][idx].real.cpu().detach().numpy())
            ax.imshow(intensity_list[0][idx].real.cpu().reshape(size,size),cmap="Greys")
            ax.set_axis_off()
            plt.savefig(fname+"_%d.pdf" % int(i),bbox_inches = 'tight', pad_inches = 0)
        elif i==n-1:
            ax = axs # add box to last image (detector)
            rect0 = patches.Rectangle((46,46),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect1 = patches.Rectangle((93,46),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect2 = patches.Rectangle((140,46),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect3 = patches.Rectangle((46,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect4 = patches.Rectangle((78,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect5 = patches.Rectangle((109,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect6 = patches.Rectangle((140,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect7 = patches.Rectangle((46,125),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect8 = patches.Rectangle((93,125),20,20,linewidth=1,edgecolor='r',facecolor='none')
            rect9 = patches.Rectangle((140,125),20,20,linewidth=1,edgecolor='r',facecolor='none')
            if det_box:
                ax.add_patch(rect0)
                ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                ax.add_patch(rect4)
                ax.add_patch(rect5)
                ax.add_patch(rect6)
                ax.add_patch(rect7)
                ax.add_patch(rect8)
                ax.add_patch(rect9)
            ax.imshow(intensity_list[-1][idx].abs().cpu().numpy().reshape(size,size),cmap=cmap, vmax=intensity_list[-1][idx].abs().cpu().numpy().max()*1,
            		vmin=intensity_list[-1][idx].abs().cpu().numpy().min()*1 )
            ax.set_axis_off()
            plt.savefig(fname+"_%d.pdf" % int(i),bbox_inches = 'tight', pad_inches = 0)
        else:
            ax = axs
            ax.imshow(intensity_list[i][idx].abs().cpu().numpy().reshape(size,size),cmap=cmap ,vmax=intensity_list[i][idx].abs().cpu().numpy().max()*1,
                    vmin=intensity_list[i][idx].abs().cpu().numpy().min()*1)
            ax.set_axis_off()
            plt.savefig(fname+"_%d.pdf" % int(i),bbox_inches = 'tight', pad_inches = 0)




def forward_func_visualization(intensity_list, size, cmap='Greys', idx=0, fname="forward_func_visualization.pdf",
            save_input=False, det_box=False, intensity_plot=True):
    n = len(intensity_list)
    size = size
    fig, axs = plt.subplots(1, n)
    fig.set_figheight(22)
    fig.set_figwidth(22)
    ax = axs[0] # first image is input
    print(n, intensity_list[0][idx].real.cpu().shape)
    if save_input:
        with open(fname+"_img.npy", 'wb') as f:
            np.save(f, intensity_list[0][idx].real.cpu().detach().numpy())
    ax.imshow(intensity_list[0][idx].real.cpu().reshape(size,size),cmap="Greys")
    ax.set_axis_off()

    ax = axs[-1] # add box to last image (detector)
    rect0 = patches.Rectangle((46,46),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect1 = patches.Rectangle((93,46),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect2 = patches.Rectangle((140,46),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect3 = patches.Rectangle((46,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect4 = patches.Rectangle((78,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect5 = patches.Rectangle((109,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect6 = patches.Rectangle((140,85),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect7 = patches.Rectangle((46,125),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect8 = patches.Rectangle((93,125),20,20,linewidth=1,edgecolor='r',facecolor='none')
    rect9 = patches.Rectangle((140,125),20,20,linewidth=1,edgecolor='r',facecolor='none')
    if det_box:
        ax.add_patch(rect0)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)
        ax.add_patch(rect5)
        ax.add_patch(rect6)
        ax.add_patch(rect7)
        ax.add_patch(rect8)
        ax.add_patch(rect9)
    if intensity_plot:
        #im = ax.imshow( (intensity_list[-1][idx].abs()**2).cpu().numpy().reshape(size,size),cmap=cmap, vmax=intensity_list[-1][idx].abs().cpu().numpy().max()*1,
        #    vmin=intensity_list[-1][idx].abs().cpu().numpy().min()*1 )
        im = ax.imshow( (intensity_list[-1][idx].abs()**2).cpu().numpy().reshape(size,size),cmap=cmap, vmax=intensity_list[-1][idx].abs().cpu().numpy().max()*1)
    else:
        #im = ax.imshow(intensity_list[-1][idx].abs().cpu().numpy().reshape(size,size),cmap=cmap, vmax=intensity_list[-1][idx].abs().cpu().numpy().max()*1,
        #    vmin=intensity_list[-1][idx].abs().cpu().numpy().min()*1 )
        im = ax.imshow(intensity_list[-1][idx].abs().cpu().numpy().reshape(size,size),cmap=cmap, vmax=intensity_list[-1][idx].abs().cpu().numpy().max()*1)
    ax.set_axis_off()
    for i in range(1,n-1):
        ax = axs[i]
        im = ax.imshow(intensity_list[i][idx].abs().cpu().numpy().reshape(size,size),cmap=cmap)#,vmax=intensity_list[i][idx].abs().cpu().numpy().max()*1,
                #vmin=intensity_list[i][idx].abs().cpu().numpy().min()*1)
        ax.set_axis_off()

    plt.savefig(fname,bbox_inches = 'tight', pad_inches = 0)
    fig = plt.figure()
    if intensity_plot:
        im = plt.imshow( (intensity_list[-1][idx].abs()**2).cpu().numpy().reshape(size,size),cmap=cmap)
    else:
        im = plt.imshow(intensity_list[-1][idx].abs().cpu().numpy().reshape(size,size),cmap=cmap)
    fig.colorbar(im)
    plt.savefig(fname+"_detector.pdf",bbox_inches = 'tight', pad_inches = 0)


def data_to_cplex_slm(batch, device, num_classes=10, input_padding=0, reverse_onehot=False, save_inputs=False, binarize=False):
    #print("testing data mapping")
    assert(device)
    images = batch[0].to(device).double()  # (64, 1, args.sys_size, args.sys_size) float32 1. 0.
    if binarize:
        threshold = 0.5
        images = torch.where(images <= threshold, images, 1.0).double()
        images = torch.where(images > threshold, images, 0.03162).double()
        images = torch.where(images != 1.0, images, -1.0).float()
    else:
        images = torch.where(images <= 0, images, -images).double()
        images = torch.where(images != 0, images, 0.03162).float()
    labels_ = batch[1].to(device)  # int64 9 0
    images = torch.nn.functional.pad(images, pad=(input_padding,input_padding,input_padding,input_padding))
    labels = torch.nn.functional.one_hot(labels_, num_classes=10).float()
    if reverse_onehot:
        labels = (labels + 1) % 2
    images = torch.squeeze(torch.cat((images.unsqueeze(-1),
                    torch.zeros_like(images.unsqueeze(-1))), dim=-1), dim=1)
    if save_inputs:
        for i in range(images.shape[0]):
            filename = "input_" + str(i)+"_"+str(labels_[i].cpu().item())  + ".npy"
            print(images[i].shape)
            with open(filename, 'wb') as f:
                np.save(filename, images[i].cpu())
        assert(0), "currently only save the first batch"
    return torch.view_as_complex(images), labels

def data_to_cplex_slm2(batch, device, num_classes=10, input_padding=0, reverse_onehot=False, save_inputs=False, binarize=False):
    #print("testing data mapping")
    assert(device)
    images = batch.to(device).double()  # (64, 1, args.sys_size, args.sys_size) float32 1. 0.
    if binarize:
        threshold = 0.5
        images = torch.where(images <= threshold, images, 1.0).double()
        images = torch.where(images > threshold, images, 0.03162).double()
        images = torch.where(images != 1.0, images, -1.0).float()
    else:
        images = torch.where(images <= 0, images, -images).double()
        images = torch.where(images != 0, images, 0.03162).float()
    images = torch.nn.functional.pad(images, pad=(input_padding,input_padding,input_padding,input_padding))
    images = torch.squeeze(torch.cat((images.unsqueeze(-1),
                    torch.zeros_like(images.unsqueeze(-1))), dim=-1), dim=1)
    if save_inputs:
        for i in range(images.shape[0]):
            filename = "input_" + str(i)+"_"+str(labels_[i].cpu().item())  + ".npy"
            print(images[i].shape)
            with open(filename, 'wb') as f:
                np.save(filename, images[i].cpu())
        assert(0), "currently only save the first batch"
    return torch.view_as_complex(images)


