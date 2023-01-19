import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random, os
import argparse
import numpy as np
from collections import OrderedDict
from scipy.special import softmax
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from load_data import EMNIST
import orig.data as dataset
import layers as layers

from train_mtl import Trainer
from donn_model_mtl import DiffractiveClassifier_Raw

# try:
#     import wandb
#     has_wandb = True
# except ImportError: 
#     has_wandb = False

###########################################################

def main(args): 

    print("---------------")
    print(args)
    print("---------------")
    
    # if has_wandb and args.enable_wandb:
    #     wandb.init(project='DONN', entity='zhoushanglin100')
    #     wandb.init(config=args)
    #     wandb.config.update(args)

    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ### load data
    data_root = "../../../"

    transform = transforms.Compose([transforms.Resize((200,200),interpolation=2),transforms.ToTensor()])

    train_dataset_m = torchvision.datasets.MNIST(data_root+"data", train=True, transform=transform, download=True)
    val_dataset_m = torchvision.datasets.MNIST(data_root+"data", train=False, transform=transform, download=True)
    train_dataset_f = torchvision.datasets.FashionMNIST(data_root+"Fdata", train=True, transform=transform, download=True)
    val_dataset_f = torchvision.datasets.FashionMNIST(data_root+"Fdata", train=False, transform=transform, download=True)
    train_dataset_k = torchvision.datasets.KMNIST(data_root+"Kdata", train=True, transform=transform, download=True)
    val_dataset_k = torchvision.datasets.KMNIST(data_root+"Kdata", train=False, transform=transform, download=True)
    train_dataset_e = EMNIST(data_root+"Edata", train=True, divide=0, transform=transform, download=True)
    val_dataset_e = EMNIST(data_root+"Edata", train=False, divide=0, transform=transform, download=True)

    train_dataloader_m = DataLoader(dataset=train_dataset_m, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=False)
    val_dataloader_m = DataLoader(dataset=val_dataset_m, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=False)
    train_dataloader_f = DataLoader(dataset=train_dataset_f, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=False)
    val_dataloader_f = DataLoader(dataset=val_dataset_f, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=False)
    train_dataloader_k = DataLoader(dataset=train_dataset_k, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=False)
    val_dataloader_k = DataLoader(dataset=val_dataset_k, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=False)
    train_dataloader_e = DataLoader(dataset=train_dataset_e, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=False)
    val_dataloader_e = DataLoader(dataset=val_dataset_e, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=False)

    tasks = ['mnist', 'fmnist', 'kmnist', 'emnist']
    train_dataloader = {"mnist": train_dataloader_m, 
                        "fmnist": train_dataloader_f, 
                        "kmnist": train_dataloader_k, 
                        "emnist": train_dataloader_e}
    val_dataloader = {"mnist": val_dataloader_m, 
                      "fmnist": val_dataloader_f, 
                      "kmnist": val_dataloader_k, 
                      "emnist": val_dataloader_e}

    headsDict = nn.ModuleDict()
    trainDataloaderDict = {task: [] for task in tasks}
    valDataloaderDict = {task: [] for task in tasks}
    criterionDict = {}
    metricDict = {}

    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    for task in tasks:
        headsDict[task] = layers.Detector(x_loc = [40, 40, 40, 90, 90, 90, 90, 140, 140, 140], 
                                          y_loc = [40, 90, 140, 30, 70, 110, 150, 40, 90, 140], 
                                          det_size = 20, size = args.sys_size)

        trainDataloaderDict[task] = train_dataloader[task]
        valDataloaderDict[task] = val_dataloader[task]

        criterionDict[task] = criterion
        metricDict[task] = []


    ### Define MTL model
    mtlmodel = DiffractiveClassifier_Raw(num_layers = args.depth, 
                                         wavelength = args.wavelength, 
                                         pixel_size = args.pixel_size, 
                                         sys_size=args.sys_size, 
                                         pad = args.pad,
                                         distance = args.distance,
                                         amp_factor=args.amp_factor, 
                                         approx=args.approx,
                                         heads_dict=headsDict)
    mtlmodel = mtlmodel.to(device)

    # if has_wandb and args.enable_wandb:
    #     wandb.watch(mtlmodel)

    ### Define training framework
    trainer = Trainer(mtlmodel, 
                      trainDataloaderDict, valDataloaderDict, 
                      criterionDict, metricDict, 
                      print_iters=10, val_iters=100, 
                      save_iters=100, save_num=1, 
                      policy_update_iters=100)

    # ----------------
    ### validation
    if args.evaluate:
        print(">>>>>>>> Validation <<<<<<<<<<")
        # ckpt = torch.load(f"{savepath}post_train_{args.post_iters}iter.model")
        ckpt = torch.load(args.evaluate)
        # print(ckpt["state_dict"].keys())
        mtlmodel.load_state_dict(ckpt["state_dict"])
        # print(mtlmodel)
        trainer.validate('mtl', hard=True) 

        ## policy visualization        
        policy_list = {"mnist": [], "fmnist": [], "kmnist": [], "emnist": []}
        for name, param in mtlmodel.named_parameters():
            if 'policy' in name and not torch.eq(param, torch.tensor([0., 0., 0.]).cuda()).all():
                policy = param.data.cpu().detach().numpy()
                distribution = softmax(policy, axis=-1)
                if '.mnist' in name:
                    policy_list['mnist'].append(distribution)
                elif '.fmnist' in name:
                    policy_list['fmnist'].append(distribution)
                elif '.kmnist' in name:
                    policy_list['kmnist'].append(distribution)
                elif '.emnist' in name:
                    policy_list['emnist'].append(distribution)
        print(policy_list)

        spectrum_list = []
        ylabels = {'mnist': 'MNIST',
                    "fmnist": "FMNIST",
                    "kmnist": "KMNIST",
                    "emnist": "EMNIST"} 
        tickSize = 15
        labelSize = 16
        for task in tasks:
            policies = policy_list[task]    
            spectrum = np.stack([policy for policy in policies])
            spectrum = np.repeat(spectrum[np.newaxis,:,:],1,axis=0)
            spectrum_list.append(spectrum)
            
            plt.figure(figsize=(10,5))
            plt.xlabel('Layer No.', fontsize=labelSize)
            plt.xticks(fontsize=tickSize)
            plt.ylabel(ylabels[task], fontsize=labelSize)
            plt.yticks(fontsize=tickSize)
            
            ax = plt.subplot()
            im = ax.imshow(spectrum.T)
            ax.set_yticks(np.arange(3))
            ax.set_yticklabels(['shared', 'specific', 'skip'])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)

            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=tickSize)
            plt.savefig(f"spect_{task}.png")
            plt.close()

        return

    # ==================================
    ### Train

    checkpoint = 'checkpoint/'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    savepath = checkpoint+args.save_dir+"/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    print(f"All ckpts save to {savepath}")

    # ----------------
    ### Step 1: pre-train
    print(">>>>>>>> pre-train <<<<<<<<<<")
    trainer.pre_train(iters=args.pretrain_iters, lr=args.lr, 
                      savePath=savepath, writerPath=savepath)

    # ----------------
    ### Step 2: alter-train
    print(">>>>>>>> alter-train <<<<<<<<<<")
    loss_lambda = {'mnist': 1, 
                   'fmnist': 1, 
                   'kmnist': 1, 
                   'emnist': 1,
                   'policy': 0.0005}
    trainer.alter_train_with_reg(iters=args.alter_iters, policy_network_iters=(50,200), 
                                 policy_lr=0.01, network_lr=0.0001,
                                 loss_lambda=loss_lambda,
                                 savePath=savepath, writerPath=savepath)

    # ----------------
    ### Step 3: sample policy from trained policy distribution and save
    print(">>>>>>>> Sample Policy <<<<<<<<<<")
    policy_list = {"mnist": [], "fmnist": [], "kmnist": [], "emnist": []}
    name_list = {"mnist": [], "fmnist": [], "kmnist": [], "emnist": []}

    for name, param in mtlmodel.named_parameters():
        if 'policy' in name :
            print(name)
            if '.mnist' in name:
                policy_list['mnist'].append(param.data.cpu().detach().numpy())
                name_list['mnist'].append(name)
            elif '.fmnist' in name:
                policy_list['fmnist'].append(param.data.cpu().detach().numpy())
                name_list['fmnist'].append(name)
            elif '.kmnist' in name:
                policy_list['kmnist'].append(param.data.cpu().detach().numpy())
                name_list['kmnist'].append(name)
            elif '.emnist' in name:
                policy_list['emnist'].append(param.data.cpu().detach().numpy())
                name_list['emnist'].append(name)

    shared = 1
    sample_policy_dict = OrderedDict()
    for task in tasks:
        count = 0
        for name, policy in zip(name_list[task], policy_list[task]):
            if count < shared:
                sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0]).cuda()
            else:
                distribution = softmax(policy, axis=-1)
                distribution /= sum(distribution)
                choice = np.random.choice((0, 1, 2), p=distribution)
                if choice == 0:
                    sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0]).cuda()
                elif choice == 1:
                    sample_policy_dict[name] = torch.tensor([0.0, 1.0, 0.0]).cuda()
                elif choice == 2:
                    sample_policy_dict[name] = torch.tensor([0.0, 0.0, 1.0]).cuda()
            count += 1

    # sample_policy_dict = OrderedDict()
    # for task in tasks:
    #     for name, policy in zip(name_list[task], policy_list[task]):
    #         distribution = softmax(policy, axis=-1)
    #         distribution /= sum(distribution)

    #         choice = np.random.choice((0, 1, 2), p = distribution)
    #         if choice == 0:
    #             sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0])
    #         elif choice == 1:
    #             sample_policy_dict[name] = torch.tensor([0.0, 1.0, 0.0])
    #         elif choice == 2:
    #             sample_policy_dict[name] = torch.tensor([0.0, 0.0, 1.0])

    sample_path = savepath
    sample_state = {'state_dict': sample_policy_dict}
    torch.save(sample_state, sample_path + 'sample_policy.model')

    # # ----------------
    ### Step 4: post train from scratch
    print(">>>>>>>> Post-train <<<<<<<<<<")
    loss_lambda = {'mnist': 1, 'fmnist': 1, 'kmnist': 1, 'emnist': 1}
    trainer.post_train(iters=args.post_iters, lr=0.01,
                        decay_lr_freq=2000, decay_lr_rate=0.5,
                        loss_lambda=loss_lambda,
                        savePath=savepath, writerPath=savepath,
                        reload='sample_policy.model')
    
# # --------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### DONN parameters
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--pretrain-iters', type=int, default=3000, help='#iterations for pre-training, default: 10000')
    parser.add_argument('--alter-iters', type=int, default=6000, help='#iterations for alter-train, default: 20000')
    parser.add_argument('--post-iters', type=int, default=30000, help='#iterations for post-train, default: 30000')
    parser.add_argument('--lr', type=float, default=0.01, help='pre-train learning rate')
    parser.add_argument('--save-dir', type=str, default='avg_L5', help="save the model")
    parser.add_argument('--evaluate', type=str, help="Model path for evaulation")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--depth', type=int, default=5, help='number of fourier optic transformations/num of layers')
    parser.add_argument('--sys-size', type=int, default=200, help='system size (dim of each diffractive layer)')
    parser.add_argument('--distance', type=float, default=0.3, help='layer distance (default=0.1 meter)')
    parser.add_argument('--amp-factor', type=float, default=4, help='regularization factors to balance phase-amplitude where they share same downstream graidents')
    parser.add_argument('--pixel-size', type=float, default=0.000036, help='the size of pixel in diffractive layers')
    parser.add_argument('--pad', type=int, default=50, help='the padding size ')
    parser.add_argument('--approx', type=str, default='Fresnel3', help="Use which Approximation, Sommerfeld, fresnel or fraunhofer.")
    parser.add_argument('--wavelength', type=float, default=5.32e-7, help='wavelength')
    # parser.add_argument('--enable-wandb', action='store_true', default=False, help='whether to use wandb to log')

    args_ = parser.parse_args()

    main(args_)
