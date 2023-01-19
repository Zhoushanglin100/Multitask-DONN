import copy
import torch
import torch.nn as nn

from mtl.base_node import BasicNode
from mtl.layer_containers import LazyLayer
import layers


class DiffractiveLayerRawNode(BasicNode):
    def __init__(self, 
                 wavelength = 5.32e-7, 
                 pixel_size = 3.6e-5, 
                 size = 200, 
                 pad = 0, 
                 distance=0.1, 
                 amplitude_factor = 6, 
                 approx = "Fresnel", 
                 phase_mod = True,
                 task_list = ['basic']
                 ):
        super(DiffractiveLayerRawNode, self).__init__(taskList=task_list)
        self.taskSp = True
        self.basicOp = layers.DiffractiveLayerRaw(wavelength = wavelength, 
                                                  pixel_size = pixel_size,
                                                  size = size, 
                                                  pad = pad, 
                                                  distance = distance,
                                                  amplitude_factor = amplitude_factor, 
                                                  approx = approx,
                                                  phase_mod = phase_mod)
        self.weight = self.basicOp.phase
        self.policy = nn.ParameterDict()
        self.dsOp = nn.ModuleDict()
        self.build_layer()
        self.generate_taskOp()


    def build_layer(self):
        super(DiffractiveLayerRawNode, self).build_layer()
        self.generate_dsOp()
    

    def generate_dsOp(self):
        if len(self.taskList) > 1:
            for task in self.taskList:
                self.dsOp[task] = nn.ModuleList()
                self.dsOp[task].append(LazyLayer())
        return


    def generate_taskOp(self):
        if len(self.taskList) > 1:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.basicOp)
                self.policy[task] = nn.Parameter(torch.tensor([0., 0., 0.]))
        return


    def compute_mtl(self, x, task, tau=5, hard=False):
        policy_task = self.policy[task]
        if hard is False:
            # Policy-train
            # possibility of each task
            possiblity = torch.nn.functional.gumbel_softmax(policy_task, tau=tau, hard=hard)
            feature_common = self.compute_common(x)
            feature_specific = self.compute_specific(x, task)
            feature_downsample = self.compute_downsample(x, task)
            feature = feature_common * possiblity[0] + \
                      feature_specific * possiblity[1] + \
                      feature_downsample * possiblity[2]
        else:
            # Post-train or Validation
            branch = torch.argmax(policy_task).item()
            if branch == 0:
                feature = self.compute_common(x)
            elif branch == 1:
                feature = self.compute_specific(x, task)
            elif branch == 2:
                feature = self.compute_downsample(x, task)
        return feature

    
    def compute_combined(self, x, task):
        feature_list = [self.compute_common(x)]
        if self.taskSp:
            feature_list.append(self.compute_specific(x, task))
            feature_list.append(self.compute_downsample(x, task))
        return torch.mean(torch.stack(feature_list), dim=0)

    
    def compute_downsample(self, x, task):
        for op in self.dsOp[task]:
            x = op(x)
        return x



class Sequential(nn.Module):

    def __init__(self, seq: nn.Sequential):
        """
            wrapper for nn.Sequential in order to apply MTL forwarding
        Args:
            seq: actual sequence of layers,
        """
        super(Sequential, self).__init__()
        self.models = seq

    def forward(self, x, stage='common', task=None, tau=5, hard=False, policy_idx=None):
        for node in self.models:
            if isinstance(node, DiffractiveLayerRawNode):
                x = node(x, stage, task, tau, hard)
            else:
                x = node(x)
        return x


class ModuleList(nn.Module):
    def __init__(self, lst: nn.ModuleList):
        super(ModuleList, self).__init__()
        self.models = lst

    def forward(self, x, stage='common', task=None, tau=5, hard=False, policy_idx=None):
        for index, layer in enumerate(self.models):
            if isinstance(layer, DiffractiveLayerRawNode):
                x = layer(x, stage, task, tau, hard)
            else:
                x = layer(x)
        return x