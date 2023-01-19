import warnings
import torch

from mtl.mtl_model import mtl_model
from layers_mtl import DiffractiveLayerRawNode, ModuleList, Sequential
import layers

class DiffractiveClassifier_Raw(mtl_model):
    def __init__(self,
                 wavelength=5.32e-7, pixel_size=0.000036,
                 sys_size = 200, pad = 100,
                 distance=0.1, num_layers=2, amp_factor=6,
                 approx="Fresnel3",
                 heads_dict={}
                 ):
        super(DiffractiveClassifier_Raw, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pad = pad
        self.approx=approx
        
        self.task_list = heads_dict.keys()
        self.heads_dict = heads_dict

        # self.diffractive_layers = torch.nn.ModuleList([layers.DiffractiveLayerRaw(wavelength=self.wavelength, pixel_size=self.pixel_size,
        #                                                                             size=self.size, pad = self.pad, distance=self.distance,
        #                                                                             amplitude_factor = amp_factor, approx=self.approx,
        #                                                                             phase_mod=True) for _ in range(num_layers)])
        # self.last_diffraction = layers.DiffractiveLayerRaw(wavelength=self.wavelength, pixel_size=self.pixel_size,
        #                                                     size=self.size, pad = self.pad, distance=self.distance,
        #                                                     approx=self.approx, phase_mod=False)
        # self.detector = layers.Detector(x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=self.size)

        # self.diffractive_layers = ModuleList(torch.nn.ModuleList([DiffractiveLayerRawNode(wavelength = self.wavelength, 
        #                                                                                   pixel_size = self.pixel_size,
        #                                                                                   size=self.size, 
        #                                                                                   pad = self.pad, 
        #                                                                                   distance = self.distance,
        #                                                                                   amplitude_factor = amp_factor, 
        #                                                                                   approx = self.approx,
        #                                                                                   phase_mod = True,
        #                                                                                   task_list = self.task_list) for _ in range(num_layers)]))
        self.diffractive_layers = [DiffractiveLayerRawNode(wavelength = self.wavelength, 
                                                           pixel_size = self.pixel_size,
                                                           size=self.size, 
                                                           pad = self.pad, 
                                                           distance = self.distance,
                                                           amplitude_factor = amp_factor, 
                                                           approx = self.approx,
                                                           phase_mod = True,
                                                           task_list = self.task_list) for _ in range(num_layers)]
        self.diffractive_layers = Sequential(torch.nn.Sequential(*self.diffractive_layers))

        self.compute_depth()

    def forward(self, x, stage='common', task=None, tau=5, hard=False, policy_idx=None):

        # for index, layer in enumerate(self.diffractive_layers):
        #     x = layer(x, stage, task, tau, hard, policy_idx)
        # x = self.last_diffraction(x, stage, task, tau, hard, policy_idx)
        # output = self.detector(x)

        x = self.diffractive_layers(x, stage, task, tau, hard, policy_idx)

        if task != None:
            output = self.heads_dict[task](x)
            return output
        else:
            warnings.warn('No task specified, return feature directly')
            return x
        