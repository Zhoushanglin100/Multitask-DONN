import numpy as np
import torch
import torch.nn.functional as F

from AutoMTL.APIs.donn.mtl.mtl_model import mtl_model

import lightbridge.layers as layers
import lightbridge.utils as utils


class DiffractiveClassifier_Raw(mtl_model):
    def __init__(self, det_x_loc, det_y_loc, det_size,
                 wavelength=5.32e-7, pixel_size=0.000036,
                 batch_norm=False, sys_size = 200, pad = 100,
                 distance=0.1, num_layers=2, amp_factor=6,
                 approx="Fresnel3",
                 heads_dict={}):
        super(DiffractiveClassifier_Raw, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pad = pad
        self.approx=approx
        self.approx=approx
        self.diffractive_layers = torch.nn.ModuleList([layers.DiffractiveLayerRaw(wavelength=self.wavelength, pixel_size=self.pixel_size,
                                                                                    size=self.size, pad = self.pad, distance=self.distance,
                                                                                    amplitude_factor = amp_factor, approx=self.approx,
                                                                                    phase_mod=True) for _ in range(num_layers)])
        self.last_diffraction = layers.DiffractiveLayerRaw(wavelength=self.wavelength, pixel_size=self.pixel_size,
                                                            size=self.size, pad = self.pad, distance=self.distance,
                                                            approx=self.approx, phase_mod=False)
        self.detector = layers.Detector(x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=self.size)

        self.task_list = heads_dict.keys()
        self.heads_dict = heads_dict

    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
        x = self.last_diffraction(x)
        output = self.detector(x)
        return output

    def to_slm(self, fname):
        for index, layer in enumerate(self.diffractive_layers):
            w = torch.argmax(torch.nn.functional.gumbel_softmax(layer.voltage,tau=1,hard=True).cpu(), dim=-1)
            with open(fname+str(index)+".npy",'wb') as f:
                print("saving model %s layer(%d) into numpy array %s" % (fname, index, fname+str(index)+".npy"), w.shape)
                np.save(f,w)
            f.close()
        return

    def save_weights_numpy(self, fname):
        for index, layer in enumerate(self.diffractive_layers):
            w = torch.argmax(torch.nn.functional.gumbel_softmax(layer.voltage,tau=1,hard=True).cpu(), dim=-1)
            with open(fname+str(index)+".npy",'wb') as f:
                print("saving model %s layer(%d) into numpy array %s" % (fname, index, fname+str(index)+".npy"), w.shape)
                np.save(f,w)
            f.close()
        return

    def prop_view(self, x):
        prop_list = []
        prop_list.append(x)
        x = x #* self.amp_factor
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
            prop_list.append(x)
        x = self.last_diffraction(x)
        prop_list.append(x)
        for i in range(x.shape[0]):
            print(i)
            utils.forward_func_visualization(prop_list, self.size, fname="mnist_%s.pdf" % i, idx=i, intensity_plot=False)
        output = self.detector(x)
        return

    def phase_view(self,x, cmap="hsv"):
        phase_list = []
        for index, layer in enumerate(self.diffractive_layers):
            phase_list.append(torch.argmax(torch.nn.functional.gumbel_softmax(layer.voltage,tau=1,hard=True).cpu(), dim=-1))
        utils.phase_visualization(phase_list,size=self.size, cmap="gray", fname="prop_view_reflection.pdf")
        return