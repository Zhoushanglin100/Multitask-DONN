import torch
import numpy as np
from orig.get_h import _field_Fresnel

class Detector(torch.nn.Module):
    """ Implementation of detector plane for multi-task classification

    The outputs are collected with specific defined detector regions over the entire light propogation.
    The outputs are (optional) normlized using functions such as softmax to enable effcient training of D2NNs.

    Args:
        >> detector plane design <<
        x: the x-axis location for your detector region (left-top)
        y: the y-axis location for your detector region (left-top)
        det_size: the size of the detector region
        size: the system size
        activation: activation function for training. default=torch.nn.Softmax(dim=-1)
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference:
    """
    def __init__(self, x_loc, y_loc, det_size=20, size=200, 
                    activation = torch.nn.Softmax(dim=-1), 
                    intensity_mode=False):
        super(Detector, self).__init__()
        self.size = size
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.det_size = det_size
        self.activation = activation
        self.intensity_mode = intensity_mode
    def forward(self, x):
        if self.intensity_mode:
            x = x.abs() ** 2 # intensity mode
        else:
            x = x.abs()
        assert len(self.x_loc)==len(self.y_loc) and len(self.x_loc) > 1, 'the input location information is wrong!'

        detectors = torch.cat((x[:, self.x_loc[0] : self.x_loc[0] + self.det_size, self.y_loc[0] : self.y_loc[0] + self.det_size].mean(dim=(1, 2)).unsqueeze(-1),
                               x[:, self.x_loc[1] : self.x_loc[1] + self.det_size, self.y_loc[1] : self.y_loc[1] + self.det_size].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)
        for i in range(2, len(self.x_loc)):
            detectors = torch.cat((detectors, x[:, self.x_loc[i] : self.x_loc[i] + self.det_size, self.y_loc[i] : self.y_loc[i] + self.det_size].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)

        assert self.x_loc[-1] + self.det_size < self.size and self.y_loc[-1] + self.det_size < self.size, 'the region is out of detector!'
        if self.activation == None:
            return detectors
        else:
            return self.activation(detectors)


class DiffractiveLayerRaw(torch.nn.Module):
    """ Implementation of diffractive layer without hardware constraints

    Args:
	size: system size
	distance: diffraction distance
	name: name of the layer
	amplitude_factor: training regularization factor w.r.t amplitude vs phase in backpropogation
	phase_mod: enable phase modulation or just diffraction. default: True
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference:
    """
    def __init__(self, wavelength=5.32e-7, pixel_size=3.6e-5, size=200, pad = 0, distance=0.1, 
                 name="diffractive_layer_raw",
				 amplitude_factor = 6, mesh_size = 1, approx = "Fresnel", phase_mod=True):
        super(DiffractiveLayerRaw, self).__init__()

        self.size = size                            # 200 * 200 neurons in one layer
        self.distance = distance                    # distance bewteen two layers (3cm)
        self.pad = pad
        self.ll = pixel_size * (self.size + self.pad*2)                          # layer length (8cm)
        self.wl = wavelength                    # wave length
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * 3.1415926 / self.wl       # wave number
        self.approx = approx
        self.mesh_size = mesh_size
        assert (self.pad > 0), "padding in forward diffraction has to be greater than 0 (need more explainations here)"
        self.pixel_size = pixel_size
        self.ddi = 1 / self.pixel_size
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - ((self.size + self.pad * 2) // 2)) * self.fi) + np.square((y - ((self.size + self.pad * 2) // 2)) * self.fi),
            shape=((self.size + self.pad * 2), (self.size + self.pad * 2)), dtype=np.complex64)

        if self.approx == "Fresnel":
            print("Network is constructed using Fresnel approximation")
            h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
            # self.h (syssize, syssize, 2)
            self.h = torch.nn.Parameter(torch.view_as_complex(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1)), requires_grad=False)
        elif self.approx == "Fraunhofer":
            print('Network is constructed using fraunhofer.4 approximation')
            wn = self.wn
            distance = self.distance
            r = np.fromfunction(
                   lambda x, y: np.square((x - (self.size + self.pad * 2) / 2) * self.pixel_size) + np.square((y - (self.size + self.pad * 2) / 2 ) * self.pixel_size), shape=(self.size+self.pad*2, self.size+self.pad*2), dtype=np.float64)

            temp = np.fromfunction(
                   lambda x, y: np.sinc(wn * (x - (self.size + self.pad * 2) /2 ) * self.pixel_size / distance * self.pixel_size*np.sqrt(self.fill_factor) / 2 /np.pi) * np.sinc(wn * (y - (self.size + self.pad * 2) / 2) * self.pixel_size / distance * self.pixel_size*np.sqrt(self.fill_factor) / 2 /np.pi),
                   shape=(self.size+self.pad*2, self.size+self.pad*2), dtype=np.float64)
            h = temp * np.exp(1.0j * wn * r / (2 * distance))* np.exp(1.0j * wn * distance)/(1.0j * 2 * np.pi/wn * distance) * self.pixel_size * self.pixel_size * self.fill_factor
            h = torch.from_numpy(h)
            h = torch.fft.fftshift(h)
            self.h = torch.nn.Parameter(torch.fft.fft2(h.to(torch.complex64)), requires_grad=False)
        elif self.approx == "Fresnel2":
            print('Network is constructed using Fresnel.2 approximation')
            wn = self.wn
            distance = self.distance
            r = np.fromfunction(
                lambda x, y: np.square((x - (self.size + self.pad*2) // 2) * self.pixel_size) + np.square((y - (self.size + self.pad*2) // 2) * self.pixel_size), shape=(self.size + self.pad*2, self.size + self.pad*2), dtype=np.float64)

            h = np.exp(1.0j * self.wn/2/distance * r) #/(1.0j * self.wl * distance)
            h = torch.fft.ifftshift(torch.from_numpy(h))
            self.h = torch.nn.Parameter(torch.fft.fft2(h.to(torch.complex64)), requires_grad=False)
        elif self.approx == "Sommerfeld":
            print("Network is constructed using Sommerfeld approximation")
            wn = self.wn * self.pixel_size
            distance = self.distance * self.ddi
            r = np.fromfunction(
                    lambda x, y: np.square((x - ((self.size + self.pad * 2)//2))) + np.square((y - ((self.size + self.pad * 2) // 2))) + np.square(distance),
                    shape = ((self.size + self.pad * 2), (self.size + self.pad * 2)), dtype=np.float64)
            r = torch.from_numpy(r)
            h = 1 / (2 * np.pi) * distance / r
            r = np.sqrt(r)
            temp = wn * r
            temp = torch.view_as_complex(torch.stack((torch.cos(temp), torch.sin(temp)), dim=-1))
            h = h * (1 / r - 1.0j * wn) * temp
            h = torch.fft.fftshift(h)
            self.h = torch.nn.Parameter(torch.fft.fft2(h.to(torch.complex64)), requires_grad=False)
        elif self.approx == 'Fresnel3':
            return_in_outK, comp, iiij2N, iiij2No2, iiijN =  _field_Fresnel(self.distance, self.size + 2*self.pad, self.pixel_size, self.wl)
            iiij2No2 = torch.from_numpy(iiij2No2.astype('float32'))
            iiij2N = torch.from_numpy(iiij2N.astype('float32'))
            iiijN = torch.from_numpy(iiijN.astype('float32'))
            return_in_outK = torch.complex(torch.tensor([return_in_outK.real.astype('float32')]), 
                                           torch.tensor([return_in_outK.imag.astype('float32')]))
            self.h = torch.nn.Parameter(torch.fft.fft2(return_in_outK.to(torch.complex64)), requires_grad=False)
            self.comp = torch.nn.Parameter(torch.complex(torch.tensor([comp.real]), torch.tensor([comp.imag])),requires_grad=False)
            self.iiij2N = torch.nn.Parameter(iiij2N,requires_grad=False)
            self.iiij2No2 = torch.nn.Parameter(iiij2No2,requires_grad=False)
            self.iiijN = torch.nn.Parameter(iiijN,requires_grad=False)
            self.N = self.size + self.pad * 2
            self.No2 = int(self.N/2)
        else:
            assert(0), "approximation function %s is not implemented; currently supporting Fresnel,Freshnel2, Sommerfeld,Fraunhofer"

        # phase parameter init
        self.phase = torch.nn.Parameter(torch.from_numpy( 2 * np.pi * torch.nn.init.xavier_uniform_(torch.empty(self.size,self.size)).numpy() ), requires_grad=True)
        self.register_parameter(name, self.phase)
        self.phase_model = phase_mod
        self.amplitude_factor = amplitude_factor

    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        if self.approx == 'Fresnel3':

            waves = torch.nn.functional.pad(waves, (self.pad,self.pad,self.pad,self.pad)) # pad to eliminate perodic effects

            batch_size = waves.shape[0]
            if torch.cuda.is_available():
                in_outF = torch.zeros((batch_size, 2*self.N, 2*self.N), dtype=torch.complex64).cuda(waves.get_device())
                in_outF[:, (self.N - self.No2):(self.N + self.No2), (self.N - self.No2):(self.N + self.No2)] = waves[:, (self.N - 2*self.No2): self.N,(self.N - 2*self.No2):self.N].cuda(waves.get_device()) #cutting off field if N odd (!)
            else:
                in_outF = torch.zeros((batch_size, 2*self.N, 2*self.N), dtype=torch.complex64)
                in_outF[:, (self.N - self.No2):(self.N + self.No2), (self.N - self.No2):(self.N + self.No2)] = waves[:, (self.N - 2*self.No2): self.N,(self.N - 2*self.No2):self.N]   #cutting off field if N odd (!)
            in_outF[:, (self.N - self.No2):(self.N + self.No2), (self.N - self.No2):(self.N + self.No2)] *= self.iiij2No2
            in_outF = torch.fft.ifft2(torch.fft.fft2(in_outF) * self.h * self.iiij2N)

            Ftemp = (in_outF[:, self.No2:self.N + self.No2, self.No2:self.N + self.No2] - in_outF[:, self.No2-1:self.N + self.No2 -1, self.No2:self.N + self.No2])
            Ftemp += in_outF[:, self.No2 - 1:self.N + self.No2 - 1, self.No2 - 1:self.N + self.No2 - 1]
            Ftemp -= in_outF[:, self.No2 : self.N + self.No2, self.No2 - 1:self.N + self.No2 - 1]
            Ftemp *= 0.25 * self.comp
            Ftemp *= self.iiijN
            
            temp = torch.nn.functional.pad(Ftemp, (-self.pad,-self.pad,-self.pad,-self.pad)) # reverse pad for next prop (center crop)

        else:
            waves = torch.nn.functional.pad(waves, (self.pad,self.pad,self.pad,self.pad))
            temp = torch.fft.ifft2( torch.fft.fft2(waves) * self.h )
            temp = torch.nn.functional.pad(temp, (-self.pad,-self.pad,-self.pad,-self.pad))

        if not self.phase_model:
            return temp
            
        exp_j_phase = torch.view_as_complex(torch.stack((self.amplitude_factor*torch.cos(self.phase),
				                                         self.amplitude_factor*torch.sin(self.phase)), dim=-1))
        x = temp * exp_j_phase
        return x
