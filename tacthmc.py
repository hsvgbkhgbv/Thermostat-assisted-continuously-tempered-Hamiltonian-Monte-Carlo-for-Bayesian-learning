import torch
import numpy as np
import re


class TACTHMC:

    def __init__(self, model, N, eta_theta0, eta_xi0, c_theta0, c_xi0, gamma_theta0, gamma_xi0, enable_cuda, standard_interval=0.1, gaussian_decay=1e-3, version='accurate', temper_model='Metadynamics'):
        self.N = N
        self.model = model
        self.version = version
        self.pattern1 = re.compile(r'linear|conv')
        self.pattern2 = re.compile(r'lstm')
        self.model.register_buffer('xi' , torch.zeros(1))
        self.model.register_buffer('r_xi', torch.zeros(1))
        self.model.register_buffer('eta_xi', torch.Tensor([eta_xi0*self.N]))
        self.model.register_buffer('gamma_xi', torch.Tensor([gamma_xi0]))
        self.model.register_buffer('c_xi', torch.Tensor([c_xi0]))
        self.model.register_buffer('z_xi', model.c_xi)
        self.model.register_buffer('n_xi', torch.zeros(1))
        self.standard_interval = standard_interval
        self.temper_model_name = temper_model
        if self.temper_model_name == 'Metadynamics':
            self.temper_model = Metadynamics(gaussian_decay=gaussian_decay, enable_cuda=enable_cuda)
        elif self.temper_model_name == 'ABF':
            self.temper_model = ABF(enable_cuda=enable_cuda)
        for name, module in self.model._modules.items():
            if self.pattern1.match(name):
                size_w = module.weight.data.shape
                size_b = module.bias.data.shape
                module.register_buffer('r_w', torch.zeros(size_w))
                module.register_buffer('r_b', torch.zeros(size_b))
                module.register_buffer('eta_theta', torch.Tensor([eta_theta0*self.N]))
                module.register_buffer('c_theta', torch.Tensor([c_theta0]))
                module.register_buffer('gamma_theta', torch.Tensor([gamma_theta0]))
                if self.version == 'approx':
                    module.register_buffer('z_u', module.c_theta)
                elif self.version == 'accurate':
                    module.register_buffer('z_w', module.c_theta)
                    module.register_buffer('z_b', module.c_theta)
                module.register_buffer('n_w', torch.zeros(size_w))
                module.register_buffer('n_b', torch.zeros(size_b))
            elif self.pattern2.match(name):
                size_wih = module.weight_ih_l0.data.shape
                size_bih = module.bias_ih_l0.data.shape
                size_whh = module.weight_hh_l0.data.shape
                size_bhh = module.bias_hh_l0.data.shape
                module.register_buffer('r_wih', torch.zeros(size_wih))
                module.register_buffer('r_bih', torch.zeros(size_bih))
                module.register_buffer('r_whh', torch.zeros(size_whh))
                module.register_buffer('r_bhh', torch.zeros(size_bhh))
                module.register_buffer('eta_theta', torch.Tensor([eta_theta0*self.N]))
                module.register_buffer('c_theta', torch.Tensor([c_theta0]))
                module.register_buffer('gamma_theta', torch.Tensor([gamma_theta0]))
                if self.version == 'approx':
                    module.register_buffer('z_u', module.c_theta)
                elif self.version == 'accurate':
                    module.register_buffer('z_wih', module.c_theta)
                    module.register_buffer('z_bih', module.c_theta)
                    module.register_buffer('z_whh', module.c_theta)
                    module.register_buffer('z_bhh', module.c_theta)
                module.register_buffer('n_wih', torch.zeros(size_wih))
                module.register_buffer('n_bih', torch.zeros(size_bih))
                module.register_buffer('n_whh', torch.zeros(size_whh))
                module.register_buffer('n_bhh', torch.zeros(size_bhh))
        self.resample_momenta_xi()

    def get_z_u(self):
        buffer_ = []
        for name, module in self.model._modules.items():
            if self.version == 'approx':
                buffer_.append(torch.norm(module.z_u))
            elif self.version == 'accurate':
                if self.pattern1.match(name):
                    buffer_.append(torch.norm(module.z_w))
                    buffer_.append(torch.norm(module.z_b))
                elif self.pattern2.match(name):
                    buffer_.append(torch.norm(module.z_wih))
                    buffer_.append(torch.norm(module.z_bih))
                    buffer_.append(torch.norm(module.z_whh))
                    buffer_.append(torch.norm(module.z_bhh))
        return sum(buffer_)

    def get_z_xi(self):
        return self.model.z_xi.item()

    def get_fU(self):
        return self.fU

    def update(self, loss):
        xi = self.model.xi
        eta_xi = self.model.eta_xi
        r_xi = self.model.r_xi
        z_xi = self.model.z_xi
        c_xi = self.model.c_xi
        n_xi = self.model.n_xi
        gamma_xi = self.model.gamma_xi
        g, dg = self.g_fn(xi)
        invg, dinvg = 1/g, -dg/g**2
        z_xi.add_(dinvg**2 * (r_xi**2 - eta_xi / self.N) / gamma_xi)
        self.fU = - dinvg * loss.data
        r_xi.add_(self.fU * eta_xi + dinvg * n_xi.normal_() * (2 * c_xi * eta_xi / self.N).sqrt_() - dinvg**2 * z_xi * r_xi + self.temper_model.estimate_force(xi) * eta_xi)
        # resample momenta so as to avoid so large values
        if torch.abs(r_xi) >= 0.25:
            self.resample_momenta_xi()
        # udpate tempering model
        if self.temper_model_name == 'Metadynamics':
            self.temper_model.update(xi)
        elif self.temper_model_name == 'ABF':
            self.temper_model.update(xi, self.fU)
        # update the position of tempering variable
        if torch.abs(xi.add(r_xi)) >= 1:
            r_xi = - r_xi
        else:
            xi.add_(r_xi)
        # update the relevant variables of parameters
        for name, module in self.model._modules.items():
            if self.pattern1.match(name):
                w, dw = module.weight.data, module.weight.grad.data
                b, db = module.bias.data, module.bias.grad.data
                if self.version == 'approx':
                    module.z_u.add_(invg**2 * (((module.r_w**2).sum() + (module.r_b**2).sum()) / (w.numel() + b.numel()) - module.eta_theta / self.N) / module.gamma_theta)
                    module.r_w.add_(invg * (- dw) * module.eta_theta + invg * module.n_w.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_u * module.r_w)
                    module.r_b.add_(invg * (- db) * module.eta_theta + invg * module.n_b.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_u * module.r_b)
                elif self.version == 'accurate':
                    module.z_w.add_(invg**2 * ((module.r_w**2).sum() / w.numel() - module.eta_theta / self.N) / module.gamma_theta)
                    module.z_b.add_(invg**2 * ((module.r_b**2).sum() / b.numel() - module.eta_theta / self.N) / module.gamma_theta)
                    module.r_w.add_(invg * (- dw) * module.eta_theta + invg * module.n_w.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_w * module.r_w)
                    module.r_b.add_(invg * (- db) * module.eta_theta + invg * module.n_b.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_b * module.r_b)
                w.add_(module.r_w)
                b.add_(module.r_b)
            elif self.pattern2.match(name):
                wih, dwih = module.weight_ih_l0.data, module.weight_ih_l0.grad.data
                bih, dbih = module.bias_ih_l0.data, module.bias_ih_l0.grad.data
                whh, dwhh = module.weight_hh_l0.data, module.weight_hh_l0.grad.data
                bhh, dbhh = module.bias_hh_l0.data, module.bias_hh_l0.grad.data
                if self.version == 'approx':
                    module.z_u.add_(invg**2 * (((module.r_wih**2).sum() + (module.r_bih**2).sum() + (module.r_whh**2).sum() + (module.r_bhh**2).sum()) / (wih.numel() + whh.numel() + bih.numel() + bhh.numel()) - module.eta_theta / self.N) / module.gamma_theta)
                    module.v_wih.add_(invg * (- dwih) * module.eta_theta + invg * module.n_wih.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_u * module.r_wih)
                    module.v_bih.add_(invg * (- dbih) * module.eta_theta + invg * module.n_bih.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_u * module.r_bih)
                    module.v_whh.add_(invg * (- dwhh) * module.eta_theta + invg * module.n_whh.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_u * module.r_whh)
                    module.v_bhh.add_(invg * (- dbhh) * module.eta_theta + invg * module.n_bhh.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_u * module.r_bhh)
                elif self.version == 'accurate':
                    module.z_wih.add_(invg**2 * ((module.r_wih**2).sum() / wih.numel() - module.eta_theta / self.N) / module.gamma_theta)
                    module.z_bih.add_(invg**2 * ((module.r_bih**2).sum() / bih.numel() - module.eta_theta / self.N) / module.gamma_theta)
                    module.z_whh.add_(invg**2 * ((module.r_whh**2).sum() / whh.numel() - module.eta_theta / self.N) / module.gamma_theta)
                    module.z_bhh.add_(invg**2 * ((module.r_bhh**2).sum() / bhh.numel() - module.eta_theta / self.N) / module.gamma_theta)
                    module.r_wih.add_(invg * (- dwih) * module.eta_theta + invg * module.n_wih.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_wih * module.r_wih)
                    module.r_bih.add_(invg * (- dbih) * module.eta_theta + invg * module.n_bih.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_bih * module.r_bih)
                    module.r_whh.add_(invg * (- dwhh) * module.eta_theta + invg * module.n_whh.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_whh * module.r_whh)
                    module.r_bhh.add_(invg * (- dbhh) * module.eta_theta + invg * module.n_bhh.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_() - invg**2 * module.z_bhh * module.r_bhh)
                wih.add_(module.r_wih)
                bih.add_(module.r_bih)
                whh.add_(module.r_whh)
                bhh.add_(module.r_bhh)

    def resample_momenta_xi(self):
        self.model.r_xi.normal_().mul_((self.model.eta_xi / self.N).sqrt_())

    def resample_momenta(self):
        for name, module in self.model._modules.items():
            if self.pattern1.match(name):
                module.r_w.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.r_b.normal_().mul_((module.eta_theta / self.N).sqrt_())
            elif self.pattern2.match(name):
                module.r_wih.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.r_bih.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.r_whh.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.r_bhh.normal_().mul_((module.eta_theta / self.N).sqrt_())

    def g_fn(self, z):
        n = 3
        (z0, y0) = (self.standard_interval, 1)
        (z1, y1) = (1.0, 6)
        a  = (y1 - y0) / (z1 - z0)**n
        u  = torch.abs(z) - z0
        e1 = (z >= z0).float()
        e2 = (z <= -z0).float()
        g  = (a * u**n) * (e1 + e2) + 1
        dg = (a * n * u**(n - 1)) * (e1 - e2)
        return g, dg


class ABF:

    def __init__(self, zlower=-1., zupper=1., dz=1e-3, enable_cuda=False):
        self.zlower = float(zlower)
        self.zupper = float(zupper)
        self.dz = float(dz)
        self.nbins = int(np.ceil((self.zupper - self.zlower) / self.dz))
        if enable_cuda:
            self.cbins = self.cbins.cuda()
        self.memory = torch.zeros((2, self.nbins))
        if enable_cuda:
            self.memory = self.memory.cuda()
        self.nbins_end = self.nbins - 1

    def update(self, xi, fcurr):
        idx = int((xi - self.zlower) / self.dz)
        _idx = self.nbins_end - idx
        if 0 <= idx < self.nbins:
            self.memory[0, idx] += 1
            self.memory[0, _idx] += 1
            self.memory[1, idx] *= 1 - 1 / self.memory[0, idx]
            self.memory[1, idx] += - fcurr.item() / self.memory[0, idx]
            self.memory[1, _idx] *= 1 - 1 / self.memory[0, _idx]
            self.memory[1, _idx] += fcurr.item() / self.memory[0, _idx]

    def estimate_force(self, xi):
        idx = int((xi - self.zlower) / self.dz)
        return self.memory[1, idx]

    def loader(self, filename, enable_cuda):
        if enable_cuda:
            self.memory = torch.load(filename, map_location=lambda storage, loc: storage.cuda())
            print ('Successfully Loaded ABF!')
        else:
            self.memory = torch.load(filename, map_location=lambda storage, loc: storage)
            print ('Successfully Loaded ABF!')

    def saver(self, path):
        torch.save(self.memory, path+'abf.pt')
        print ('Successfully Saved ABF!')


class Metadynamics:

    def __init__(self, zlower=-1., zupper=1., dz=1e-3, gaussian_decay=1e-6, enable_cuda=False):
        self.zlower = float(zlower)
        self.zupper = float(zupper)
        self.dz = float(dz)
        self.nbins = int(np.ceil((self.zupper - self.zlower) / self.dz))
        self.cbins = self.zlower + torch.arange(self.nbins) * self.dz + 0.5 * self.dz
        if enable_cuda:
            self.cbins = self.cbins.cuda()
        self.memory = torch.zeros((2, self.nbins))
        if enable_cuda:
            self.memory = self.memory.cuda()
        self.sigma = float(dz)
        self.gaussian_decay = gaussian_decay
        self.nbins_end = self.nbins - 1

    def decayed_gaussian(self, xi, mean):
        return self.gaussian_decay * torch.exp(- 1.0 / (2 * self.sigma**2) * (xi - mean)**2)

    def update(self, xi):
        idx = int((xi - self.zlower) / self.dz)
        if 0 <= idx < self.nbins:
            self.memory[1][idx] += self.decayed_gaussian(self.cbins[idx], self.cbins[idx]).item()
            self.memory[0][idx] += 1
            _idx = self.nbins_end - idx
            if not (idx == int(self.nbins_end / 2)):
                self.memory[1][_idx] +=  self.decayed_gaussian(self.cbins[_idx], self.cbins[_idx]).item()
            if idx == 0:
                self.memory[1][idx + 1] += self.decayed_gaussian(self.cbins[idx + 1], self.cbins[idx]).item()
                self.memory[1][_idx - 1] += self.decayed_gaussian(self.cbins[_idx - 1], self.cbins[_idx]).item()
            elif 0 < idx < self.nbins_end :
                self.memory[1][idx + 1] += self.decayed_gaussian(self.cbins[idx + 1], self.cbins[idx]).item()
                self.memory[1][idx - 1] += self.decayed_gaussian(self.cbins[idx - 1], self.cbins[idx]).item()
                if not (idx == int(self.nbins_end / 2) or idx == int((self.nbins_end + 1) / 2) or idx == int((self.nbins_end - 1) / 2)):
                    self.memory[1][_idx + 1] += self.decayed_gaussian(self.cbins[_idx + 1], self.cbins[_idx]).item()
                    self.memory[1][_idx - 1] += self.decayed_gaussian(self.cbins[_idx - 1], self.cbins[_idx]).item()
            else:
                self.memory[1][idx - 1] += self.decayed_gaussian(self.cbins[idx - 1], self.cbins[idx]).item()
                self.memory[1][_idx + 1] += self.decayed_gaussian(self.cbins[_idx + 1], self.cbins[_idx]).item()

    def offset(self):
        self.memory[1].sub_(torch.min(self.memory[1]))

    def loader(self, filename, enable_cuda):
        if enable_cuda:
            self.memory = torch.load(filename, map_location=lambda storage, loc: storage.cuda())
            print ('Successfully Loaded Metadynamics!')
        else:
            self.memory = torch.load(filename, map_location=lambda storage, loc: storage)
            print ('Successfully Loaded Metadynamics!')

    def saver(self, path):
        torch.save(self.memory, path+'matadynamics.pt')
        print ('Successfully Saved Metadynamics!')

    def estimate_force(self, xi):
        idx  = int((xi - self.zlower) / self.dz)
        self.idx = idx
        if 0 <= idx < self.nbins / 2.0:
            diff = self.memory[1][idx + 1] - self.memory[1][idx]
        else:
            diff = self.memory[1][idx] - self.memory[1][idx - 1]
        return - diff / self.dz
