import torch
import re


class SGHMC:

    def __init__(self, model, N, eta_theta0, c_theta0):
        self.N = N
        self.model = model
        self.pattern1 = re.compile(r'linear|conv')
        self.pattern2 = re.compile(r'lstm')
        for name, module in self.model._modules.items():
            if self.pattern1.match(name):
                size_w = module.weight.data.shape
                size_b = module.bias.data.shape
                module.register_buffer('v_w', torch.zeros(size_w))
                module.register_buffer('v_b', torch.zeros(size_b))
                module.register_buffer('eta_theta', torch.Tensor([eta_theta0*self.N]))
                module.register_buffer('c_theta', torch.Tensor([c_theta0]))
                module.register_buffer('n_w', torch.zeros(size_w))
                module.register_buffer('n_b', torch.zeros(size_b))
            elif self.pattern2.match(name):
                size_wih = module.weight_ih_l0.data.shape
                size_bih = module.bias_ih_l0.data.shape
                size_whh = module.weight_hh_l0.data.shape
                size_bhh = module.bias_hh_l0.data.shape
                module.register_buffer('v_wih', torch.zeros(size_wih))
                module.register_buffer('v_bih', torch.zeros(size_bih))
                module.register_buffer('v_whh', torch.zeros(size_whh))
                module.register_buffer('v_bhh', torch.zeros(size_bhh))
                module.register_buffer('eta_theta', torch.Tensor([eta_theta0*self.N]))
                module.register_buffer('c_theta', torch.Tensor([c_theta0]))
                module.register_buffer('n_wih', torch.zeros(size_wih))
                module.register_buffer('n_bih', torch.zeros(size_bih))
                module.register_buffer('n_whh', torch.zeros(size_whh))
                module.register_buffer('n_bhh', torch.zeros(size_bhh))

    def update(self):
        for name, module in self.model._modules.items():
            if self.pattern1.match(name):
                w, dw = module.weight.data, module.weight.grad.data
                b, db = module.bias.data, module.bias.grad.data
                module.v_w.add_(- dw * module.eta_theta - module.c_theta * module.v_w + module.n_w.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_())
                module.v_b.add_(- db * module.eta_theta - module.c_theta * module.v_b + module.n_b.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_())
                w.add_(module.v_w)
                b.add_(module.v_b)
            elif self.pattern2.match(name):
                wih, dwih = module.weight_ih_l0.data, module.weight_ih_l0.grad.data
                bih, dbih = module.bias_ih_l0.data, module.bias_ih_l0.grad.data
                whh, dwhh = module.weight_hh_l0.data, module.weight_hh_l0.grad.data
                bhh, dbhh = module.bias_hh_l0.data, module.bias_hh_l0.grad.data
                module.v_wih.add_(- dwih * module.eta_theta - module.c_theta * module.v_wih + module.n_wih.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_())
                module.v_bih.add_(- dbih * module.eta_theta - module.c_theta * module.v_bih + module.n_bih.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_())
                module.v_whh.add_(- dwhh * module.eta_theta - module.c_theta * module.v_whh + module.n_whh.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_())
                module.v_bhh.add_(- dbhh * module.eta_theta - module.c_theta * module.v_bhh + module.n_bhh.normal_() * (2 * module.c_theta * module.eta_theta / self.N).sqrt_())
                wih.add_(module.v_wih)
                bih.add_(module.v_bih)
                whh.add_(module.v_whh)
                bhh.add_(module.v_bhh)

    def resample_momenta(self):
        for name, module in self.model._modules.items():
            if self.pattern1.match(name):
                module.v_w.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.v_b.normal_().mul_((module.eta_theta / self.N).sqrt_())
            elif self.pattern2.match(name):
                module.v_wih.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.v_bih.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.v_whh.normal_().mul_((module.eta_theta / self.N).sqrt_())
                module.v_bhh.normal_().mul_((module.eta_theta / self.N).sqrt_())
