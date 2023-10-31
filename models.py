import time

import torch
from torch import nn, Tensor
from torchvision import models as tvmodels

from einops import rearrange


class PAC_Cell(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True,
                 rnn_hdim: int = 128):
        super(PAC_Cell, self).__init__()

        assert model_name in {'PAC_Net', 'P_Net', 'C_Net', 'baseline'}
        self.rnn_hdim = rnn_hdim

        self.backbone_builder = {
            'PAC_Net': tvmodels.resnet18,
            'P_Net': tvmodels.resnet34,
            'C_Net': tvmodels.resnet34,
            'baseline': tvmodels.resnet50,
        }[model_name]
        self.backbone_weight = {
            'PAC_Net': tvmodels.ResNet18_Weights.DEFAULT,
            'P_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'C_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'baseline': tvmodels.ResNet50_Weights.DEFAULT,
        }[model_name] if pretrained else None

        self.rnn_cell = nn.GRUCell

        self.p_cell = self.cell_builder()
        self.c_cell = self.cell_builder()

    def forward(self, h: Tensor, frames: tuple[Tensor, Tensor]):
        diff_frame, frame = frames
        h = self.propagate(h, diff_frame)
        h = self.calibrate(h, frame)
        return h

    def propagate(self, h: Tensor, diff_frame: Tensor):
        feature = self.p_cell['feature_extractor'][diff_frame]
        return self.p_cell['rnn_cell'](input=feature, hx=h)

    def calibrate(self, h: Tensor, frame: Tensor):
        feature = self.c_cell['feature_extractor'][frame]
        return self.c_cell['rnn_cell'](input=feature, hx=h)

    def cell_builder(self):
        backbone = self.backbone_builder(weights=self.weight, progress=True)
        # backbone = self.backbone_builder(progress=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, self.rnn_hdim)
        return nn.ModuleDict({
            'feature_extractor': backbone,
            'rnn_cell': self.rnn_cell(input_size=self.rnn_hdim, hidden_size=self.rnn_hdim)
        })


class PAC_Net_Base(nn.Module):
    def __init__(self, model_name: str, pretrained: bool,
                 rnn_type: str = 'gru', rnn_hdim: int = 128,
                 v_loss: bool = True, **kwargs):
        super(PAC_Net_Base, self).__init__()

        self.rnn_hdim = rnn_hdim
        self.v_loss = v_loss

        assert model_name in {'PAC_Net', 'P_Net', 'C_Net', 'baseline'}
        # CNN
        self.backbone_builder = {
            'PAC_Net': tvmodels.resnet18,
            'P_Net': tvmodels.resnet34,
            'C_Net': tvmodels.resnet34,
            'baseline': tvmodels.resnet50,
        }[model_name]
        self.backbone_weight = {
            'PAC_Net': tvmodels.ResNet18_Weights.DEFAULT,
            'P_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'C_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'baseline': tvmodels.ResNet50_Weights.DEFAULT,
        }[model_name] if pretrained else None

        # RNN
        rnn_dict = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
        } if model_name == 'PAC_Net' else {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM}
        self.rnn_builder = rnn_dict[rnn_type]

        # MLP
        mlp_dim = rnn_hdim // 2
        act = nn.Tanh if model_name == 'P_Net' else nn.Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(rnn_hdim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 2),
            act()
        )

        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, Ix: tuple):
        raise NotImplementedError

    def vis_forward(self, Ix: tuple, **kwargs):
        return self.forward(Ix)[1].detach().cpu()

    def compute_v_loss(self, x_pred: Tensor, x_gt: Tensor):
        v_pred = torch.sub(x_pred[:, 1:], x_pred[:, :-1])
        v_gt = torch.sub(x_gt[:, 1:], x_gt[:, :-1])

        return self.criterion(v_pred, v_gt)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class PAC_Net(PAC_Net_Base):
    def __init__(self, pretrained: bool = True,
                 rnn_type: str = 'gru', rnn_hdim: int = 128,
                 v_loss: bool = True, warm_up: int = 32):
        super(PAC_Net, self).__init__(model_name='PAC_Net', pretrained=pretrained,
                                      rnn_type=rnn_type, rnn_hdim=rnn_hdim, v_loss=v_loss)
        self.warmup_frames = warm_up

        # predict module
        self.c_encoder, self.c_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                       self.rnn_builder, rnn_hdim)
        self.p_encoder, self.p_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                       self.rnn_builder, rnn_hdim)
        init_modules = [self.c_encoder.fc, self.p_encoder.fc, self.decoder]

        if self.warmup_frames > 0:
            # warmup module
            self.warmup_c_encoder, self.warmup_c_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                                         self.rnn_builder, rnn_hdim)
            self.warmup_p_encoder, self.warmup_p_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                                         self.rnn_builder, rnn_hdim)
            init_modules.extend([self.warmup_c_encoder.fc, self.warmup_p_encoder.fc])

        for m in init_modules:
            m.apply(self._init_weights)

    def forward(self, Ix: tuple):
        I, x_gt = Ix  # (B, C, T, H, W)
        delta_I = torch.sub(I[:, :, 1:], I[:, :, :-1]).float()
        B, T = x_gt.shape[:2]
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            warmup_delta_I, delta_I = delta_I[:, :, :self.warmup_frames], delta_I[:, :, self.warmup_frames:]
            hv_t = self.warm_up(warmup_I, warmup_delta_I)
        else:
            hv_t = None

        fx = self.c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        Hx = torch.zeros(B, T, self.rnn_hdim, device=x_gt.device)
        for t in range(T):
            hx_t = self.c_cell(input=fx[t], hx=hv_t)
            Hx[:, t] = hx_t
            if t < T - 1:
                hv_t = self.p_cell(input=fv[t], hx=hx_t)

        x_pred = self.decoder(Hx)  # * self.factor
        loss_x = self.criterion(x_pred, x_gt[:, self.warmup_frames:])
        loss_v = self.compute_v_loss(x_pred, x_gt[:, self.warmup_frames:]) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred

    def warm_up(self, I: Tensor, delta_I: Tensor):
        B, T = I.size(0), I.size(2)
        fx = self.warmup_c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.warmup_p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
        for t in range(T):
            hx_t = self.warmup_c_cell(input=fx[t], hx=hv_t)
            hv_t = self.warmup_p_cell(input=fv[t], hx=hx_t)

        return hv_t

    @staticmethod
    def _make_model(backbone_builder, weight, rnn_cell, rnn_hdim):
        encoder = backbone_builder(weights=weight, progress=True)
        # encoder = backbone_builder(progress=True)
        encoder.fc = nn.Linear(encoder.fc.in_features, rnn_hdim)

        return encoder, rnn_cell(input_size=rnn_hdim, hidden_size=rnn_hdim)

    def vis_forward(self, Ix: tuple, phase: str = 'x'):
        I, x_gt = Ix  # (B, C, T, H, W)
        delta_I = torch.sub(I[:, :, 1:], I[:, :, :-1]).float()
        B, T = x_gt.shape[:2]
        H = torch.zeros(B, T, self.rnn_hdim, device=I.device)

        # warm up
        tic = time.time()
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            warmup_delta_I, delta_I = delta_I[:, :, :self.warmup_frames], delta_I[:, :, self.warmup_frames:]
            warmup_fx = self.warmup_c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
            warmup_fx = rearrange(warmup_fx, '(b t) d -> t b d', b=B)

            warmup_fv = self.warmup_p_encoder(rearrange(warmup_delta_I, 'b c t h w -> (b t) c h w'))
            warmup_fv = rearrange(warmup_fv, '(b t) d -> t b d', b=B)

            hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
            for t in range(self.warmup_frames):
                hx_t = self.warmup_c_cell(input=warmup_fx[t], hx=hv_t)
                hv_t = self.warmup_p_cell(input=warmup_fv[t], hx=hx_t)
                H[:, t] = hx_t if phase == 'x' else hv_t
        else:
            hv_t = None

        # tracking
        fx = self.c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        Hx = torch.zeros(B, T, self.rnn_hdim, device=x_gt.device)
        for t in range(T):
            hx_t = self.c_cell(input=fx[t], hx=hv_t)
            Hx[:, t] = hx_t
            if t < T - 1:
                hv_t = self.p_cell(input=fv[t], hx=hx_t)
            H[:, t + self.warmup_frames] = hx_t if phase == 'x' else hv_t

        x_pred = self.decoder(H) * self.factor
        toc = time.time()
        fps = 320/(toc-tic)

        return x_pred.detach().cpu()  # , fps


class P_Net(PAC_Net_Base):
    def __init__(self, pretrained: bool = True,
                 rnn_type: str = 'gru', rnn_hdim: int = 128, rnn_layer: int = 2,
                 v_loss: bool = True, **kwargs):
        super(P_Net, self).__init__(model_name='P_Net', pretrained=pretrained,
                                    rnn_type=rnn_type, rnn_hdim=rnn_hdim, v_loss=v_loss)

        self.encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, rnn_hdim)

        self.rnn = self.rnn_builder(input_size=rnn_hdim, hidden_size=rnn_hdim, num_layers=rnn_layer, batch_first=True)

        for m in [self.encoder.fc, self.decoder]:
            m.apply(self._init_weights)

    def forward(self, Ix: tuple):
        ori_I, x_gt = Ix  # (B, C, T, H, W)
        delta_I = torch.sub(ori_I[:, :, 1:], ori_I[:, :, :-1]).float()
        B, T = x_gt.shape[:2]

        delta_I = rearrange(delta_I, 'b c t h w -> (b t) c h w')
        fv = self.encoder(delta_I)
        fv = rearrange(fv, '(b t) d -> b t d', b=B)

        Hv = self.rnn(fv)[0]
        v_pred = self.decoder(Hv)  # * self.factor  # (B, T, 2)
        x_pred = torch.zeros_like(x_gt)
        x_pred[:, 0] = x_gt[:, 0]
        for t in range(T - 1):
            x_pred[:, t + 1] = x_pred[:, t] + v_pred[:, t]

        loss_x = self.criterion(x_pred, x_gt)
        loss_v = self.compute_v_loss(v_pred, x_gt) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred

    def compute_v_loss(self, v_pred: Tensor, x_gt: Tensor):
        v_gt = torch.sub(x_gt[:, 1:], x_gt[:, :-1])
        return self.criterion(v_pred, v_gt)


class C_Net(PAC_Net_Base):
    def __init__(self, pretrained: bool = False,
                 rnn_type: str = 'gru', rnn_hdim: int = 128, rnn_layer: int = 2,
                 v_loss: bool = True, warm_up: int = 32):
        super(C_Net, self).__init__(model_name='C_Net', pretrained=pretrained,
                                    rnn_type=rnn_type, rnn_hdim=rnn_hdim,
                                    v_loss=v_loss)
        self.warmup_frames = warm_up

        self.encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, rnn_hdim)
        self.rnn = self.rnn_builder(input_size=rnn_hdim, hidden_size=rnn_hdim, num_layers=rnn_layer,
                                    batch_first=True)
        init_modules = [self.encoder.fc, self.decoder]

        if self.warmup_frames > 0:
            self.warmup_encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
            self.warmup_encoder.fc = nn.Linear(self.warmup_encoder.fc.in_features, rnn_hdim)
            self.warmup_rnn = self.rnn_builder(input_size=rnn_hdim, hidden_size=rnn_hdim, num_layers=rnn_layer,
                                               batch_first=True)
            init_modules.append(self.warmup_encoder.fc)

        for m in init_modules:
            m.apply(self._init_weights)

    def forward(self, Ix):
        I, x_gt = Ix
        B, T = x_gt.shape[:2]
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            hx = self.warm_up(warmup_I.float())
        else:
            hx = None

        fx = self.encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)
        Hx = self.rnn(fx, hx)[0]  # (B, T, D)
        x_pred = self.decoder(Hx)  # * self.factor  # (B, T, 2)

        loss_x = self.criterion(x_pred, x_gt[:, self.warmup_frames:])
        loss_v = self.compute_v_loss(x_pred, x_gt[:, self.warmup_frames:]) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred

    def vis_forward(self, Ix: tuple, **kwargs):
        I, x_gt = Ix
        B, T = x_gt.shape[:2]
        H = torch.zeros(B, T, self.rnn_hdim, device=x_gt.device)

        # warmup
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            warmup_fx = self.warmup_encoder(rearrange(warmup_I.float(), 'b c t h w -> (b t) c h w'))
            warmup_fx = rearrange(warmup_fx, '(b t) d -> b t d', b=B)
            Hx, hx = self.warmup_rnn(warmup_fx)
            H[:, :self.warmup_frames] = Hx
        else:
            hx = None

        # tracking
        fx = self.encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)
        Hx = self.rnn(fx, hx)[0]  # (B, T, D)
        H[:, self.warmup_frames:] = Hx
        x_pred = self.decoder(H) * self.factor  # (B, T, 2)

        return x_pred.detach().cpu()

    def warm_up(self, I: Tensor):
        B = I.size(0)
        fx = self.warmup_encoder(rearrange(I, 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)

        return self.warmup_rnn(fx)[1]


class NLOS_baseline(PAC_Net_Base):
    def __init__(self, pretrained: bool = False, rnn_hdim=128,
                 v_loss: bool = True, **kwargs):
        super(NLOS_baseline, self).__init__(model_name='C_Net', pretrained=pretrained, rnn_hdim=rnn_hdim,
                                            v_loss=v_loss)

        self.encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, rnn_hdim)

        init_modules = [self.encoder.fc, self.decoder]
        for m in init_modules:
            m.apply(self._init_weights)

    def forward(self, Ix):
        I, x_gt = Ix
        B, T = x_gt.shape[:2]

        fx = self.encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)
        x_pred = self.decoder(fx)  # * self.factor  # (B, T, 2)

        loss_x = self.criterion(x_pred, x_gt)
        loss_v = self.compute_v_loss(x_pred, x_gt) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred
