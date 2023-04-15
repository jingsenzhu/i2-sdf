import torch.nn as nn
import numpy as np

import utils
from .embedder import *
from .density import LaplaceDensity
from .ray_sampler import ErrorBoundSampler


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            embed_type=None,
            sphere_scale=1.0,
            output_activation=None,
            **kwargs
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if embed_type:
            embed_fn, input_ch = get_embedder(embed_type, input_dims=d_in, **kwargs)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        
        print(f"[INFO] Implicit network dims: {dims}")

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
                if out_dim < 0:
                    print(dims)
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif (embed_type or self.use_grid) and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif (embed_type or self.use_grid) and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.output_activation = None
        if output_activation is not None:
            self.output_activation = activations[output_activation]

    def get_param_groups(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]

    def forward(self, input):

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def feature(self, x):
        return self.forward(x)[:,1:]

    def get_outputs(self, x, returns_grad=True):
        x.requires_grad_(returns_grad)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        if returns_grad:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            return sdf, feature_vectors, gradients
        else:
            return sdf, feature_vectors, None

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

activations = {
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(),
    'softplus': nn.Softplus()
}

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            embed_type=None,
            embed_point=None,
            output_activation='sigmoid',
            **kwargs
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.d_out = d_out

        self.embedview_fn = None
        if embed_type:
            embedview_fn, input_ch = get_embedder(embed_type, input_dims=3, **kwargs)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        
        if mode == 'idr':
            self.embedpoint_fn = None
            if embed_point is not None:
                embedpoint_fn, input_ch = get_embedder(input_dims=3, **embed_point)
                self.embedpoint_fn = embedpoint_fn
                dims[0] += (input_ch - 3)

        print(f"[INFO] Rendering network dims: {dims}")
        self.num_layers = len(dims)
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()
        self.output_activation = activations[output_activation]

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        # elif self.mode == 'nerf':
        else:
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        x = self.output_activation(x)

        return x

