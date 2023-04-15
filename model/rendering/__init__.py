import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import cv2
from .brdf import *


class RenderingLayer(nn.Module):
    def __init__(self, spp, split_n_pixels, preserve_light=True) -> None:
        super().__init__()
        self.spp = spp
        self.split_n_pixels = split_n_pixels
        self.preserve_light = preserve_light
    
    def forward(
            self,
            model,
            surface_points,
            view_direction,
            Kd,
            Ks,
            normal,
            rough,
            radiance_scale=None,
            intersect_func=None
        ):
        """
        Render according to material, normal and lighting conditions
        Params:
            model: NeRF model to predict radiance
            surface_points, view_direction, albedo, normal, rough, metal: (bn, c)
        """
        bn = normal.size(0)

        cx, cy, cz = create_frame(normal)
        wi_x = torch.sum(cx*view_direction, dim=1)
        wi_y = torch.sum(cy*view_direction, dim=1)
        wi_z = torch.sum(cz*view_direction, dim=1)
        wi = torch.stack([wi_x, wi_y, wi_z], dim=1)
        wi_mask = (wi[:,2] >= 0.00001)
        wi[:,2,...] = torch.where(wi[:,2,...] < 0.00001, torch.ones_like(wi[:,2,...]) * 0.00001, wi[:,2,...])
        wi = F.normalize(wi, dim=1, eps=1e-6)
        # wi_mask = torch.where(wi[:,2:3,...] < 0, torch.zeros_like(wi[:,2:3,...]), torch.ones_like(wi[:,2:3,...]))
        wi = wi.unsqueeze(1) # (bn, 1, 3)

        # with torch.no_grad():
        if True:
            samples = torch.rand(bn, self.spp, 3, device=normal.device)
            pS = probabilityToSampleSpecular(Kd, Ks)
            clamp_value = 0.0
            pS.clamp_min_(clamp_value)
            sample_diffuse = samples[:,:,0] >= pS

            ls_diffuse = square_to_cosine_hemisphere(samples[:,:,1:])
            ls_specular = sample_ggx_specular(samples[:,:,1:], rough, wi)
            wo = torch.where(sample_diffuse.unsqueeze(2).expand(bn, self.spp, 3), ls_diffuse, ls_specular) # (bn, spp, 3)
        pdfs = pdf_ggx(Kd, Ks, rough, wi, wo, clamp_value).unsqueeze(2)
        eval_diff, eval_spec, wo_mask = eval_ggx(Kd, Ks, rough, wi, wo)
        # wo_mask = torch.all(wo_mask, dim=1)

        direction = to_global(wo, cx.unsqueeze(1), cy.unsqueeze(1), cz.unsqueeze(1))

        # surface_points = surface_points + 0.01 * view_direction # prevent self-intersection
        surface_points = surface_points.unsqueeze(1).expand_as(direction).reshape(-1, 3)
        direction = direction.reshape(-1, 3)
        surface_points = surface_points + direction * 0.01 # prevent self-intersection
        
        pts_splits = torch.split(surface_points, self.split_n_pixels, dim=0)
        dirs_splits = torch.split(direction, self.split_n_pixels, dim=0)
        radiance = []
        # with torch.no_grad():
        for pts, dirs in zip(pts_splits, dirs_splits):
            radiance.append(model.get_incident_radiance(pts, dirs, intersect_func))
        radiance = torch.cat(radiance, dim=0)

        radiance = radiance.view(bn, self.spp, 3)
        if radiance_scale is not None:
            radiance = radiance * radiance_scale[None,None,:]
        pdfs = torch.clamp(pdfs, min=0.00001)
        ndl = torch.clamp(wo[:,:,2:], min=0)

        brdfDiffuse = eval_diff.expand(bn, self.spp, 3) * ndl / pdfs
        colorDiffuse = torch.mean(brdfDiffuse * radiance, dim=1)
        brdfSpec = eval_spec.expand(bn, self.spp, 3) * ndl / pdfs
        colorSpec = torch.mean(brdfSpec * radiance, dim=1)

        return colorDiffuse, colorSpec, wi_mask


