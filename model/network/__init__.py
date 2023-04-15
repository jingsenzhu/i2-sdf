import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import utils
from model.network.mlp import ImplicitNetwork, RenderingNetwork
from model.network.density import LaplaceDensity, AbsDensity
from model.network.ray_sampler import ErrorBoundSampler
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import DBSCAN


"""
For modeling more complex backgrounds, we follow the inverted sphere parametrization from NeRF++ 
https://github.com/Kai-46/nerfplusplus 
"""
class I2SDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.feature_vector_size
        self.scene_bounding_sphere = getattr(conf, 'scene_bounding_sphere', 1.0)

        # Foreground object's networks
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0, **conf.implicit_network)
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.rendering_network)

        self.use_light = hasattr(conf, 'light_network')
        if self.use_light:
            # self.light_network = RenderingNetwork(self.feature_vector_size, mode='nerf', output_activation='sigmoid', use_dir=False, **conf.light_network)
            self.light_network = ImplicitNetwork(0, 0, d_in=self.feature_vector_size, d_out=1, geometric_init=False, embed_type=None, output_activation='sigmoid', **conf.light_network)

        self.density = LaplaceDensity(**conf.density)

        # Background's networks
        self.use_bg = hasattr(conf, 'bg_network')
        if self.use_bg:
            bg_feature_vector_size = conf.bg_network.feature_vector_size
            self.bg_implicit_network = ImplicitNetwork(bg_feature_vector_size, 0.0, **conf.bg_network.implicit_network)
            self.bg_rendering_network = RenderingNetwork(bg_feature_vector_size, **conf.bg_network.rendering_network)
            self.bg_density = AbsDensity(**getattr(conf.bg_network, 'density', {}))
        else:
            print("[INFO] BG Network Disabled")
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, inverse_sphere_bg=self.use_bg, **conf.ray_sampler)
        self.use_normal = conf.get('use_normal', False)
        self.detach_light_feature = conf.get('detach_light_feature', True)
    
    def init_emission_groups(self, n_emitters, pointcloud, init_emission=1.0, use_dbscan=False):
        if use_dbscan:
            """
            Use DBSCAN algorithm to initialize emitter cluster centroids for K-Means from a small random batch
            Note that DBSCAN can automatically determine the number of clusters
            """
            pt_samples = pointcloud[torch.randperm(len(pointcloud))[:10000]].cpu().numpy()
            lab_samples = torch.from_numpy(DBSCAN(n_jobs=16).fit_predict(pt_samples))
            if n_emitters != len(torch.unique(lab_samples)):
                print(f"[ERROR] Inconsistent emitter count: {n_emitters} / {len(torch.unique(lab_samples))}")
                # n_emitters = len(torch.unique(lab_samples))
                exit()
            init_centroids = torch.zeros(n_emitters, 3)
            for i in range(n_emitters):
                idx = (lab_samples == i).int().argmax()
                init_centroids[i,:] = torch.from_numpy(pt_samples[idx, :])
            init_centroids = init_centroids.to(pointcloud.device)
        else:
            """
            Use K-Means plus plus to initialize emitter cluster centroids for K-Means
            """
            init_centroids = utils.kmeans_pp_centroid(pointcloud, n_emitters)
        self.emitter_clusters = KMeans(n_emitters)
        labels = self.emitter_clusters.fit_predict(pointcloud, init_centroids)
        print("[INFO] emitters clustered")
        self.emissions = nn.Parameter(torch.empty(n_emitters, 3).fill_(init_emission), True)
        return labels, self.emitter_clusters.centroids
    
    def get_param_groups(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]

    def forward(self, input, predict_only=False):

        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = utils.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs_norm = torch.linalg.vector_norm(ray_dirs, dim=1)
        ray_dirs = F.normalize(ray_dirs, dim=1)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)

        if self.use_bg:
            z_vals, z_vals_bg = z_vals
        z_max = z_vals[:,-1]
        z_vals = z_vals[:,:-1]
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        returns_grad = self.use_normal or (not self.training) or (self.rendering_network.mode == 'idr')
        # with torch.enable_grad():
        with torch.set_grad_enabled(returns_grad):
        # with torch.inference_mode(not returns_grad):
            sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat, returns_grad)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, bg_transmittance = self.volume_rendering(z_vals, z_max, sdf)

        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        weight_sum = torch.sum(weights, -1, keepdim=True)
        # dist = torch.sum(weights / weight_sum.clamp(min=1e-6) * z_vals, 1)
        dist = torch.sum(weights * z_vals, 1)
        depth_values = dist / torch.clamp(ray_dirs_norm, min=1e-6)
        # depth_values = torch.sum(weights * z_vals, 1) / torch.clamp(torch.sum(weights, 1), min=1e-6) # (bn,)

        # Background rendering
        if self.use_bg:
            N_bg_samples = z_vals_bg.shape[1]
            z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

            bg_dirs = ray_dirs.unsqueeze(1).repeat(1,N_bg_samples,1)
            bg_locs = cam_loc.unsqueeze(1).repeat(1,N_bg_samples,1)

            bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]
            bg_points_flat = bg_points.reshape(-1, 4)
            bg_dirs_flat = bg_dirs.reshape(-1, 3)

            output = self.bg_implicit_network(bg_points_flat)
            bg_sdf = output[:,:1]
            bg_feature_vectors = output[:, 1:]
            bg_rgb_flat = self.bg_rendering_network(None, None, bg_dirs_flat, bg_feature_vectors)
            bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)

            bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)

            bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)

            # Composite foreground and background
            bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
            rgb_values = fg_rgb_values + bg_rgb_values
        else:
            rgb_values = fg_rgb_values

        output = {
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'weight_sum': weight_sum
        }
        
        if self.use_light:
            light_features = F.relu(feature_vectors)
            if self.detach_light_feature:
                light_features = light_features.detach_()
            # lmask_flat = self.light_network(None, None, None, light_features)
            lmask_flat = self.light_network(light_features)
            lmask = lmask_flat.reshape(-1, N_samples, 1)
            lmask_values = torch.sum(weights.unsqueeze(-1).detach() * lmask, 1)
            output['light_mask'] = lmask_values

        if predict_only:
            return output

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3, device=cam_loc.device).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)

            # Add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            n_eik_near = eik_near_points.size(0)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)

            # Add neighbor points near surface for smoothness loss
            eik_near_neighbors = eik_near_points + torch.empty_like(eik_near_points).uniform_(-0.005, 0.005)
            eikonal_points = torch.cat([eikonal_points, eik_near_neighbors], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta[:n_eik_points+n_eik_near,]
            normals = grad_theta[n_eik_points:,]
            normals = F.normalize(normals, dim=1, eps=1e-6)
            diff_norm = torch.norm(normals[:n_eik_near,:] - normals[n_eik_near:,:], dim=1)
            output['diff_norm'] = diff_norm

            # Sample pointclouds for bubble loss
            if 'pointcloud' in input:
                surface_points = input['pointcloud']
                cam_loc_selected = cam_loc[np.random.randint(0, len(cam_loc)),:]
                surface_points = torch.cat([surface_points, cam_loc_selected.unsqueeze(0)], dim=0)
                surface_sdf = self.implicit_network.get_sdf_vals(surface_points)
                output['surface_sdf'] = surface_sdf[:-1,:]

            # Accumulate gradients for normal loss
            if self.use_normal:
                normals = F.normalize(gradients, dim=-1)
                normals = normals.reshape(-1, N_samples, 3)
                normal_map = torch.sum(weights.unsqueeze(-1).detach() * normals, 1)
                normal_map = F.normalize(normal_map, dim=-1)
                output['normal_values'] = normal_map

        # elif not self.training:
        else:
            # Accumulate gradients for normal visualization
            gradients = gradients.detach()
            normals = F.normalize(gradients, dim=-1)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            normal_map = F.normalize(normal_map, dim=-1)
            output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, z_max, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples

        # included also the dist from the sphere intersection
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=free_energy.device), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1]) # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat([bg_dists, torch.tensor([1e10], device=bg_dists.device).unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1, device=bg_free_energy.device), bg_free_energy[:, :-1]], dim=-1)  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        bg_weights = bg_alpha * bg_transmittance # probability of the ray hits something here

        return bg_weights

    def depth2pts_outside(self, ray_o, ray_d, depth):

        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.scene_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.scene_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts


class I2SDFLoss(nn.Module):
    def __init__(self, eikonal_weight=0.1, smooth_weight=0.0, mask_weight=0.0, depth_weight=0.1, normal_weight=0.05, angular_weight=0.05, bubble_weight=0.0, min_bubble_iter=0, max_bubble_iter=None, smooth_iter=None, light_mask_weight=0.0, eikonal_weight_bubble=0.0):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_loss = F.l1_loss
        self.smooth_weight = smooth_weight
        self.mask_weight = mask_weight
        self.depth_weight = depth_weight
        self.normal_weight = normal_weight
        self.angular_weight = angular_weight
        self.bubble_weight = bubble_weight
        # self.eikonal_weight_bubble = eikonal_weight_bubble if eikonal_weight_bubble else self.eikonal_weight
        self.min_bubble_iter = min_bubble_iter
        self.max_bubble_iter = max_bubble_iter
        self.smooth_iter = smooth_iter
        if self.bubble_weight > 0 and self.max_bubble_iter is not None and self.smooth_iter < self.max_bubble_iter:
            self.smooth_iter = self.max_bubble_iter # Disable smoothness loss during bubble steps
        self.light_mask_weight = light_mask_weight

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, mask_pred, mask_gt):
        return F.binary_cross_entropy(mask_pred.clip(1e-3, 1.0 - 1e-3), mask_gt)
    
    def get_depth_loss(self, depth, depth_gt, depth_mask):
        depth_gt = depth_gt.flatten()
        depth_mask = depth_mask.flatten()
        # TODO: Add support for scale invariant depth loss (like MonoSDF)
        return F.mse_loss(depth[depth_mask], depth_gt[depth_mask])
    
    def get_normal_l1_loss(self, normal, normal_gt, normal_mask):
        normal_gt = normal_gt.reshape(-1, 3)
        normal_mask = normal_mask.flatten()
        return torch.abs(1 - torch.sum(normal[normal_mask] * normal_gt[normal_mask], dim=-1)).mean()
    
    def get_normal_angular_loss(self, normal, normal_gt, normal_mask):
        normal_gt = normal_gt.reshape(-1, 3)
        normal_mask = normal_mask.flatten()
        dot = torch.sum(normal[normal_mask] * normal_gt[normal_mask], dim=-1)
        angle = torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) / math.tau
        return angle.clamp_max(0.5).abs().mean()

    def forward(self, model_outputs, ground_truth, current_step):
        rgb_gt = ground_truth['rgb']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
        
        smooth_activated = self.smooth_iter is None or current_step > self.smooth_iter
        if smooth_activated and self.smooth_weight > 0 and 'diff_norm' in model_outputs:
            smooth_loss = model_outputs['diff_norm'].mean()
        else:
            smooth_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
        
        if 'mask' in ground_truth and self.mask_weight > 0:
            mask_loss = self.get_mask_loss(model_outputs['weight_sum'], ground_truth['mask'])
        else:
            mask_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
        
        if 'depth' in ground_truth and self.depth_weight > 0:
            depth_loss = self.get_depth_loss(model_outputs['depth_values'], ground_truth['depth'], ground_truth['depth_mask'])
        else:
            depth_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
        
        if 'normal' in ground_truth and self.normal_weight > 0:
            normal_loss = self.get_normal_l1_loss(model_outputs['normal_values'], ground_truth['normal'], ground_truth['normal_mask'])
        else:
            normal_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()

        if 'normal' in ground_truth and self.angular_weight > 0:
            angular_loss = self.get_normal_l1_loss(model_outputs['normal_values'], ground_truth['normal'], ground_truth['normal_mask'])
        else:
            angular_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
        
        if 'surface_sdf' in model_outputs and self.bubble_weight > 0:
            bubble_loss = model_outputs['surface_sdf'].abs().mean()
        else:
            bubble_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
        
        if 'light_mask' in model_outputs and self.light_mask_weight > 0:
            light_mask_loss = self.get_mask_loss(model_outputs['light_mask'].reshape(-1, 1), ground_truth['light_mask'].reshape(-1, 1))
        else:
            light_mask_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()

        loss = rgb_loss + \
                self.eikonal_weight * eikonal_loss + \
                 self.smooth_weight * smooth_loss + \
                  self.mask_weight * mask_loss + \
                   self.depth_weight * depth_loss + \
                    self.normal_weight * normal_loss + \
                     self.angular_weight * angular_loss + \
                      self.bubble_weight * bubble_loss + \
                       self.light_mask_weight * light_mask_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'mask_loss': mask_loss,
            'depth_loss': depth_loss,
            'normal_loss': normal_loss,
            'angular_loss': angular_loss,
            'bubble_loss': bubble_loss,
            'light_mask_loss': light_mask_loss
        }

        return output
