import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F


def linear_to_srgb(data):
    return torch.where(data <= 0.0031308, data * 12.92, 1.055 * (data ** (1 / 2.4)) - 0.055)


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    # psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())
    psnr = -10. * torch.log(mse) / np.log(10)

    return psnr


def load_rgb(path, normalize_rgb = False, is_hdr = False):
    if not is_hdr:
        img = imageio.imread(path)
        img = skimage.img_as_float32(img)
    else:
        img = cv2.imread(path, -1)[:,:,::-1].copy()

    if normalize_rgb: # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def load_mask(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return img # (h, w)


def load_depth(path):
    img = cv2.imread(path, -1)
    if len(img.shape) == 3:
        img = img[:,:,-1]
    return img

def load_normal(path):
    img = cv2.imread(path, -1)[:,:,::-1]
    return img.copy()


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def depth_to_world(uv, intrinsics, pose, depth, depth_mask=None):
    x_cam, y_cam = torch.unbind(uv, dim=1)
    z_cam = torch.ones_like(x_cam)
    xyz_view = lift(x_cam, y_cam, z_cam, intrinsics)
    xyz_view[:,:-1] = xyz_view[:,:-1] * depth.unsqueeze(1)
    if depth_mask is not None:
        xyz_view = xyz_view[depth_mask,:]
    xyz_world = pose @ xyz_view.T
    return xyz_world.T


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4, device=pose.device).repeat(pose.shape[0],1,1).float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples), device=pose.device)
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    # z_cam = -depth.view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)
    # pixel_points_cam[:,:,0] = -pixel_points_cam[:,:,0]
    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    # ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:,:4].detach())
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    sk = intrinsics[..., 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3), device=q.device)
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_general_sphere_intersections(cam_loc, ray_directions, center, r):
    n_rays = cam_loc.size(0)
    # print(cam_loc.shape, ray_directions.shape)
    cam_loc = cam_loc - center.unsqueeze(0)
    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)
    intersect_mask = (under_sqrt >= 0).squeeze(-1) # (n_rays,)
    under_sqrt = under_sqrt[intersect_mask,:]
    ray_cam_dot = ray_cam_dot[intersect_mask,:]
    sphere_intersections = torch.sqrt(under_sqrt) * torch.tensor([-1, 1], device=cam_loc.device).float() - ray_cam_dot
    front_mask = (sphere_intersections > 0).all(dim=-1)
    intersect_mask[intersect_mask.clone()] &= front_mask
    sphere_intersections = sphere_intersections[front_mask,:]
    intersection_normals = cam_loc[intersect_mask,:] + ray_directions[intersect_mask,:] * sphere_intersections[:,:1]
    intersection_points = intersection_normals + center.unsqueeze(0)
    intersection_normals = F.normalize(intersection_normals, dim=1, eps=1e-8)
    return intersection_points, intersection_normals, intersect_mask


def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.tensor([-1, 1], device=cam_loc.device).float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections

def add_depth_noise(depth, depth_mask, scale=1):
    mu = 0.0001125 * depth**2 + 0.0048875
    sigma = 0.002925 * depth**2 + 0.003325
    noise = torch.randn_like(depth) * sigma + mu
    return (depth + noise * scale) * depth_mask