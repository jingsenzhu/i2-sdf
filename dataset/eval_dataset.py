from copy import deepcopy
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import utils.plots as plt
import torch.nn.functional as F
import utils
from utils import rend_util
from tqdm.contrib import tzip, tenumerate
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import cv2

class GridDataset(Dataset):
    """
    Used for mesh extraction
    """
    def __init__(self, points, xyz) -> None:
        super().__init__()
        self.grid_points = points
        self.xyz = xyz
    
    def __len__(self):
        return self.grid_points.size(0)
    
    def __getitem__(self, index):
        return self.grid_points[index]


class PlotDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 plot_nimgs,
                 scan_id=0,
                 is_val=False,
                 data=None,
                 is_hdr=False,
                 indices=None,
                 use_lmask=False,
                 **kwargs
                 ):

        self.instance_dir = os.path.join('data', data_dir, 'scan{0}'.format(scan_id))
        val_dir = '{0}/val'.format(self.instance_dir)
        is_val = is_val and os.path.exists(val_dir)
        lmask_dir = '{0}/light_mask'.format(self.instance_dir)
        self.use_lmask = use_lmask and os.path.exists(lmask_dir)
        if is_val:
            print("[INFO] Validation set detected")
        if is_val or data is None:
            assert os.path.exists(self.instance_dir), "Data directory is empty"

            if is_val:
                image_dir = val_dir
            elif is_hdr:
                image_dir = '{0}/hdr'.format(self.instance_dir)
            else:
                image_dir = '{0}/image'.format(self.instance_dir)
            image_paths = sorted(utils.glob_imgs(image_dir))
            if indices is not None:
                print(f"[INFO] Selecting indices: {indices}")
                image_paths = [image_paths[i] for i in indices]
            self.n_images = len(image_paths)
            self.indices = indices if indices is not None else list(range(self.n_images))

            self.cam_file = '{0}/cameras_normalize.npz'.format(self.instance_dir)
            camera_dict = np.load(self.cam_file)
            scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.indices] if not is_val else [camera_dict['scale_mat_0'].astype(np.float32)] * len(self.indices)
            world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.indices] if not is_val else [camera_dict['val_mat_%d' % idx].astype(np.float32) for idx in self.indices]

            self.intrinsics_all = []
            self.pose_all = []
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())
            self.intrinsics_all = torch.stack(self.intrinsics_all, 0)
            self.pose_all = torch.stack(self.pose_all, 0)
            self.rgb_images = []
            for path in image_paths:
                rgb = rend_util.load_rgb(path, is_hdr=is_hdr)
                self.img_res = [rgb.shape[1], rgb.shape[2]]
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())
            self.rgb_images = torch.stack(self.rgb_images, 0)
            if self.use_lmask:
                self.lightmask_images = []
                lmask_paths = sorted(utils.glob_imgs(lmask_dir))
                for path in lmask_paths:
                    lmask = rend_util.load_mask(path)
                    lmask = lmask.reshape(-1, 1)
                    self.lightmask_images.append(torch.from_numpy(lmask).float())
                self.lightmask_images = torch.stack(self.lightmask_images, 0)
            self.total_pixels = self.rgb_images.size(1)
        else:
            self.intrinsics_all = data['intrinsics']
            self.pose_all = data['pose']
            self.rgb_images = data['rgb']
            self.n_images = len(self.rgb_images)
            self.img_res = [data['img_res'][0], data['img_res'][1]]
            self.total_pixels = self.img_res[0] * self.img_res[1]
            if 'light_mask' in data:
                self.lightmask_images = data['light_mask']
                self.use_lmask = True
        
        if (scale := kwargs.get('downsample', 1)) > 1:
            old_img_res = deepcopy(self.img_res)
            self.img_res[0] //= scale
            self.img_res[1] //= scale
            self.total_pixels = self.img_res[0] * self.img_res[1]
            self.rgb_images = self.rgb_images.transpose(1, 2).reshape(-1, 3, old_img_res[0], old_img_res[1])
            self.rgb_images = F.interpolate(self.rgb_images, self.img_res, mode='area')
            self.rgb_images = self.rgb_images.reshape(-1, 3, self.total_pixels).transpose(1, 2)
            if self.use_lmask:
                self.lightmask_images = self.lightmask_images.transpose(1, 2).reshape(-1, 1, old_img_res[0], old_img_res[1])
                self.lightmask_images = F.interpolate(self.lightmask_images, self.img_res, mode='area')
                self.lightmask_images = self.lightmask_images.reshape(-1, 1, self.total_pixels).transpose(1, 2)

            self.intrinsics_all = self.intrinsics_all.clone()
            self.intrinsics_all[:,0,0] /= scale
            self.intrinsics_all[:,1,1] /= scale
            self.intrinsics_all[:,0,2] /= scale
            self.intrinsics_all[:,1,2] /= scale
        
        print(f"[INFO] Plot image size: {self.img_res[1]}x{self.img_res[0]}, {self.total_pixels} pixels in total")
        if plot_nimgs == -1:
            self.plot_nimgs = self.n_images
        else:
            self.plot_nimgs = min(plot_nimgs, self.n_images)
        self.shuffle = kwargs.get('shuffle', True)
        if self.shuffle:
            self.shuffle_plot_index()

    def shuffle_plot_index(self):
        if self.shuffle:
            self.plot_index = torch.randperm(self.n_images)[:self.plot_nimgs]

    def __len__(self):
        return self.plot_nimgs
    
    def get_uv(self):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        return uv

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.plot_index[idx]
        uv = self.get_uv()

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.use_lmask:
            ground_truth['light_mask'] = self.lightmask_images[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)


class InterpolateDataset(torch.utils.data.Dataset):
    """
    View interpolation: specify 2 view ids from training set and generate a video moving between them
    """
    def __init__(self,
                 data_dir,
                #  img_res,
                 id0,
                 id1,
                 num_frames=60,
                 scan_id=0,
                 **kwargs
                 ):

        self.instance_dir = os.path.join('data', data_dir, 'scan{0}'.format(scan_id))
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        image_dir = '{0}/image'.format(self.instance_dir)
        im = cv2.imread(f"{image_dir}/{id0:04d}.png")
        h, w, _ = im.shape
        self.img_res = [h, w]
        self.total_pixels = h * w

        self.cam_file = '{0}/cameras_normalize.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        P0 = camera_dict['world_mat_%d' % id0].astype(np.float32) @ camera_dict['scale_mat_%d' % id0].astype(np.float32)
        P1 = camera_dict['world_mat_%d' % id1].astype(np.float32) @ camera_dict['scale_mat_%d' % id1].astype(np.float32)
        P0 = P0[:3,:]
        P1 = P1[:3,:]
        K, pose0 = rend_util.load_K_Rt_from_P(None, P0)
        _, pose1 = rend_util.load_K_Rt_from_P(None, P1)
        rots = Rot.from_matrix(np.stack([pose0[:3,:3].T, pose1[:3,:3].T]))
        slerp = Slerp([0, 1], rots)

        if (scale := kwargs.get('downsample', 1)) > 1:
            self.img_res[0] = self.img_res[0] // scale
            self.img_res[1] = self.img_res[1] // scale
            self.total_pixels = self.img_res[0] * self.img_res[1]
            K[0,0] /= scale
            K[1,1] /= scale
            K[0,2] /= scale
            K[1,2] /= scale

        self.intrinsics = torch.from_numpy(K).float()
        self.pose_all = []
        for i in range(num_frames):
            ratio = np.sin(((i / num_frames) - 0.5) * np.pi) * 0.5 + 0.5
            t = (1 - ratio) * pose0[:3,3] + ratio * pose1[:3,3]
            R = slerp(ratio).as_matrix()
            pose = np.eye(4, dtype=np.float32)
            pose[:3,3] = t
            pose[:3,:3] = R.T
            self.pose_all.append(torch.from_numpy(pose).float())
        self.pose_all = torch.stack(self.pose_all)
        self.n_frames = num_frames

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics,
            "pose": self.pose_all[idx]
        }
        return idx, sample

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)


class RelightDataset(PlotDataset):
    def __init__(self, data_dir, edit_cfg, scan_id=0, is_val=False, **kwargs):
        super().__init__(data_dir, 1, scan_id, is_val, None, False, [edit_cfg['index']], True, **kwargs)
        self.edit_mask = 'mask' in edit_cfg
        if self.edit_mask:
            self.mask = rend_util.load_mask(edit_cfg['mask']).astype(np.float32)
            mh, mw = self.mask.shape
            if mh != self.img_res[0] or mw != self.img_res[1]:
                self.mask = cv2.resize(self.mask, (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_AREA)
                self.mask = (self.mask > 0.5)
            self.mask = torch.from_numpy(self.mask).float().flatten()
            if 'normal' in edit_cfg:
                self.loadattr(edit_cfg, 'normal', 0)
                self.normal = self.normal.reshape(-1, 3)
                self.normal = F.normalize(self.normal, dim=-1, eps=1e-6)
            if 'rough' in edit_cfg:
                self.loadattr(edit_cfg, 'rough', 1)
                self.rough = self.rough.reshape(-1, 1)
            if 'kd' in edit_cfg:
                self.loadattr(edit_cfg, 'kd', 2)
                self.kd = self.kd.reshape(-1, 3)
            if 'ks' in edit_cfg:
                self.loadattr(edit_cfg, 'ks', 2)
                self.ks = self.ks.reshape(-1, 3)
        self.uv = self.get_uv()
    
    def loadattr(self, edit_cfg, attr, mode=0):
        if mode == 0:
            im = rend_util.load_normal(edit_cfg[attr])
        elif mode == 1:
            im = cv2.imread(edit_cfg[attr], -1)
            if len(im.shape) == 3:
                im = im[:,:,-1]
        else:
            im = rend_util.load_rgb(edit_cfg[attr]).transpose(1, 2, 0)
        h, w = im.shape[:2]
        if h != self.img_res[0] or w != self.img_res[1]:
            im = cv2.resize(im, (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_AREA)
        setattr(self, attr, torch.from_numpy(im).float())

    def __len__(self):
        return self.total_pixels
    
    def __getitem__(self, idx):
        sample = {
            "uv": self.uv[idx].unsqueeze(0),
            "intrinsics": self.intrinsics_all[0],
            "pose": self.pose_all[0] 
        }
        ground_truth = {
            "rgb": self.rgb_images[0][idx],
            'light_mask': self.lightmask_images[0][idx]
            # 'edit_mask': self.edit_mask[idx]
        }
        if self.edit_mask:
            ground_truth['mask'] = self.mask[idx]
        if hasattr(self, 'normal'):
            ground_truth['normal'] = self.normal[idx]
        if hasattr(self, 'rough'):
            ground_truth['rough'] = self.rough[idx]
        if hasattr(self, 'kd'):
            ground_truth['kd'] = self.kd[idx]
        if hasattr(self, 'ks'):
            ground_truth['ks'] = self.ks[idx]
        return idx, sample, ground_truth

    
class RelightVideoDataset(PlotDataset):
    def __init__(self, data_dir, edit_cfg, scan_id=0, is_val=False, **kwargs):
        self.n_frames = edit_cfg['n_frames']
        self.img_idx = edit_cfg['index']
        super().__init__(data_dir, 1, scan_id, is_val, None, False, [edit_cfg['index']] * self.n_frames, True, **kwargs)
        self.edit_mask = 'mask' in edit_cfg
        if self.edit_mask:
            self.mask = rend_util.load_mask(edit_cfg['mask']).astype(np.float32)
            mh, mw = self.mask.shape
            if mh != self.img_res[0] or mw != self.img_res[1]:
                self.mask = cv2.resize(self.mask, (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_AREA)
                self.mask = (self.mask > 0.5)
            self.mask = torch.from_numpy(self.mask).float().flatten()
        self.uv = self.get_uv()
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx):
        sample = {
            "uv": self.uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        ground_truth = {
            "rgb": self.rgb_images[idx],
            'light_mask': self.lightmask_images[idx]
            # 'edit_mask': self.edit_mask[idx]
        }
        if self.edit_mask:
            ground_truth['mask'] = self.mask
        return idx, sample, ground_truth
