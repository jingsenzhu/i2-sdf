import torch
import pytorch_lightning as pl
import numpy as np
import os
from glob import glob
from torch.utils.data import DataLoader
import utils
from utils import rend_util
import utils.plots as plt
import dataset
import model
from skimage import measure
import cv2
import trimesh
from rich.progress import track
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

lpips = LPIPS()

class SDFMeshSystem(pl.LightningModule):
    def __init__(self, conf, exp_dir, resolution, score=False, far_clip=5.0) -> None:
        super().__init__()
        self.expdir = exp_dir
        conf_model = conf.model
        conf_model.use_normal = False
        self.model = model.I2SDFNetwork(conf_model)
        self.resolution = resolution
        self.grid_boundary = conf.plot.grid_boundary
        self.initialized = False
        self.instance_dir = os.path.join('data', conf.dataset.data_dir, 'scan{0}'.format(conf.dataset.scan_id))
        camera_dict = np.load(os.path.join(self.instance_dir, 'cameras_normalize.npz'))
        self.scale_mat = camera_dict['scale_mat_0']
        self.scan_id = conf.dataset.scan_id
        self.score = score
        if score:
            self.n_imgs = len(os.listdir(os.path.join(self.instance_dir, 'image')))
            self.poses = []
            self.far_clip = far_clip
            for i in range(self.n_imgs):
                K, pose = rend_util.load_K_Rt_from_P(None, camera_dict[f'world_mat_{i}'][:3,:])
                self.poses.append(pose)
            self.K = K
            self.H, self.W, _ = cv2.imread(os.path.join(self.instance_dir, 'image', '0000.png')).shape

    def initialize(self):
        grid = plt.get_grid_uniform(100, self.grid_boundary)
        z = []
        points = grid['grid_points']
        for pnts in track(torch.split(points, 1000000, dim=0)):
            z.append(self.model.implicit_network(pnts)[:,0].detach().cpu().numpy())
        z = np.concatenate(z, axis=0).astype(np.float32)
        verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                         level=0,
                         spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))
        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        self.vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(self.vecs) < 0:
            self.vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), self.vecs)
        helper = torch.bmm(self.vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                        (recon_pc - s_mean).unsqueeze(-1)).squeeze().cpu()
        grid_aligned = plt.get_grid(helper, self.resolution)
        grid_points = grid_aligned['grid_points']
        g = []
        for pnts in track(torch.split(grid_points, 1000000, dim=0)):
            g.append(torch.bmm(self.vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)
        points = grid_points.cpu()
        self.test_dataset = dataset.GridDataset(points, grid_aligned['xyz'])
        self.grid_points = grid_points
        self.initialized = True

    def test_dataloader(self):
        assert self.initialized
        print(len(self.test_dataset))
        return DataLoader(self.test_dataset, batch_size=2000000, shuffle=False, num_workers=32)
    
    def test_step(self, batch, batch_idx):
        return self.model.implicit_network(batch)[:,0].detach().cpu().numpy()
    
    def test_epoch_end(self, outputs) -> None:
        # z = torch.cat(outputs, dim=0).cpu().numpy()
        z = np.concatenate(outputs, axis=0).astype(np.float32)
        if (not (np.min(z) > 0 or np.max(z) < 0)):
            verts, faces, normals, values = measure.marching_cubes(
                            volume=z.reshape(self.test_dataset.xyz[1].shape[0], self.test_dataset.xyz[0].shape[0], self.test_dataset.xyz[2].shape[0]).transpose([1, 0, 2]),
                            level=0,
                            spacing=(self.test_dataset.xyz[0][2] - self.test_dataset.xyz[0][1],
                                     self.test_dataset.xyz[0][2] - self.test_dataset.xyz[0][1],
                                     self.test_dataset.xyz[0][2] - self.test_dataset.xyz[0][1]))
            verts = torch.from_numpy(verts).float().cuda()
            verts = torch.bmm(self.vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
            verts = (verts + self.grid_points[0]).cpu().numpy()
            mesh = trimesh.Trimesh(verts, faces, normals)
            mesh.apply_transform(self.scale_mat)
            mesh_folder = os.path.join(self.expdir, 'eval/mesh')
            os.makedirs(mesh_folder, exist_ok=True)
            mesh.export(os.path.join(mesh_folder, 'scan{0}.ply'.format(self.scan_id)), 'ply')
            if self.score:
                from utils import mesh_util
                import open3d as o3d
                mesh = mesh_util.refuse(mesh, self.poses, self.K, self.H, self.W)
                out_mesh_path = os.path.join(mesh_folder, 'scan{0}_refined.ply'.format(self.scan_id))
                o3d.io.write_triangle_mesh(out_mesh_path, mesh)
                mesh = trimesh.load(out_mesh_path)
                print("[INFO] Pred mesh refined")
                gt_mesh = trimesh.load(os.path.join(self.instance_dir, 'mesh.ply'))
                gt_mesh = mesh_util.refuse(gt_mesh, self.poses, self.K, self.H, self.W, self.far_clip)
                out_mesh_path = os.path.join(mesh_folder, 'scan{0}_gt.ply'.format(self.scan_id))
                o3d.io.write_triangle_mesh(out_mesh_path, gt_mesh)
                gt_mesh = trimesh.load(out_mesh_path)
                print("[INFO] GT mesh refined")
                metrics = mesh_util.evaluate(mesh, gt_mesh)
                with open(f"{mesh_folder}/metrics.txt", 'w') as f:
                    for k in metrics:
                        f.write(f"{k.upper()}: {metrics[k]}\n")
                print(f"[INFO] Metrics saved to {mesh_folder}/metrics.txt\n")

    def forward(self):
        raise NotImplementedError("forward not supported by trainer")


class VolumeRenderSystem(pl.LightningModule):
    def __init__(self, conf, exp_dir, indices=None, is_val=False, score_mesh=False, full_res=False) -> None:
        super().__init__()
        self.expdir = exp_dir
        conf_model = conf.model
        conf_model.use_normal = False
        self.model = model.I2SDFNetwork(conf_model)
        self.scan_id = conf.dataset.scan_id
        dataset_conf = conf.dataset
        if full_res:
            dataset_conf.downsample = 1
        self.test_dataset = dataset.PlotDataset(**dataset_conf, plot_nimgs=-1, shuffle=False, indices=indices, is_val=is_val)
        self.total_pixels = self.test_dataset.total_pixels
        self.img_res = self.test_dataset.img_res
        self.split_n_pixels = conf.train.split_n_pixels
        self.expdir = os.path.join(self.expdir, 'eval')
        if is_val:
            self.expdir = os.path.join(self.expdir, 'test')
        os.makedirs(os.path.join(self.expdir, 'rendering'), exist_ok=True)
        os.makedirs(os.path.join(self.expdir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.expdir, 'normal'), exist_ok=True)

    def test_dataloader(self):
        print(len(self.test_dataset))
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=self.test_dataset.collate_fn)

    @torch.inference_mode(False)
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        indices, model_input, ground_truth = batch
        # idx = batch_idx
        idx = self.test_dataset.indices[batch_idx]
        split = utils.split_input(model_input, self.total_pixels, self.split_n_pixels)
        res = []
        for s in split:
            out = utils.detach_dict(self.model(s))
            d = {
                'rgb_values': out['rgb_values'].detach(),
                'depth_values': out['depth_values'].detach()
            }
            d['normal_map'] = out['normal_map'].detach()
            # d['surface_point'] = out['surface_point'].detach()
            del out
            res.append(d)
        model_outputs = utils.merge_output(res, self.total_pixels, 1)
        _, num_samples, _ = ground_truth['rgb'].shape
        model_outputs['rgb_values'] = model_outputs['rgb_values'].reshape(1, num_samples, 3)
        model_outputs['depth_values'] = model_outputs['depth_values'].reshape(1, num_samples, 1)
        plt.plot_imgs_wo_gt(model_outputs['normal_map'].reshape(1, num_samples, 3), self.expdir, "{:04d}w".format(idx), 1, self.img_res, is_hdr=True)
        normal_map = model_outputs['normal_map'].reshape(num_samples, 3).T # (3, h*w)
        R = model_input['pose'].squeeze()[:3,:3].T
        normal_map = R @ normal_map
        model_outputs['normal_map'] = normal_map.T.reshape(1, num_samples, 3)
        plt.plot_imgs_wo_gt(model_outputs['normal_map'], self.expdir, "{:04d}".format(idx), 1, self.img_res, is_hdr=True)
        model_outputs['normal_map'] = (model_outputs['normal_map'] + 1.) / 2.
        plt.plot_imgs_wo_gt(model_outputs['normal_map'], self.expdir, "{:04d}".format(idx), 1, self.img_res)
        # plt.plot_imgs_wo_gt(model_outputs['surface_point'].reshape(1, num_samples, 3), self.expdir, "{:04d}p".format(idx), 1, self.img_res, is_hdr=True)

        plt.plot_images(model_outputs['rgb_values'], ground_truth['rgb'], self.expdir, "{:04d}".format(idx), 1, self.img_res)
        plt.plot_imgs_wo_gt(model_outputs['rgb_values'], self.expdir, "{:04d}_pred".format(idx), 1, self.img_res, 'rendering')
        plt.plot_depths(model_outputs['depth_values'], self.expdir, "{:04d}".format(idx), 1, self.img_res)
        plt.plot_depths(model_outputs['depth_values'], self.expdir, "{:04d}".format(idx), 1, self.img_res, None)
        pred_img = model_outputs['rgb_values'].T.reshape(3, *self.img_res).unsqueeze(0)
        gt_img = ground_truth['rgb'].T.reshape(3, *self.img_res).unsqueeze(0)
        return {
            'psnr': utils.get_psnr(model_outputs['rgb_values'], ground_truth['rgb']).item(),
            'ssim': ssim(pred_img, gt_img).item(),
            'lpips': lpips(pred_img.clamp(0, 1) * 2 - 1, gt_img.clamp(0, 1) * 2 - 1).item()
        }
    
    def test_epoch_end(self, outputs):
        with open(os.path.join(self.expdir, 'metrics.txt'), 'w') as f:
            f.write(f"# IMAGE RESOLUTION {self.img_res}\n")
            psnr_sum = ssim_sum = lpips_sum = 0
            psnrs = []
            ssims = []
            lpipss = []
            for i, metrics in enumerate(outputs):
                f.write(f"[{i:04d}] [PSNR]{metrics['psnr']:.2f} [SSIM]{metrics['ssim']:.2f} [LPIPS]{metrics['lpips']:.2f}\n")
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
                psnr_sum += metrics['psnr']
                ssim_sum += metrics['ssim']
                lpips_sum += metrics['lpips']
            f.write(f"[MEAN] [PSNR]{psnr_sum/len(outputs):.2f} [SSIM]{ssim_sum/len(outputs):.2f} [LPIPS]{lpips_sum/len(outputs):.2f}\n")
            np.savez_compressed(os.path.join(self.expdir, 'metrics.npz'), psnr=np.array(psnrs), ssim=np.array(ssims), lpips=np.array(lpipss))

    def forward(self):
        raise NotImplementedError("forward not supported by trainer")


class ViewInterpolateSystem(pl.LightningModule):
    def __init__(self, conf, exp_dir, id0, id1, n_frames=60, frame_rate=24, use_normal=True) -> None:
        super().__init__()
        self.expdir = exp_dir
        conf_model = conf.model
        conf_model.use_normal = False
        self.model = model.I2SDFNetwork(conf_model)
        self.scan_id = conf.dataset.scan_id
        dataset_conf = conf.dataset
        self.test_dataset = dataset.InterpolateDataset(**dataset_conf, id0=id0, id1=id1, num_frames=n_frames)
        self.total_pixels = self.test_dataset.total_pixels
        self.img_res = self.test_dataset.img_res
        self.split_n_pixels = conf.train.split_n_pixels
        self.n_frames = n_frames
        self.frame_rate = frame_rate
        self.video_dir = os.path.join(self.expdir, 'eval/interpolate')
        self.id0 = id0
        self.id1 = id1
        self.use_normal = use_normal
        os.makedirs(self.video_dir, exist_ok=True)
        self.frame_dir = os.path.join(self.video_dir, f"{self.id0:04d}_{self.id1:04d}")
        os.makedirs(self.frame_dir, exist_ok=True)
        if self.use_normal:
            self.normal_fdir = os.path.join(self.video_dir, f"{self.id0:04d}_{self.id1:04d}_normal")
            os.makedirs(self.normal_fdir, exist_ok=True)

    def test_dataloader(self):
        print(len(self.test_dataset))
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=self.test_dataset.collate_fn)
    
    @torch.inference_mode(False)
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        indices, model_input = batch
        idx = batch_idx
        split = utils.split_input(model_input, self.total_pixels, self.split_n_pixels)
        res = []
        res_normal = []
        for s in split:
            out = utils.detach_dict(self.model(s, predict_only=not self.use_normal))
            rgb = out['rgb_values'].detach()
            res.append(rgb)
            if self.use_normal:
                res_normal.append(out['normal_map'])
            del out
        rendered = torch.cat(res, dim=0).reshape(self.img_res[0], self.img_res[1], 3).cpu().numpy()
        rendered = (rendered * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"{self.frame_dir}/{idx:04d}.png", rendered[:,:,::-1])
        if self.use_normal:
            normal_map = torch.cat(res_normal, dim=0).reshape(-1, 3).T
            R = model_input['pose'].squeeze()[:3,:3].T
            normal_map = R @ normal_map
            normal_map = normal_map.T.reshape(self.img_res[0], self.img_res[1], 3).cpu().numpy()
            normal_map = (((normal_map + 1) * 0.5) * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(f"{self.normal_fdir}/{idx:04d}.png", normal_map[:,:,::-1])

    
    def test_epoch_end(self, outputs):
        import ffmpeg
        (
            ffmpeg
            .input(os.path.join(self.frame_dir, '*.png'), pattern_type='glob', framerate=self.frame_rate)
            .output(os.path.join(self.video_dir, f"scan{self.scan_id}_{self.id0:04d}_{self.id1:04d}.mp4"), vcodec='h264')
            .overwrite_output()
            .run()
        )
        if self.use_normal:
            (
                ffmpeg
                .input(os.path.join(self.normal_fdir, '*.png'), pattern_type='glob', framerate=self.frame_rate)
                .output(os.path.join(self.video_dir, f"scan{self.scan_id}_{self.id0:04d}_{self.id1:04d}_normal.mp4"), vcodec='h264')
                .overwrite_output()
                .run()
            )

    def forward(self):
        raise NotImplementedError("forward not supported by trainer")

