import math
import torch
import pytorch_lightning as pl
import numpy as np
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import utils
from utils import rend_util
import utils.plots as plt
import dataset
import model
from tqdm import trange
from pytorch_lightning.callbacks import RichProgressBar
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


lpips = LPIPS()

class ReconstructionTrainer(pl.LightningModule):
    def __init__(self, conf, prog_bar: RichProgressBar, exp_dir, model_only=False, val_mesh=False, is_val=False, **kwargs) -> None:
        super().__init__()
        self.conf = conf
        self.prog_bar = prog_bar
        self.batch_size = conf.train.batch_size
        self.bubble_batch_size = getattr(conf.train, 'bubble_batch_size', self.batch_size)
        self.expdir = exp_dir
        self.val_mesh = val_mesh

        conf_model = conf.model
        use_normal = (getattr(conf.loss, 'normal_weight', 0) > 0) or (getattr(conf.loss, 'angular_weight', 0) > 0)
        conf_model.use_normal = use_normal
        self.model = model.I2SDFNetwork(conf_model)
        
        if model_only:
            return

        print('[INFO] Loading data ...')
        dataset_conf = conf.dataset
        self.scan_id = dataset_conf.scan_id
        self.train_dataset = dataset.ReconDataset(
                **dataset_conf, 
                use_mask=getattr(conf.loss, 'mask_weight', 0) > 0, 
                use_depth=getattr(conf.loss, 'depth_weight', 0) > 0, 
                use_normal=use_normal, 
                use_bubble=getattr(conf.loss, 'bubble_weight', 0) > 0, 
                use_lightmask=getattr(conf.loss, 'light_mask_weight', 0) > 0
            )
        if self.train_dataset.use_bubble:
            os.makedirs(os.path.join(self.expdir, 'hotmap'), exist_ok=True)
            os.makedirs(os.path.join(self.expdir, 'countmap'), exist_ok=True)
            self.pdf_criterion = getattr(conf.train, 'pdf_criterion', 'DEPTH')
            assert self.pdf_criterion in ['RGB', 'DEPTH']

        self.is_hdr = self.train_dataset.is_hdr
        self.plots_dir = os.path.join(self.expdir, 'plots')
        self.trace_bub_idx = self.conf.train.get('trace_bub_idx', -1)
        if self.trace_bub_idx != -1:
            os.makedirs(f"{self.plots_dir}/bubble", exist_ok=True)
            print(f"[INFO] Activate hotmap visualization for #{self.trace_bub_idx}")
            self.plot_dataset = dataset.PlotDataset(**dataset_conf, indices=[self.trace_bub_idx], plot_nimgs=1, is_val=is_val)
        else:
            data = {
                'intrinsics': self.train_dataset.intrinsics_all,
                'pose': self.train_dataset.pose_all,
                'rgb': self.train_dataset.rgb_images,
                'img_res': self.train_dataset.img_res
            }
            if self.train_dataset.use_lightmask:
                data['light_mask'] = self.train_dataset.lightmask_images
            self.plot_dataset = dataset.PlotDataset(**dataset_conf, data=data, plot_nimgs=conf.plot.plot_nimgs, is_val=is_val)

        os.makedirs(self.plots_dir, exist_ok=True)
        with open(f"{self.expdir}/config.yml", 'w') as f:
            f.write(self.conf.dump())
        if self.train_dataset.use_bubble:
            points = self.train_dataset.pointcloud
            index = torch.randperm(points.size(0))[:200000]
            points = points[index,:]
            plt.visualize_pointcloud(points, f"{self.expdir}/pointcloud.html")
            print(f"[INFO] Pointcloud visualization success: saved to {self.expdir}/pointcloud.html")
            self.pdf_prune = self.train_dataset.pdf_prune
            self.pdf_max = self.train_dataset.pdf_max

        # self.ds_len = len(self.train_dataset)
        self.ds_len = self.train_dataset.n_images
        print('[INFO] Finish loading data. Data-set size: {0}'.format(self.ds_len))
        epoch_steps = len(self.train_dataset) / self.batch_size
        self.nepochs = int(math.ceil(200000 / epoch_steps))

        self.loss = model.I2SDFLoss(**conf.loss)
        self.total_pixels = self.plot_dataset.total_pixels
        self.img_res = self.plot_dataset.img_res
        self.bubble_activated = False
        self.uniform_bubble = getattr(self.conf.train, 'uniform_bubble', False)
        if self.uniform_bubble:
            print("[INFO] Ablation study: uniform sampling for bubble loss")
        self.checkpoint_freq = self.conf.train.checkpoint_freq
        self.split_n_pixels = self.conf.train.split_n_pixels
        self.plot_conf = self.conf.plot
        self.progbar_task = None
        if self.train_dataset.use_lightmask and getattr(self.conf.train, 'flip_light', False):
            self.train_dataset.lightmask_images = 1.0 - self.train_dataset.lightmask_images
            self.plot_dataset.lightmask_images = 1.0 - self.plot_dataset.lightmask_images

    def forward(self):
        raise NotImplementedError("forward not supported by trainer")

    def plot_hotmap(self, path):
        assert self.bubble_activated
        ds = self.train_dataset
        hotmaps = torch.zeros(self.ds_len * ds.total_pixels)
        hotmaps[ds.pixlinks] = self.pdf.cpu()
        hotmaps = hotmaps.reshape(self.ds_len, *ds.img_res)
        for i, hotmap in enumerate(hotmaps):
            hotmap = hotmap.numpy()
            # hotmap /= max(1e-4, hotmap.max())
            hotmap = (hotmap * 255).astype(np.uint8)
            hotmap = cv2.applyColorMap(hotmap, cv2.COLORMAP_MAGMA)
            cv2.imwrite(os.path.join(path, "{:04d}.png".format(i)), hotmap)
            if self.trace_bub_idx == i:
                cv2.imwrite(os.path.join(f"{self.plots_dir}/bubble", f"{self.global_step}_hot.png"), hotmap)
    
    def plot_countmap(self, path):
        assert self.bubble_activated
        ds = self.train_dataset
        countmaps = torch.zeros(self.ds_len * ds.total_pixels)
        countmaps[ds.pixlinks] = self.sample_count.cpu().float()
        countmaps = countmaps.reshape(self.ds_len, *ds.img_res)
        countmaps = countmaps / max(1, countmaps.max())
        for i, countmap in enumerate(countmaps):
            countmap = countmap.numpy()
            countmap = (countmap * 255).astype(np.uint8)
            countmap = cv2.applyColorMap(countmap, cv2.COLORMAP_MAGMA)
            cv2.imwrite(os.path.join(path, "{:04d}.png".format(i)), countmap)
            if self.trace_bub_idx == i:
                cv2.imwrite(os.path.join(f"{self.plots_dir}/bubble", f"{self.global_step}_cnt.png"), countmap)

    def update_pdf(self, value, idx):
        assert self.bubble_activated
        ds = self.train_dataset
        value = value.to(self.pdf.device)
        if self.pdf_max is not None:
            value = value.clamp(max=self.pdf_max)
        value[value < self.pdf_prune] = 0 # PDF pruning
        link = ds.pointlinks[idx]
        mask = (link != -1)
        link = link[mask]
        value = value[mask]
        self.pdf[link] = value
    
    def sample_bubble(self, batch_size):
        assert self.bubble_activated
        ds = self.train_dataset
        if self.uniform_bubble:
            sample_idx = torch.randperm(ds.pointcloud.size(0), device=ds.pointcloud.device)[:batch_size]
            return ds.pointcloud[sample_idx,:]
        sample_idx = torch.where(self.pdf > 0)[0]
        pdf_samples = self.pdf[sample_idx]
        pointcloud_samples = ds.pointcloud[sample_idx,:]
        if sample_idx.size(0) >= (1 << 24):
            # print(sample_idx.size(0), self.pdf.size(0), (1 << 24))
            print("[ERROR] PDF capacity exceeds maximum limit of PyTorch")
            exit(1)
        idx = torch.multinomial(pdf_samples, batch_size, replacement=False) # importance sampling
        self.sample_count[sample_idx[idx]] += 1
        return pointcloud_samples[idx,:]

    def initialize_bubble_pdf(self, split_size):
        ds = self.train_dataset
        # ds.pdf = ds.pdf.cuda()
        self.register_buffer('pdf', torch.zeros(len(ds.pointcloud)), False)
        self.register_buffer('sample_count', torch.zeros(len(ds.pointcloud)), False)
        self.pdf = self.pdf.cuda()
        # self.sample_count = self.sample_count.cuda()
        for i in trange(ds.n_images):
            intrinsics = ds.intrinsics_all[i].cuda().unsqueeze(0)
            pose = ds.pose_all[i].cuda().unsqueeze(0)
            img = ds.rgb_images[i].cuda() if self.pdf_criterion != 'DEPTH' else ds.depth_images[i].cuda()
            uv = ds.uv.cuda().unsqueeze(1) # (h*w, 1, 2)
            img_splits = torch.split(img, split_size)
            uv_splits = torch.split(uv, split_size)
            indices = torch.arange(i * ds.total_pixels, (i + 1) * ds.total_pixels, dtype=torch.long, device='cuda')
            index_splits = torch.split(indices, split_size)
            for img_split, uv_split, index_split in zip(img_splits, uv_splits, index_splits):
                data = {
                    'uv': uv_split,
                    'intrinsics': intrinsics.repeat(len(uv_split), 1, 1),
                    'pose': pose.repeat(len(uv_split), 1, 1)
                }
                model_output = self.model.forward(data, True)
                if self.pdf_criterion == 'RGB':
                    self.update_pdf((model_output['rgb_values'].detach().clamp(0, 1) - img_split.clamp(0, 1)).abs().mean(dim=-1), index_split)
                # elif self.pdf_criterion == 'DEPTH':
                else:
                    self.update_pdf((model_output['depth_values'].detach() - img_split).abs(), index_split)

    def configure_optimizers(self):
        lr = self.conf.train.learning_rate
        optimizer = optim.Adam(self.model.get_param_groups(lr), eps=1e-15)
        decay_rate = getattr(self.conf.train, 'sched_decay_rate', 0.1)
        decay_steps = self.nepochs * self.ds_len
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate ** (1./decay_steps))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn, num_workers=4)
        
    def val_dataloader(self):
        return DataLoader(self.plot_dataset, batch_size=self.conf.plot.plot_nimgs, shuffle=False, collate_fn=self.train_dataset.collate_fn)

    def log_if_nonzero(self, name, value, *args, **kwargs):
        if value > 0:
            self.log(name, value, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        indices, img_indices, model_input, ground_truth = batch
        if not self.bubble_activated and self.train_dataset.use_bubble and self.global_step >= self.loss.min_bubble_iter and self.global_step < self.loss.max_bubble_iter:
            # Start bubble step
            with torch.no_grad():
                self.bubble_activated = True
                self.train_dataset.pointcloud = self.train_dataset.pointcloud.cuda()
                # self.loss.eikonal_weight = self.loss.eikonal_weight_pointcloud

                # Disable normal loss, since it will discourage the growth of bubbles
                self.loss.normal_weight_bak = self.loss.normal_weight
                self.loss.normal_weight = 0.0
                self.loss.angular_weight_bak = self.loss.angular_weight
                self.loss.angular_weight = 0.0
                if not self.uniform_bubble:
                    print(f"[INFO] Start to initializing pointcloud PDF, criterion: {self.pdf_criterion}")
                    self.initialize_bubble_pdf(self.split_n_pixels) # initialize PDF maps for each image by computing losses
                    torch.save(self.pdf, os.path.join(self.expdir, 'checkpoints', "pdf.pt"))
                    torch.cuda.empty_cache()
                    self.plot_hotmap(os.path.join(self.expdir, 'hotmap'))
                    print("[INFO] Finish to initializing pointcloud PDF")
                    print(f"[INFO] {torch.count_nonzero(self.pdf).item()}/{self.pdf.size(0)} points to be sampled")

        if self.bubble_activated:
            model_input['pointcloud'] = self.sample_bubble(self.bubble_batch_size)

        model_outputs = self.model(model_input)
        if self.bubble_activated and not self.uniform_bubble:
            with torch.no_grad():
                if self.pdf_criterion == 'RGB':
                    self.update_pdf((model_outputs['rgb_values'].detach().clamp(0, 1) - ground_truth['rgb'].clamp(0, 1)).abs().mean(dim=-1), indices)
                # elif self.pdf_criterion == 'DEPTH':
                else:
                    self.update_pdf((model_outputs['depth_values'].detach() - ground_truth['depth']).abs(), indices)

        loss_output = self.loss(model_outputs, ground_truth, self.global_step)
        if self.bubble_activated and self.loss.max_bubble_iter is not None and self.global_step >= self.loss.max_bubble_iter:
            # End bubble step
            self.train_dataset.use_bubble = False
            self.bubble_activated = False
            del self.train_dataset.pointcloud
            del self.train_dataset.pointlinks
            del self.train_dataset.pixlinks
            if not self.uniform_bubble:
                delattr(self, 'pdf')
                delattr(self, 'sample_count')
            torch.cuda.empty_cache()
            # self.loss.eikonal_weight = self.conf.loss.eikonal_weight
            # Restore normal loss
            self.loss.normal_weight = self.loss.normal_weight_bak
            self.loss.angular_weight = self.loss.angular_weight_bak
        loss = loss_output['loss']

        with torch.no_grad():
            psnr = rend_util.get_psnr(model_outputs['rgb_values'].detach(), ground_truth['rgb'].view(-1, 3))
            self.log('train/loss', loss.item())
            self.log('train/psnr', psnr.item(), True)
            self.log('train/rgb_loss', loss_output['rgb_loss'].item())
            self.log_if_nonzero('train/eikonal_loss', loss_output['eikonal_loss'].item())
            self.log_if_nonzero('train/smooth_loss', loss_output['smooth_loss'].item())
            self.log_if_nonzero('train/mask_loss', loss_output['mask_loss'].item())
            self.log_if_nonzero('train/depth_loss', loss_output['depth_loss'].item())
            self.log_if_nonzero('train/normal_loss', loss_output['normal_loss'].item())
            self.log_if_nonzero('train/angular_loss', loss_output['angular_loss'].item())
            self.log_if_nonzero('train/bubble_loss', loss_output['bubble_loss'].item())
            self.log_if_nonzero('train/light_mask_loss', loss_output['light_mask_loss'].item())
            self.log('train/beta', self.model.density.beta.item())

        return loss


    def validation_step(self, batch, batch_idx):

        indices, model_input, ground_truth = batch

        split = utils.split_input(model_input, self.total_pixels, self.split_n_pixels)
        res = []
        if self.progbar_task is None and self.prog_bar.progress:
            self.progbar_task = self.prog_bar.progress.add_task("[cyan]Validation split", total=len(split))
        elif self.progbar_task:
            self.prog_bar.progress.reset(self.progbar_task, total=len(split), visible=True)

        for s in split:
            out = utils.detach_dict(self.model(s))
            d = {
                'rgb_values': out['rgb_values'].detach(),
                'depth_values': out['depth_values'].detach()
            }
            if 'normal_map' in out:
                d['normal_map'] = out['normal_map'].detach()
            if 'light_mask' in out:
                d['light_mask'] = out['light_mask'].detach()
            del out
            res.append(d)
            if self.progbar_task:
                self.prog_bar.progress.update(self.progbar_task, advance=1, refresh=True)
        if self.progbar_task:
            self.prog_bar.progress.update(self.progbar_task, visible=False)
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        def get_plot_data(model_outputs, pose, ground_truth):
            rgb_gt = ground_truth['rgb']
            batch_size, num_samples, _ = rgb_gt.shape
            rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
            if self.is_hdr:
                eval_hdr = rgb_eval
                gt_hdr = rgb_gt
                rgb_eval = rend_util.linear_to_srgb(rgb_eval.clamp(0, 1))
                rgb_gt = rend_util.linear_to_srgb(rgb_gt.clamp(0, 1))
            depth_eval = model_outputs['depth_values'].reshape(batch_size, num_samples, 1)
            plot_data = {
                'rgb_gt': rgb_gt,
                'pose': pose,
                'rgb_eval': rgb_eval,
                'depth_eval': depth_eval
            }
            if self.is_hdr:
                plot_data['hdr_gt'] = gt_hdr
                plot_data['hdr_eval'] = eval_hdr
            if 'normal_map' in model_outputs:
                normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
                normal_map = normal_map.transpose(1, 2) # (bn, 3, h*w)
                R = pose[:,:3,:3].transpose(1, 2)
                normal_map = torch.bmm(R, normal_map) # world to camera
                normal_map = normal_map.transpose(1, 2)
                normal_map = (normal_map + 1.) / 2.
                plot_data['normal_map'] = normal_map
            if 'light_mask' in model_outputs:
                plot_data['lmask_eval'] = model_outputs['light_mask'].reshape(batch_size, num_samples, 1)
                plot_data['lmask_gt'] = ground_truth['light_mask'].reshape(batch_size, num_samples, 1)
            return plot_data

        plot_data = get_plot_data(model_outputs, model_input['pose'], ground_truth)
        return {
            'indices': indices,
            'plot_data': plot_data
        }

    def validation_epoch_end(self, outputs) -> None:
        self.plot_dataset.shuffle_plot_index()
        indices = torch.cat([x['indices'] for x in outputs], dim=0)
        plot_data = utils.merge_dict([x['plot_data'] for x in outputs])

        rgb_eval = plot_data['rgb_eval']
        rgb_gt = plot_data['rgb_gt']
        psnr = rend_util.get_psnr(rgb_eval, rgb_gt)
        self.log('val/psnr', psnr.item())
        rgb_gt = rgb_gt.transpose(1, 2).view(-1, 3, *self.img_res) # (bn, h*w, 3) => (bn, 3, h, w)
        rgb_eval = rgb_eval.transpose(1, 2).view(-1, 3, *self.img_res)
        self.log('val/ssim', ssim(rgb_eval, rgb_gt).item())
        lpips.to(rgb_eval.device)
        self.log('val/lpips', lpips(rgb_eval.clamp(0, 1) * 2 - 1, rgb_gt.clamp(0, 1) * 2 - 1).item())

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs('{0}/rendering'.format(self.plots_dir), exist_ok=True)
        if self.is_hdr:
            os.makedirs('{0}/hdr'.format(self.plots_dir), exist_ok=True)
        os.makedirs('{0}/depth'.format(self.plots_dir), exist_ok=True)
        if 'normal_map' in plot_data:
            os.makedirs('{0}/normal'.format(self.plots_dir), exist_ok=True)
        if 'lmask_eval' in plot_data:
            os.makedirs('{0}/light_mask'.format(self.plots_dir), exist_ok=True)
        if self.val_mesh:
            os.makedirs('{0}/mesh'.format(self.plots_dir), exist_ok=True)
        if self.bubble_activated and not self.uniform_bubble:
            self.plot_hotmap(os.path.join(self.expdir, 'hotmap'))
            self.plot_countmap(os.path.join(self.expdir, 'countmap'))
        plt.plot(self.model.implicit_network,
                indices,
                plot_data,
                self.plots_dir,
                self.global_step,
                self.img_res,
                meshing=self.val_mesh,
                **self.plot_conf
                )


