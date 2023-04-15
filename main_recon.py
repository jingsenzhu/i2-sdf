import torch
import yaml
import pytorch_lightning as pl
import argparse
import os
import utils
import model
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from rich.progress import TextColumn
import GPUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True, help="Path to (.yml) config file.")
    parser.add_argument('-d', "--device_ids", type=int, nargs='+', default=None, help="GPU devices to use")
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_mode', choices=['render', 'mesh', 'interpolate'], default='render')
    parser.add_argument('-v', '--version', type=int, nargs='?')
    parser.add_argument('--inter_id', type=int, nargs=2, required=False, help='2 view ids for interpolation video.')
    parser.add_argument('-i', '--indices', nargs='*', type=int, help='If set, render only specified indices of the dataset instead of all images.')
    parser.add_argument('--n_frames', type=int, default=60, help='Number of frames in the interpolation video.')
    parser.add_argument('--frame_rate', type=int, default=24, help='Frame rate of the interpolation video.')
    parser.add_argument('-f', '--full_res', action='store_true', help='If set, dataset downscaling will be ignored.')
    parser.add_argument('--is_val', action='store_true', help='If set, render the validation set instead of training set.')
    parser.add_argument('--val_mesh', action='store_true', help='If set, extract and save mesh every validation epoch.')
    parser.add_argument('--score', action='store_true', help='If set, evaluate the meshing score (need to provide GT mesh).')
    parser.add_argument('--far_clip', type=float, default=5.0)
    parser.add_argument('--ckpt', type=str, default='last')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution for marching cube algorithm')
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.conf) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = utils.CfgNode(cfg_dict)
    
    expname = args.expname if args.expname else cfg.train.expname
    scan_id = cfg.dataset.scan_id if args.scan_id == -1 else args.scan_id
    cfg.dataset.scan_id = scan_id
    expname = expname + '_' + str(scan_id)

    if args.version is None and (v := args.conf.find("version_")) != -1:
        args.version = int(args.conf[v + 8:args.conf.find("/config")])
        print(f"[INFO] Loaded version {args.version} from config file")
    
    if args.version is not None:
        logger = loggers.TensorBoardLogger(save_dir=args.exps_folder, name=expname, version=args.version)
    else:
        logger = loggers.TensorBoardLogger(save_dir=args.exps_folder, name=expname)
    
    if args.device_ids is None:
        args.device_ids = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                              excludeID=[], excludeUUID=[])
        print("Selected GPU {} automatically".format(args.device_ids[0]))
    torch.cuda.set_device(args.device_ids[0])
    torch.set_float32_matmul_precision('medium')
    progbar_callback = utils.RichProgressBarWithScanId(scan_id, leave=False)
    pl.seed_everything(args.seed)
    
    if args.test:
        version = args.version if args.version is not None else logger.version - 1
        exp_dir = os.path.join(logger.root_dir, f"version_{version}")
        del logger
        if args.test_mode == 'render':
            system = model.VolumeRenderSystem(cfg, exp_dir, indices=args.indices, is_val=args.is_val, full_res=args.full_res)
            if not args.ckpt.endswith('.ckpt'):
                args.ckpt += '.ckpt'
            ckpt = torch.load(os.path.join(exp_dir, 'checkpoints', args.ckpt), map_location='cuda')
            system.load_state_dict(ckpt['state_dict'])
            model.lpips.cuda()
        elif args.test_mode == 'mesh':
            system = model.SDFMeshSystem(cfg, exp_dir, args.resolution, args.score)
            if not args.ckpt.endswith('.ckpt'):
                args.ckpt += '.ckpt'
            ckpt = torch.load(os.path.join(exp_dir, 'checkpoints', args.ckpt), map_location='cuda')
            system.load_state_dict(ckpt['state_dict'])
            system.cuda()
            system.eval()
            system.initialize()
        # elif args.test_mode == 'interpolate':
        else:
            system = model.ViewInterpolateSystem(cfg, exp_dir, *args.inter_id, n_frames=args.n_frames, frame_rate=args.frame_rate)
            if not args.ckpt.endswith('.ckpt'):
                args.ckpt += '.ckpt'
            ckpt = torch.load(os.path.join(exp_dir, 'checkpoints', args.ckpt), map_location='cuda')
            system.load_state_dict(ckpt['state_dict'])
        trainer = pl.Trainer(
            logger=False,
            accelerator='gpu',
            devices=args.device_ids,
            callbacks=[progbar_callback]
        )
        trainer.test(system)
    else:
        max_steps = cfg.train.get('steps', 200000)
        print(f"Training for {max_steps} steps")
        exp_dir = logger.log_dir
        checkpoint_callback = ModelCheckpoint(os.path.join(exp_dir, 'checkpoints'), save_last=True, every_n_train_steps=cfg.train.checkpoint_freq)
        if hasattr(cfg.train, 'plot_freq'):
            kwargs = {'val_check_interval': cfg.train.plot_freq}
        else:
            kwargs = {'check_val_every_n_epoch': cfg.train.plot_epochs}
        trainer = pl.Trainer(
            logger=logger,
            accelerator='gpu',
            devices=args.device_ids,
            strategy=None,
            callbacks=[checkpoint_callback, progbar_callback],
            max_steps=max_steps,
            **kwargs
        )
        system = model.ReconstructionTrainer(
            cfg, progbar_callback,
            exp_dir=exp_dir,
            is_val=args.is_val,
            val_mesh=args.val_mesh
        )
        trainer.fit(system)
    torch.cuda.empty_cache()