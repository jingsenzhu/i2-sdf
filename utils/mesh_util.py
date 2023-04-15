# adapted from https://github.com/zju3dv/manhattan_sdf
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
from tqdm.contrib import tenumerate, tzip


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics


class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        

def refuse(mesh, poses, K, H, W, far_clip=5.0):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for i, pose in tenumerate(poses):
        intrinsic = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=far_clip, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

def depth2mesh(depths, poses, K, H, W):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for depth, pose in tzip(depths, poses):
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    return volume.extract_triangle_mesh()