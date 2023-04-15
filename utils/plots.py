import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import numpy as np
import torch
from skimage import measure
import torchvision.utils as vutils
import trimesh
from PIL import Image
import cv2
import mcubes
from utils import rend_util


def plot(implicit_network, indices, plot_data, path, epoch, img_res, plot_nimgs, resolution=None, grid_boundary=None, meshing=True, level=0):

    if plot_data is not None:
        if 'rgb_eval' in plot_data:
            plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res)
        if 'hdr_eval' in plot_data:
            plot_images(plot_data['hdr_eval'], plot_data['hdr_gt'], path, epoch, plot_nimgs, img_res, 'hdr', True)
        if 'rgb_surface' in plot_data:
            plot_images(plot_data['rgb_surface'], plot_data['rgb_gt'], path, '{0}s'.format(epoch), plot_nimgs, img_res)
        if 'rendered' in plot_data:
            plot_images(plot_data['rendered'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, 'rendered')

        if 'normal_map' in plot_data:
            plot_imgs_wo_gt(plot_data['normal_map'], path, epoch, plot_nimgs, img_res)
        if 'depth_eval' in plot_data:
            plot_depths(plot_data['depth_eval'], path, epoch, plot_nimgs, img_res)
        if 'lmask_eval' in plot_data:
            plot_images(plot_data['lmask_eval'], plot_data['lmask_gt'], path, epoch, plot_nimgs, img_res, 'light_mask')

        if 'Kd' in plot_data:
            # plot_imgs_wo_gt(plot_data['albedo'], path, epoch, plot_nimgs, img_res, 'albedo')
            plot_images(plot_data['Ks'], plot_data['Kd'], path, epoch, plot_nimgs, img_res, 'albedo')
        if 'roughness' in plot_data:
            plot_colormap(plot_data['roughness'], path, epoch, plot_nimgs, img_res)
        if 'metallic' in plot_data:
            plot_colormap(plot_data['metallic'], path, epoch, plot_nimgs, img_res, 'metallic')
        if 'emission' in plot_data:
            plot_imgs_wo_gt(plot_data['emission'], path, '{0}'.format(epoch), plot_nimgs, img_res, 'emission', True)

    if meshing:
        path = f"{path}/mesh"
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])
        data = []

        # plot surface
        surface_traces = get_surface_trace(path=path,
                                        epoch=epoch,
                                        sdf=lambda x: implicit_network(x)[:, 0],
                                        resolution=resolution,
                                        grid_boundary=grid_boundary,
                                        level=level
                                        )

        if surface_traces is not None:
            data.append(surface_traces[0])

        # plot cameras locations
        if plot_data is not None:
            for i, loc, dir in zip(indices, cam_loc, cam_dir):
                data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

        fig = go.Figure(data=data)
        scene_dict = dict(xaxis=dict(range=[-6, 6], autorange=False),
                        yaxis=dict(range=[-6, 6], autorange=False),
                        zaxis=dict(range=[-6, 6], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1))
        fig.update_layout(scene=scene_dict, width=1200, height=1200, showlegend=True)
        filename = '{0}/surface_{1}.html'.format(path, epoch)
        offline.plot(fig, filename=filename, auto_open=False)


def visualize_pointcloud(points, filename):
    fig = go.Figure(
        data = [
            get_3D_scatter_trace(points, 'Pointcloud', 1)
        ]
    )
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1200, height=1200, showlegend=True)
    offline.plot(fig, filename=filename, auto_open=False)


def visualize_clustered_pointcloud(points, labels, centroids, filename):
    fig = go.Figure()
    if centroids is not None:
        fig.add_trace(get_3D_scatter_trace(centroids, "Centroids", 10))
    for c in torch.unique(labels):
        cluster = points[labels == c, :]
        fig.add_trace(get_3D_scatter_trace(cluster, f"Emitter #{int(c)}"))
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1200, height=1200, showlegend=True)
    offline.plot(fig, filename=filename, auto_open=False)


def visualize_marked_pointcloud(points, counts, path, epoch):
    fig = go.Figure(
        data = [
            get_3D_marked_scatter_trace(points, counts, 'Pointcloud samples')
        ]
    )
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1200, height=1200, showlegend=True)
    filename = '{0}/pointcloud/{1}.html'.format(path, epoch)
    offline.plot(fig, filename=filename, auto_open=False)


def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_marked_scatter_trace(points, marks, name='', size=1, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            color=marks.squeeze().cpu(),
            colorscale='Viridis',
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    mesh = trimesh.Trimesh(vertices, triangles)
    return mesh


def get_surface_high_res_mesh(sdf, resolution=100, grid_boundary=[-2.0, 2.0], level=0, take_components=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100, grid_boundary)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=level,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    if take_components:
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_surface_by_grid(grid_params, sdf, resolution=100, level=0, higher_res=False):
    grid_params = grid_params * [[1.5], [1.0]]

    # params = PLOT_DICT[scan_id]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()

    if higher_res:
        # get low res mesh to sample point cloud
        grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)
        z = []
        points = grid['grid_points']

        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()

        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()

        grid_aligned = get_grid(helper.cpu(), resolution, eps=0.01)
    else:
        grid_aligned = get_grid(None, resolution, input_min=input_min, input_max=input_max, eps=0.0)

    grid_points = grid_aligned['grid_points']

    if higher_res:
        g = []
        for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        if higher_res:
            verts = torch.from_numpy(verts).cuda().float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()
        else:
            verts = verts + np.array([grid_aligned['xyz'][0][0], grid_aligned['xyz'][1][0], grid_aligned['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, normals)

        # CUTTING MESH ACCORDING TO THE BOUNDING BOX
        if higher_res:
            bb = grid_params
            transformation = np.eye(4)
            transformation[:3, 3] = (bb[1,:] + bb[0,:])/2.
            bounding_box = trimesh.creation.box(extents=bb[1,:] - bb[0,:], transform=transformation)

            meshexport = meshexport.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)

    return meshexport

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
    print(x.shape, y.shape, z.shape)
    xx, yy, zz = np.meshgrid(x, y, z)
    # print(xx.shape, yy.shape, zz.shape)
    # xx = torch.from_numpy(xx.flatten()).cuda().float()
    # yy = torch.from_numpy(yy.flatten()).cuda().float()
    # zz = torch.from_numpy(zz.flatten()).cuda().float()
    # grid_points = torch.cat([xx, yy, zz], dim=1).T
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_imgs_wo_gt(normal_maps, path, epoch, plot_nrow, img_res, path_name='normal', is_hdr=False):
    normal_maps_plot = lin2img(normal_maps, img_res)
    tensor = vutils.make_grid(normal_maps_plot,
                                scale_each=False,
                                normalize=False,
                                nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    if not is_hdr:
        scale_factor = 255
        tensor = (tensor * scale_factor).astype(np.uint8)
        img = Image.fromarray(tensor)
        img.save('{0}/{1}/{2}.png'.format(path, path_name, epoch))
    else:
        cv2.imwrite('{0}/{1}/{2}.exr'.format(path, path_name, epoch), tensor[:,:,::-1])


def plot_imgs_filter(rgb_points, ground_true, path, epoch, img_res, path_name='rendering'):
    output = lin2img(rgb_points, img_res).squeeze(0) # (1, 3, h, w)
    ground_true = lin2img(ground_true, img_res).squeeze(0)
    output = output.permute(1, 2, 0).cpu().numpy()
    scale_factor = 255
    output = (output * scale_factor).astype(np.uint8)
    ground_true = ground_true.permute(1, 2, 0).cpu().numpy()
    ground_true = (ground_true * scale_factor).astype(np.uint8)
    output = output[:,:,::-1]
    ground_true = ground_true[:,:,::-1]
    filtered = cv2.ximgproc.guidedFilter(ground_true, output, 10, 2, -1)
    cv2.imwrite('{0}/{1}/{2}.png'.format(path, path_name, epoch), filtered)


def plot_colormap(mat_info, path, epoch, plot_nrow, img_res, colormap=cv2.COLORMAP_VIRIDIS, path_name='roughness'):
    mat_info_plot = lin2img(mat_info, img_res)

    tensor = vutils.make_grid(mat_info_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    if colormap is None:
        cv2.imwrite('{0}/{1}/{2}.exr'.format(path, path_name, epoch), tensor)
    else:
        tensor = (tensor * 255).astype(np.uint8)
        img = cv2.applyColorMap(tensor, colormap)
        cv2.imwrite('{0}/{1}/{2}.png'.format(path, path_name, epoch), img)


def plot_depths(depth_maps, path, epoch, plot_nrow, img_res, colormap=cv2.COLORMAP_VIRIDIS):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = vutils.make_grid(depth_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    # scale_factor = 255
    # tensor = (tensor * scale_factor).astype(np.uint8)
    if colormap is None:
        cv2.imwrite('{0}/depth/{1}.exr'.format(path, epoch), tensor)
    else:
        tensor = tensor / (tensor.max() + 1e-6)
        tensor = (tensor * 255).astype(np.uint8)
        img = cv2.applyColorMap(tensor, colormap)
        cv2.imwrite('{0}/depth/{1}.png'.format(path, epoch), img)

    # img = Image.fromarray(tensor)
    # img.save('{0}/normal_{1}.png'.format(path, epoch))


def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, path_name='rendering', is_hdr=False):
    ground_true = ground_true.cuda()

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = vutils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    if not is_hdr:
        scale_factor = 255
        tensor = (tensor * scale_factor).astype(np.uint8)
        img = Image.fromarray(tensor)
        img.save('{0}/{1}/{2}.png'.format(path, path_name, epoch))
    else:
        cv2.imwrite('{0}/{1}/{2}.exr'.format(path, path_name, epoch), tensor[:,:,::-1])


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
