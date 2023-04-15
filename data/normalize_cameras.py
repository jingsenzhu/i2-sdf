import cv2
import numpy as np
import argparse
from copy import deepcopy

def get_center_point(num_cams,cameras):
    A = np.zeros((3 * num_cams, 3 + num_cams))
    b = np.zeros((3 * num_cams, 1))
    camera_centers=np.zeros((3,num_cams))
    for i in range(num_cams):
        P0 = cameras['world_mat_%d' % i][:3, :]

        K = cv2.decomposeProjectionMatrix(P0)[0]
        R = cv2.decomposeProjectionMatrix(P0)[1]
        c = cv2.decomposeProjectionMatrix(P0)[2]
        c = c / c[3]
        camera_centers[:,i]=c[:3].flatten()

        # v = np.linalg.inv(K) @ np.array([800, 600, 1])
        # v = v / np.linalg.norm(v)

        v=R[2,:]
        A[3 * i:(3 * i + 3), :3] = np.eye(3)
        A[3 * i:(3 * i + 3), 3 + i] = -v
        b[3 * i:(3 * i + 3)] = c[:3]

    soll= np.linalg.pinv(A) @ b

    return soll,camera_centers

def normalize_cameras(original_cameras_filename,output_cameras_filename,num_of_cameras,radius,convert_coord):
    cameras = np.load(original_cameras_filename)
    if num_of_cameras==-1:
        all_files=cameras.files
        maximal_ind=0
        for field in all_files:
            if 'val' not in field:
                maximal_ind=np.maximum(maximal_ind,int(field.split('_')[-1]))
        num_of_cameras=maximal_ind+1
    soll, camera_centers = get_center_point(num_of_cameras, cameras)

    center = soll[:3].flatten()

    max_radius = np.linalg.norm((center[:, np.newaxis] - camera_centers), axis=0).max() * 1.1

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = center[0]
    normalization[1, 3] = center[1]
    normalization[2, 3] = center[2]

    normalization[0, 0] = max_radius / radius
    normalization[1, 1] = max_radius / radius
    normalization[2, 2] = max_radius / radius

    cameras_new = {}
    cameras_new = deepcopy(dict(cameras))
    for i in range(num_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        # cameras_new['world_mat_%d' % i] = cameras['world_mat_%d' % i].copy()
        # if ('val_mat_%d' % i) in cameras:
        #     cameras_new['val_mat_%d' % i] = cameras['val_mat_%d' % i].copy()
        
        def opengl2opencv(P):
            out = cv2.decomposeProjectionMatrix(P[:3,:])
            K, R, t = out[0:3]
            K = K/K[2,2]
            intrinsics = np.eye(4, dtype=np.float32)
            intrinsics[:3, :3] = K
            t = (t[:3] / t[3]).squeeze()
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3,:3] = R
            w2c[:3,3] = -R @ t
            T = np.diag([1, -1, -1, 1])
            w2c = T @ w2c
            return intrinsics @ w2c
        if convert_coord:
            cameras_new['world_mat_%d' % i] = opengl2opencv(cameras_new['world_mat_%d' % i])
            if ('val_mat_%d' % i) in cameras_new:
                cameras_new['val_mat_%d' % i] = opengl2opencv(cameras_new['val_mat_%d' % i])
            # cameras_new['world_mat_%d' % i] = T @ cameras_new['world_mat_%d' % i]
    np.savez(output_cameras_filename, **cameras_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalizing cameras')
    parser.add_argument('-i', '--input_cameras_file', type=str, default="cameras.npz",
                        help='the input cameras file')
    parser.add_argument('-o', '--output_cameras_file', type=str, default="cameras_normalize.npz",
                        help='the output cameras file')
    parser.add_argument('--id', type=int, nargs='?')
    parser.add_argument('-n', '--name', type=str, default='synthetic')
    parser.add_argument('--number_of_cams',type=int, default=-1,
                        help='Number of cameras, if -1 use all')
    parser.add_argument('-r', '--radius', type=float, default=2.0)
    parser.add_argument('-c', '--convert_coord', action='store_true')

    args = parser.parse_args()
    if args.id:
        args.input_cameras_file = f'{args.name}/scan{args.id}/cameras.npz'
        args.output_cameras_file = f'{args.name}/scan{args.id}/cameras_normalize.npz'

    normalize_cameras(args.input_cameras_file, args.output_cameras_file, args.number_of_cams, args.radius, args.convert_coord)
