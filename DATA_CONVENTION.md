# Data Convention

The format of our multi-view dataset is derived from [VolSDF](https://github.com/lioryariv/volsdf/blob/main/DATA_CONVENTION.md).

### Directory Structure

```python
scan<scan_id>/
	cameras.npz
	image/ -> {:04d}.png # tone-mapped LDR images
	depth/ -> {:04d}.exr
	normal/ -> {:04d}.exr
	mask/ -> {:04d}.png
	val/ -> {:04d}.png # validation images (LDR)
	hdr/ -> {:04d}.exr # raw HDR images
	# followings are optional
	light_mask/ -> {:04d}.png # emitter mask images
	material/ -> {:04d}_kd.exr, {:04d}_ks.exr, {:04d}_rough.exr # diffuse, specular albedo and roughness
```

Note that not all scenes contain emitter mask and material information. Only scenes with "relight" in the name are relightable.

### Camera Information

The `cameras.npz` contains each image's associated camera projection matrix `'world_mat_{i}'` and a normalization matrix `'scale_mat_{i}'`, the same as VolSDF. Besides, we also provide a validation set of images for novel view synthesis, whose associated camera projection matrices are `'val_mat_{i}'`. Validation set and training set share the same normalization matrix.

The normalization matrices may not be readily available in `cameras.npz`. You can manually run `data/normalize_cameras.py` to generate `cameras_normalize.npz`. Since our method requires the entire scene to be within a radius-3 bounding sphere, we suggest normalizing cameras by radius 2.0 or 2.5.

Note that we follow **OpenCV camera coordinate system** (X right, Y downwards, Z into the image plane).

### About EXR format

We suggest using OpenCV to load an `.exr` format `float32` image:

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1' # Enable OpenCV support for EXR
import cv2
...
im = cv2.imread(im_path, -1) # im will be an numpy.float32 array of shape (H, W, C)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # cv2 reads image in BGR shape, convert into RGB
```

We suggest using [tev](https://github.com/Tom94/tev) to preview HDR `.exr` images conveniently.