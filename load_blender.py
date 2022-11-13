import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi),  np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0,  np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    """
        theta, phi, radius: the spherical coordinates

        returns the correspoding Cartesian coordinates
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    """
        basedir: string, the data directory
        half_res: boolean, whether to downsample
        testskip: int, the step size

        returns
            imgs: an N * H * W * C ndarray, NOTE that the images have 4 channels (RGBA)
            poses: an N * 4 * 4 ndarray, NOTE that it's actually the extrinsic matrices, with row 4 [0, 0, 0, 1]
            rendered_poses: a 40 * 4 * 4 ndarray, the poses to render, also in a form of extrinsic matrix
            i_split: a list [i_train, i_val, i_test], each of element is also a list of the indices of images
    """


    splits = ['train', 'val', 'test']
    # `metas` loads the corresponding JSON configuration
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # set the step size of loading image data
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        # Load the corresponding images and poses data
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        
        # Normalize images and reduce precision to shorten training time
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        all_imgs.append(imgs)
        all_poses.append(poses)
        # Count the number of images in train, val and test datasets respectively
        counts.append(counts[-1] + imgs.shape[0])
    
    # Get the indices of train, val and test datasets
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # Get height, width and focal length
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    """
        Generate the poses to render.
        The path is a sphere around the object.
        radius and phi are fixed, while theta = `angle`
    """
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # Downsample
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


