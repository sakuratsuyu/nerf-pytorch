import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    # judge whether the images already exist
    needtoload = False
    for r in factors:
        # directory path to store downsampled images
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        # directory path to store downsampled images
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    # if the images exist, then there is no need to load
    if not needtoload:
        return
    
    # from shutil import copy
    from subprocess import check_output
    
    # load images from the directory
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    # filter and leave the images with extention 'JPG', 'jpg', 'jpeg', 'png' and 'PNG'.
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'jpeg', 'png', 'PNG']])]
    imgdir_orig = imgdir
    
    # store current working directory
    wd = os.getcwd()

    for r in factors + resolutions:
        # if `r` is in `factors`
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        # if `r` is in `resolutions`
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        # Create directory to store downsampled images
        os.makedirs(imgdir)
        # Copy images to `imgdir`
        '''
            `check_output` can execute a shell command and return the content printed in STDOUT
        '''
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        # Get the extension of the image file
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        # change to directory of the downsampled images
        os.chdir(imgdir)
        # Use `mogrify` command to resize the images, while converting it to 'png'.
        check_output(args, shell=True)
        # go back to the working directory
        os.chdir(wd)
        
        # Remove duplicates images if the origin images are ont 'png'
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    '''
    `*.npy` is a file created by numpy.
        - To save a numpy array, we can use `np.save(<file_path>, <numpy_array>)`.
        - To load a numpy array, we can use `np.load(<file_path>)`
    '''
    # Here, `pose_arr` is loaded as a (20, 17) ndarray.
    #     - `poses` is a (3, 5, 20) ndarray.
    #     - `bds` (depth bounds) is a (2, 20) ndarray.
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    
    # `img0` is the file path of the first image. (i.e. ./data/nerf_llff/fern/images/IMG_4026.JPG)
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # Read the first image and get its shape, which is (3024, 4032, 3).
    # Here we assume that the shapes of all pictures are the same.
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    ## Downsample Process 下采样
    # if specified the scale factor
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    # if specified the height
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    # if specified the width
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    # nothing specified
    else:
        factor = 1
    
    # Get the directory path of the downsampled images
    imgdir = os.path.join(basedir, 'images' + sfx)
    # if the directory does not exist
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    # Get the file paths of all images
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # if the number of the images is not equal to that of the poses
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # Get the shape of the first downsampled image
    sh = imageio.imread(imgfiles[0]).shape
    # `sh[:2]` means the height and the width
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    '''
    values stored in `poses[:, :, i]` of each image
        - R is a 3 by 3 roation matrix.
        - t is the translation vector.
        - H, W, f are the height, width, and focal length.
            0   1   2   3   4
        0   /   |   \   |   H
        1   -   R   -   t   W
        2   \   |   /   |   f

    '''

    # if there is no need to load images, then just return `poses` and `bds`
    if not load_imgs:
        return poses, bds
    
    # otherwise, load images and return `imgs` additionally

    def imread(f):
        # if the extension name of `f` is '.png', then ignore the Gamma Corretion 伽马校正
        # https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pillow_legacy.html
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    # load all imgs and normalize it (make the value in [0, 1])
    # imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = [imread(f) / 255. for f in imgfiles]
    # (20, 378, 504, 3) -> (378, 504, 3, 20)
    imgs = np.stack(imgs, -1)
    
    # recap that `pose[:, -1, 0]` means height, width and focal length
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    try:
        return x / np.linalg.norm(x)
    except:
        pass

def viewmatrix(z, up, pos):
    # Somehow calculate the 'mean' value of rotation
    # and represent by `vec0`, `vec1`, `vec2`.
    # Why to normalize again?
    vec2 = normalize(z)
    vec1_avg = up
    '''
        `np.cross()` calculate the cross product of two vector
    '''
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))

    m = np.stack([vec0, vec1, vec2, pos], axis=1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    # Get height, widhth and focal length
    # the usage of `-1:` is to get the last column.
    hwf = poses[0, :, -1:]

    # Caculate the mean value of translation vector t
    center = poses[:, :, 3].mean(0)

    # Caculate the mean value of the third column of rotation matrix R
    vec2 = normalize(poses[:, :, 2].sum(0))
    # Caculate the sum of the second column of rotation matrix R
    up = poses[:, :, 1].sum(0)

    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], axis=1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    """
        Generate the poses to render.
        The path is a spiral.

        c2w: world to camera matrix
        up: Up Vector
        rads: --
        focal: focal length
        zdelta: --
        zrate: --
        rots: times of rotation
        N: number of view at a revolution
    """
    
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):
    '''
        Calculate the average of `poses`, which can be seen as a matrix.
        All `poses` times the inverse of that matrix to recenter `poses`.
        i.e. redefine the world coordinate, let the origin in the center of the object.
        After that, the rotation matrix R is orthonormal and traslation vector t is 0 vector.
    '''

    # Make a copy of `poses`
    poses_ = poses + 0

    '''
        `c2w` is the 'average' world to camera matrix.
        Actually, `np.linalg.inv(c2w)` is the Camera TO World matrix and thus the name of the variable `c2w`.
    '''

    # Calculate the average world to camera matrix
    c2w = poses_avg(poses)
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = np.concatenate([c2w[:, :4], bottom], -2)
    
    # Add the bottom vector to each w2c matrix to build up the extrinsic matrices
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    # Recenter `poses`
    poses = np.linalg.inv(c2w) @ poses

    # We only need the first 3 rows of the extrinsic matrices and hwf remains
    poses_[:, :, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    """
        basedir: string, the data directory
        factor: int, the downsampling factor
        recenter: boolean, whether to recenter
        bd_factor: float, rescale factor
        spherify: boolean, whether to spherify
        path_zflat: boolean, ---

        returns
            imgs: an N * H * W * C ndarray, NOTE that the images have 4 channels (RGBA)
            poses: an N * 4 * 4 ndarray, NOTE that it's actually the extrinsic matrices, with row 4 [0, 0, 0, 1]
            rendered_poses: a 40 * 4 * 4 ndarray, the poses to render, also in a form of extrinsic matrix
            i_split: a list [i_train, i_val, i_test], each of element is also a list of the indices of images
    """

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    '''
    Correct rotation matrix ordering and move variable dim to axis 0
    `poses` (3, 5, 20) -> (20, 3, 5)
        > R matrix is in the form [down right back] instead of [right up back] due to LLFF standard
    `bds` (2, 20) -> (20, 2)
    `images` (378, 504, 3, 20) -> (20, 378, 504, 3)
    '''
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], axis=1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    
    # Rescale if bd_factor is provided
    # Scale the translation vector and depth bounds (?)
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        # Get the recentered (or something like normalized) matrix
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. /(((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        # shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :, 3] # ptstocam(poses[:3, 3, :].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = - close_depth * .1
            c2w_path[:3,3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
    
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    # `i_test` is the arg of the smallest `dists``,
    # which somehow means it's the cloest (holdout) to the 'average'.
    dists = np.sum(np.square(c2w[:, 3] - poses[:, :, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    # Reduce precision to shorten training time
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



