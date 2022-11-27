import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # include the origin coordinates
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            # use log steps like the essay
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # use normal steps
            freq_bands = torch.linspace(2.** 0., 2. ** max_freq, steps=N_freqs)
        
        # generate the main part of embedding function \gamma
        # recap that \gamma(p) = (sin(2^0 πp), cos(2^0 πp), ..., sin(2^{L-1} πp), cos(2^{L-1} πp))
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        
        """
            embed_fns: lists of function, [2 * N_freq]. the embedding function \gamma
            out_dim: int. the output dimension of the embedding function. The value is d * 2 * N_freq (+ d (if `include_input` is True))
        """
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        """
            Embedding Process

            Args:
                input:
            
            Returns:
                <class 'torch.Tensor'> [---]. 
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, i=0):
    '''
        Get the embedder of the corresponding dimensions

        Args:
            multries: int. log2 of max freq for positional or directional encoding, namely 'L' in the essay
            i: int. set 0 for default positional encoding, -1 for none

        Returns:
            embed: function. the embedding function
            embedder_obj.out_dim: int. the dim of input
    '''

    # `i == -1` indicates no embedding 
    if i == -1:
        return nn.Identity(), 3
    
    """
        include_input: boolean, whether to include the origin coordinates when encoding
        input_dims: int, the dims of input
        max_freq_log2: int
        num_freq: int
        log_sampling: boolean, whether to use log steps
        periodic_fns: list of functions, functions used to position encoding
    """

    embed_kwargs = {
        'include_input' : True,
        'input_dims' : 3,
        'max_freq_log2' : multires - 1,
        'num_freqs' : multires,
        'log_sampling' : True,
        'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
            Args:
                D: int. number of layers.
                W: int. number of channels.
                input_ch: int. number of position input channel.
                input_ch_views: int, number of direction input channel.
                skips: int. index of the layer that adds position encoding again.
                use_viewdirs: boolean. whether to use 5D input.
        """

        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        ## Build the MLP layers
        # the first 8 MLP layers dealing with position information
        # !NOTE at layer `skips`, add position encoding again
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])
        
        # the MLP layers dealing with direction information

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)] + [nn.Linear(W // 2, W // 2) for i in range(D // 2)])
        
        # output layers
        if use_viewdirs:
            # if it's 5D input, the output contains features, alpha(density) and RGB colors.
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            # if it's 3D input, the output is only density.
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
            Encode input (pos + dir) to RGB + sigma

            Args:
                x:  <class 'torch.Tensor'>, [batch, x, y, z, dir]. the packed input of position and direction

            Returns:
                output:  <class 'numpy.ndarray'>, [batch, RGB, sigma]. the packed output of RGB and sigma
        """

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # Propagate through the layers dealing with position
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # if `i` is the `skip` layer, then add position encoding again
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        # Propagate through the layers dealing with direction and get the raw output
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    """
        Get the origins and directions of all rays of the image in the world coordinate.

        Args:
            H: int. Height of image in pixels
            W: int. Width of image in pixels
            K: <class 'numpy.ndarray'>, [3, 3]. Intrinsic matrix.
            c2w:  <class 'torch.Tensor'>, [3, 4]. Extrinsic matrix.
        
        Returns:
            rays_o:  <class 'torch.Tensor'>, [H, W, 3]. coordinates of camera. 
            rays_d:  <class 'torch.Tensor'>, [H, W, 3]. directions of rays.
    """

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    # `torch.meshgrid()` is the transpose of `np.meshgrid`, thus we need to transpose here.
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], axis=-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """
        Get the origins and directions of all rays of the image in the world coordinate.

        Args:
            H: int. Height of image in pixels
            W: int. Width of image in pixels
            K: <class 'numpy.ndarray'>, [3, 3]. Intrinsic matrix.
            c2w: <class 'numpy.ndarray'>, [3, 4]. Extrinsic matrix.
        
        Returns:
            rays_o: <class 'numpy.ndarray'>, [H, W, 3]. coordinates of camera. 
            rays_d: <class 'numpy.ndarray'>, [H, W, 3]. directions of rays.
    """

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # `dirs` = K.inv @ (u, v, 1)
    # `dirs` = [(i - W / 2) / f, -(j - H / 2) / f, -1]
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], axis=-1) # [H, W, 3]
    # Rotate ray directions from camera frame to the world frame
    # rays_d = c2w @ dirs
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
