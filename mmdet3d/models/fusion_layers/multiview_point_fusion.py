import torch
from mmcv.cnn import ConvModule, NonLocal2d, build_norm_layer, xavier_init
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.ops import DynamicScatter
from .. import builder
from ..registry import FUSION_LAYERS


def obtain_img_coors(points, lidar2img_rt, img_scale_factor, img_crop_offset,
                     img_flip, img_shape):
    # project points from velo coordinate to camera coordinate
    num_points = points.shape[0]
    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
    pts_2d = pts_4d @ lidar2img_rt.t()

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    zero_mask = (pts_2d[:, 2] == 0)
    if zero_mask.any():
        pts_2d[zero_mask, 2] == 1e-9
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    orig_h, orig_w = img_shape
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        coor_x = orig_w - coor_x

    valid_mask = ((coor_x <= orig_w) & (coor_y <= orig_h) & (coor_x >= 0) &
                  (coor_y >= 0))
    if torch.isnan(coor_x).any() or torch.isnan(coor_y).any():
        import pdb
        pdb.set_trace()
    return coor_x, coor_y, valid_mask


def mvml_point_sample(
    mlvl_img_feats,
    points,
    lidar2img_rts,
    pcd_rotate_mat,
    img_scale_factor,
    img_crop_offset,
    pcd_trans_factor,
    pcd_scale_factor,
    pcd_flip,
    img_flip,
    img_pad_shape,
    img_shape,
    aligned=True,
    padding_mode='zeros',
    align_corners=True,
):
    """sample image features using point coordinates
    Arguments:
        img_features (Tensor): BxCxHxW image features from multi-view images.
        points (Tensor): Nx3 point cloud coordinates.
        P (Tensor): 4x4 transformation matrix.
        scale_factor (Tensor): scale_factor of images.
        img_pad_shape (int, int): int tuple indicates the h & w after padding,
            this is necessary to obtain features in feature map.
        img_shape (int, int): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
    return:
        Tensor: NxC image features sampled by point coordinates.
    """
    # Step 1: convert points to original points
    # aug order: flip -> trans -> scale -> rot
    # The transformation follows the augmentation order in data pipeline
    if pcd_flip:
        # if the points are flipped, flip them back first
        points[:, 1] = -points[:, 1]

    points -= pcd_trans_factor
    # the points should be scaled to the original scale in velo coordinate
    points /= pcd_scale_factor
    # the points should be rotated back
    # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
    # exactly an identity matrix
    # but use angle to create the inverse rot matrix neither.
    points = points @ pcd_rotate_mat.inverse()

    # Step 2: calculate coors and valid masks for points in all views
    mv_coor_x = []
    mv_coor_y = []
    mv_valid_mask = []
    for idx, single_lidar2img_rt in enumerate(lidar2img_rts):
        coor_x, coor_y, valid_mask = obtain_img_coors(points,
                                                      single_lidar2img_rt,
                                                      img_scale_factor[:2],
                                                      img_crop_offset,
                                                      img_flip, img_shape[:2])
        mv_coor_x.append(coor_x)
        mv_coor_y.append(coor_y)
        mv_valid_mask.append(valid_mask)

    # Step 3: obtain multi-level features from multi-view
    # init multi-level point feats for multi-view
    point_feats = [
        points.new_zeros(size=(points.shape[0], mlvl_img_feats[i].size(1)))
        for i in range(len(mlvl_img_feats))
    ]
    next_valid_mask = points.new_ones(
        size=(points.size(0), 1), dtype=torch.bool)
    for view_idx, (curr_coor_x, curr_coor_y, curr_valid_mask) in enumerate(
            zip(mv_coor_x, mv_coor_y, mv_valid_mask)):
        curr_valid_mask = next_valid_mask & curr_valid_mask

        coor_y = curr_coor_y[curr_valid_mask].unsqueeze(-1)
        coor_x = curr_coor_x[curr_valid_mask].unsqueeze(-1)
        # normalize feature map coors for grid_sample
        h, w = img_pad_shape
        coor_y = coor_y / h * 2 - 1
        coor_x = coor_x / w * 2 - 1
        grid = torch.cat([coor_x, coor_y],
                         dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

        # align_corner=False provides higher performance
        mode = 'bilinear' if aligned else 'nearest'
        for level_idx, img_features in enumerate(mlvl_img_feats):
            point_feats[level_idx][curr_valid_mask.squeeze()] = F.grid_sample(
                img_features[view_idx:view_idx + 1],
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners).squeeze().t()  # 1xCx1xN feats
            if torch.isnan(point_feats[level_idx]).any():
                import pdb
                pdb.set_trace()
        # update valid mask in the end for the next image
        next_valid_mask = next_valid_mask & curr_valid_mask.logical_not()

    point_feats = torch.cat(point_feats, dim=-1)
    return point_feats


@FUSION_LAYERS.register_module()
class MultiViewPointFusion(nn.Module):
    """Fuse image features from fused single scale features."""

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 act_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 refine_type=None,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(MultiViewPointFusion, self).__init__()
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.refine_type = refine_type
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.out_channels = out_channels

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.img_transform = nn.Sequential(
                nn.Conv1d(
                    mid_channels * len(img_channels),
                    out_channels,
                    1,
                    bias=False),
                norm_layer,
            )
        else:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.img_transform = nn.Sequential(
                nn.Conv1d(sum(img_channels), out_channels, 1, bias=False),
                norm_layer)
        norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
        self.pts_transform = nn.Sequential(
            nn.Conv1d(pts_channels, out_channels, 1, bias=False), norm_layer)

        if self.fuse_out:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                norm_layer,
                nn.ReLU(inplace=False))

        if self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                out_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                xavier_init(m, distribution='uniform')

    def forward(self, img_feats, pts, pts_feats, img_meta):
        """Forward function
        Args:
            img_feats (list[Tensor]): img features
            pts: [list[Tensor]]: a batch of points with shape Nx3
            pts_feats (Tensor): point features of the total batch
        Return:
            Tensor: fused multi-modality features
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_meta)
        # the transpose here is for avoid to big batch size
        img_pre_fuse = self.img_transform(
            img_pts.t().unsqueeze(0)).squeeze(0).t()
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(
            pts_feats.t().unsqueeze(0)).squeeze(0).t()

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out.unsqueeze(0)).squeeze(0)

        if self.refine_type is not None:
            fuse_out_T = fuse_out.t()[None, ..., None]  # NxC -> 1xCxNx1
            batch_idx = 0
            attentive = []
            for i in range(len(pts)):
                end_idx = batch_idx + len(pts[i])
                attentive.append(
                    self.refine(fuse_out_T[:, :, batch_idx:end_idx]))
                batch_idx = end_idx
            fuse_out = torch.cat(attentive, dim=-2).squeeze().t()
        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, img_meta):
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features from multi-view img
        # assume only one point cloud and 6 imgs are provided
        img_ins = [
            img_in.view(
                len(img_meta), -1, img_in.size(-3), img_in.size(-2),
                img_in.size(-1)) for img_in in img_ins
        ]
        for i in range(len(img_meta)):
            curr_img_ins = [img_in[i] for img_in in img_ins]
            mlvl_img_feats = self.sample_single(curr_img_ins, pts[i],
                                                img_meta[i])
            if torch.isnan(mlvl_img_feats).any():
                import pdb
                pdb.set_trace()
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta):
        pcd_scale_factor = (
            img_meta['pcd_scale_factor']
            if 'pcd_scale_factor' in img_meta.keys() else 1)
        pcd_trans_factor = (
            pts.new_tensor(img_meta['pcd_trans'])
            if 'pcd_trans' in img_meta.keys() else 0)
        pcd_rotate_mat = (
            pts.new_tensor(img_meta['pcd_rotation']) if 'pcd_rotation'
            in img_meta.keys() else torch.eye(3).type_as(pts).to(pts.device))
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'])
            if 'scale_factor' in img_meta.keys() else [1] * 6)
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        lidar2img_rts = [
            pts.new_tensor(lidar2img_rt)
            for lidar2img_rt in img_meta['lidar2img']
        ]
        pcd_flip = (
            img_meta['pcd_flip'] if 'pcd_flip' in img_meta.keys() else False)
        img_flip = (img_meta['flip'] if 'flip' in img_meta.keys() else False)
        img_pts = mvml_point_sample(
            img_feats,
            pts[:, :3],
            lidar2img_rts,
            pcd_rotate_mat,
            img_scale_factor,
            img_crop_offset,
            pcd_trans_factor,
            pcd_scale_factor,
            pcd_flip=pcd_flip,
            img_flip=img_flip,
            img_pad_shape=img_meta['pad_shape'][:2],
            img_shape=img_meta['img_shape'],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts


@FUSION_LAYERS.register_module()
class MultiViewPointFusionV2(MultiViewPointFusion):
    """Fuse image features from fused single scale features."""

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 activate_out=True,
                 fuse_out=False,
                 refine_type=None,
                 lateral_conv=True,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 img_refine_type='conv'):
        super(MultiViewPointFusionV2,
              self).__init__(img_channels, pts_channels, mid_channels,
                             out_channels, img_levels, conv_cfg, norm_cfg,
                             activation, activate_out, fuse_out, refine_type,
                             aligned, align_corners, padding_mode,
                             lateral_conv)

        self.img_refine_type = img_refine_type
        self.refine_level = refine_level
        self.lateral_convs = None
        self.img_transform = nn.Sequential(
            nn.Linear(img_channels, mid_channels),
            nn.BatchNorm1d(mid_channels, eps=1e-3, momentum=0.01),
        )
        if img_refine_type == 'conv':
            self.img_refine = ConvModule(
                img_channels,
                img_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        elif img_refine_type == 'non_local':
            self.img_refine = NonLocal2d(
                img_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        self.init_weights()

    def obtain_mlvl_feats(self, img_feats, pts, img_meta):
        feats = []
        gather_size = img_feats[self.refine_level].size()[2:]
        for i in self.img_levels:
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    img_feats[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    img_feats[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)
        if self.refine_type is not None:
            bsf = self.img_refine(bsf)

        # Sample multi-level features
        img_feats_per_point = []
        for i in range(len(img_meta)):
            img_feats_per_point.append(
                self.sample_single(bsf[i:i + 1], pts[i][:, :3], img_meta[i]))
        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts


@FUSION_LAYERS.register_module()
class MultiViewPointFusionV3(MultiViewPointFusion):
    """Fuse image features from fused single scale features."""

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 activate_out=True,
                 fuse_out=False,
                 refine_type=None,
                 img_gate=None,
                 scatter=None,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(MultiViewPointFusionV3, self).__init__(
            img_channels=img_channels,
            pts_channels=pts_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            img_levels=img_levels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=activation,
            activate_out=activate_out,
            fuse_out=fuse_out,
            refine_type=refine_type,
            aligned=aligned,
            align_corners=align_corners,
            padding_mode=padding_mode,
            lateral_conv=lateral_conv)

        self.pts_transform = nn.Sequential(
            nn.Conv2d(pts_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        )
        self.img_gate = img_gate
        if self.img_gate is not None:
            self.img_gate = builder.build_voxel_encoder(img_gate)
            self.gate_out = nn.Linear(img_gate.num_filters[-1], 1)

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))
        self.feature_scatter = DynamicScatter(**scatter)
        self.init_weights()

    def forward(self, img_feats, pts, pts_feats, coors, img_meta):
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_meta)
        img_pre_fuse = self.img_transform(img_pts)
        pts_pre_fuse = self.pts_transform(pts_feats)

        # voxelize img_pre_fuse
        if self.img_gate is not None:
            pts_feats = torch.cat(pts)
            img_weights = self.img_gate(pts_feats, coors)
            img_weights = torch.sigmoid(self.gate_out(img_weights))
            img_pre_fuse = img_pre_fuse * img_weights
        img_pre_fuse, voxel_coors = self.feature_scatter(img_pre_fuse, coors)
        img_pre_fuse = self.map_voxel_center_to_voxel(
            img_pre_fuse, voxel_coors, self.feature_scatter.point_cloud_range,
            self.feature_scatter.voxel_size)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)
        if self.refine_type is not None:
            self.refine(fuse_out)
        return fuse_out

    def map_voxel_center_to_voxel(self, voxel_mean, voxel_coors,
                                  point_cloud_range, voxel_size):
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
        canvas_y = int(
            (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        canvas_x = int(
            (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        canvas_channel = voxel_mean.size(1)
        batch_size = voxel_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()
        canvas = canvas.reshape(canvas_channel, batch_size, canvas_z, canvas_y,
                                canvas_x).transpose(1, 0)
        return canvas.squeeze(2)


@FUSION_LAYERS.register_module()
class MultiViewPointFusionV4(MultiViewPointFusionV3):
    """Fuse image features from fused single scale features."""

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 activate_out=True,
                 fuse_out=False,
                 refine_type=None,
                 img_gate=None,
                 scatter=None,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(MultiViewPointFusionV4, self).__init__(
            img_channels=img_channels,
            pts_channels=pts_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            img_levels=img_levels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=activation,
            activate_out=activate_out,
            fuse_out=fuse_out,
            scatter=scatter,
            refine_type=refine_type,
            aligned=aligned,
            align_corners=align_corners,
            padding_mode=padding_mode,
            lateral_conv=lateral_conv)

        self.pts_transform = nn.Sequential(
            nn.Conv2d(pts_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        )
        self.img_transform = nn.Sequential(
            nn.Conv2d(
                mid_channels * len(self.img_levels),
                out_channels,
                1,
                bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        )
        self.img_gate = img_gate
        if self.img_gate is not None:
            self.img_gate = builder.build_voxel_encoder(img_gate)
            self.gate_out = nn.Linear(img_gate.num_filters[-1], 1)

        self.init_weights()

    def forward(self, img_feats, pts, pts_feats, coors, img_meta):
        img_pre_fuse = self.obtain_mlvl_feats(img_feats, pts, img_meta)
        # voxelize img_pre_fuse
        if self.img_gate is not None:
            pts_feats = torch.cat(pts)
            img_weights = self.img_gate(pts_feats, coors)
            img_weights = torch.sigmoid(self.gate_out(img_weights))
            img_pre_fuse = img_pre_fuse * img_weights
        img_pre_fuse, voxel_coors = self.feature_scatter(img_pre_fuse, coors)
        img_pre_fuse = self.map_voxel_center_to_voxel(
            img_pre_fuse, voxel_coors, self.feature_scatter.point_cloud_range,
            self.feature_scatter.voxel_size)

        img_pre_fuse = self.img_transform(img_pre_fuse)
        pts_pre_fuse = self.pts_transform(pts_feats)
        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)
        if self.refine_type is not None:
            self.refine(fuse_out)
        return fuse_out
