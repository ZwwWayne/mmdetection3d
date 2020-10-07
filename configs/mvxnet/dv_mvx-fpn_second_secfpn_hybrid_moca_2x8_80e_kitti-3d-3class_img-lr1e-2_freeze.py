_base_ = './dv_mvx-fpn_second_secfpn_hybrid_moca_2x8_80e_kitti-3d-3class.py'

model = dict(img_backbone=dict(frozen_stages=4))
optimizer = dict(
    constructor='HybridOptimizerConstructor',
    pts=dict(
        type='AdamW',
        lr=0.003,
        betas=(0.95, 0.99),
        weight_decay=0.01,
        step_interval=1),
    img=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        step_interval=1))
