_base_ = './htc_r50_fpn_1x_nuim.py'
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
