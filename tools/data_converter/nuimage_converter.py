import argparse
import base64
import mmcv
import numpy as np
from nuimages import NuImages
from nuimages.utils.utils import mask_decode, name_to_index_mapping
from os import path as osp
from pycocotools import mask as mask_util


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/nuimages',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        nargs='+',
        default=['v1.0-mini'],
        required=False,
        help='specify the dataset version')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/nuimages/annotations/',
        required=False,
        help='path to save the exported json')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        required=False,
        help='workers to process semantic masks')
    parser.add_argument('--extra-tag', type=str, default='nuimages')
    args = parser.parse_args()
    return args


def get_semantic_segmentation(nuim, sd_token, img_size):
    """Get semantic segmentation map for an image.

    Args:
        nuim (obj:`NuImages`): NuImages dataset object
        sd_token (str): String Token of a key frame
        img_size (tuple[int]): Size of image.

    Returns:
        np.ndarray: Semantic segmentation map of the image
    """
    name_to_index = name_to_index_mapping(nuim.category)

    # Get image data.
    (width, height) = img_size
    semseg_mask = np.zeros((height, width)).astype('int32')

    # Load stuff / surface regions.
    surface_anns = [
        o for o in nuim.surface_ann if o['sample_data_token'] == sd_token
    ]

    # Draw stuff / surface regions.
    for ann in surface_anns:
        # Get color and mask.
        category_token = ann['category_token']
        category_name = nuim.get('category', category_token)['name']
        if ann['mask'] is None:
            continue
        mask = mask_decode(ann['mask'])

        # Draw mask for semantic segmentation.
        semseg_mask[mask == 1] = name_to_index[category_name]

    # Load object instances.
    object_anns = [
        o for o in nuim.object_ann if o['sample_data_token'] == sd_token
    ]

    # Sort by token to ensure that objects always appear in the
    # instance mask in the same order.
    object_anns = sorted(object_anns, key=lambda k: k['token'])

    # Draw object instances.
    # The 0 index is reserved for background; thus, the instances
    # should start from index 1.
    for i, ann in enumerate(object_anns, start=1):
        # Get color, box, mask and name.
        category_token = ann['category_token']
        category_name = nuim.get('category', category_token)['name']
        if ann['mask'] is None:
            continue
        mask = mask_decode(ann['mask'])

        # Draw masks for semantic segmentation and instance segmentation.
        semseg_mask[mask == 1] = name_to_index[category_name]

    return semseg_mask


def export_nuim_to_coco(nuim, out_dir, extra_tag, version, nproc):
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
    }

    print('Process category information')
    categories = []
    cat2id = dict()
    for cate_info in mmcv.track_iter_progress(nuim.category):
        if cate_info['name'] in NameMapping:
            name = NameMapping[cate_info['name']]
            if name not in cat2id:
                idx = len(categories)
                categories.append(dict(id=idx, name=name))
                cat2id.update({name: idx})

    images = []
    img2id = dict()

    seg_root = f'{out_dir}semantic_masks/'
    mmcv.mkdir_or_exist(seg_root)

    print('Process image meta information...')
    for sample_info in mmcv.track_iter_progress(nuim.sample_data):
        if sample_info['is_key_frame']:
            img_idx = len(images)
            images.append(
                dict(
                    id=img_idx,
                    token=sample_info['token'],
                    file_name=sample_info['filename'],
                    width=sample_info['width'],
                    height=sample_info['height']))
            img2id.update({sample_info['token']: img_idx})

    global save_img_seg

    def save_img_seg(img_info):
        semantic_mask = get_semantic_segmentation(
            nuim,
            img_info['token'],
            img_size=(img_info['width'], img_info['height']))
        img_filename = img_info['file_name']
        seg_filename = img_filename.replace('jpg', 'png')
        seg_filename = f'{seg_root}{seg_filename}'
        mmcv.imwrite(semantic_mask, seg_filename)
        return np.max(semantic_mask)

    print('Process semantic mask information...')
    max_cls_ids = mmcv.track_parallel_progress(
        save_img_seg, images, nproc=nproc)
    max_cls_id = max(max_cls_ids)
    print(f'Max ID of class in the semantic map: {max_cls_id}')

    print('Process annotation information...')
    annotations = []
    for single_obj in mmcv.track_iter_progress(nuim.object_ann):
        category_info = nuim.get('category', single_obj['category_token'])
        if category_info['name'] in NameMapping:
            cat_name = NameMapping[category_info['name']]
            cat_id = cat2id[cat_name]
        else:
            continue

        image_id = img2id[single_obj['sample_data_token']]
        x_min, y_min, x_max, y_max = single_obj['bbox']

        mask_anno = dict()
        if single_obj['mask'] is None:
            empty_mask = np.zeros((900, 1600, 1), order='F', dtype='uint8')
            mask_anno = mask_util.encode(empty_mask)[0]
            mask_anno['counts'] = mask_anno['counts'].decode()
        else:
            mask_anno['counts'] = base64.b64decode(
                single_obj['mask']['counts']).decode()
            mask_anno['size'] = single_obj['mask']['size']

        data_anno = dict(
            image_id=image_id,
            id=len(annotations),
            category_id=cat_id,
            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
            area=(x_max - x_min) * (y_max - y_min),
            segmentation=mask_anno,
            iscrowd=0)
        annotations.append(data_anno)

    coco_format_json = dict(
        images=images, annotations=annotations, categories=categories)

    mmcv.mkdir_or_exist(out_dir)
    out_file = osp.join(out_dir, f'{extra_tag}_{version}.json')
    print(f'Annotation dumped to {out_file}')
    mmcv.dump(coco_format_json, out_file)


def main():
    args = parse_args()
    for version in args.version:
        nuim = NuImages(
            dataroot=args.data_root, version=version, verbose=True, lazy=True)
        export_nuim_to_coco(nuim, args.out_dir, args.extra_tag, version,
                            args.nproc)


if __name__ == '__main__':
    main()
