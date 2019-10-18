import random
import numpy as np

from datasets import shape

from visualize import display_dataset


if __name__ == '__main__':
    tr_ds = shape.load_data(count=500)
    val_ds = shape.load_data(count=50)

    display_dataset(tr_ds, n=4)

    # Test on a random image
    image_id = random.choice(val_ds.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))