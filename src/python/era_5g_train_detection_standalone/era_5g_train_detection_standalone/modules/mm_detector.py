
import numpy as np
import os

from mmdet.evaluation import get_classes
from mmdet.apis import init_detector, inference_detector


# mmDetection model config and checkpoint files (=allowed values of NETAPP_MODEL_VARIANT env variable)
MODEL_VARIANTS = {

    

    'yolov3_mobilenet': {
        'config_file': 'configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py',
        'checkpoint_file': 'configs/yolo/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth',
        'with_masks': False
    },

    'mask_rcnn_r50': {
        'config_file': 'configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py',
        'checkpoint_file': 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0'
                           '.37_20200504_163245-42aa3d00.pth',
        'with_masks': True
    },

    'yolox-l': {
        'config_file': 'configs/yolox/yolox_l_8xb8-300e_coco.py',
        'checkpoint_file': 'configs/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        'with_masks': False
    },

    'dino_r50': {
        'config_file': 'configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
        'checkpoint_file': 'configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth',
        'with_masks': False
    },

    # Example:
    # 'model_variant_name': {
    #    'config_file': '',
    #    'checkpoint_file': '',
    #    'with_masks': True
    # },
}

class MMDetector():
    """Universal detector class (various detectors from on MMDET package can be selected).
    """

    def __init__(self, class_id_filter=None, **kw):
        """
        Constructor

        Args:
            class_id_filter (int): ID of the class to keep (all other detections are removed).

        Raises:
            RuntimeError: Raised when initialization failed, e.g. when required parameter is not set.
        """

        super().__init__(**kw)
        self.class_id_filter = class_id_filter
        # path to the folder where the mmdet module is installed
        self.path_to_mmdet = os.getenv("NETAPP_MMDET_PATH", None)
        # the selected model variant, possible values are listed in MODEL_VARIANTS above
        self.model_variant = os.getenv("NETAPP_MODEL_VARIANT", None)
        # the device where the inference takes a place ('cpu', 'cuda', 'cuda:0', etc.)
        self.torch_device = os.getenv("NETAPP_TORCH_DEVICE", 'cpu')
        if not self.path_to_mmdet:
            raise RuntimeError(
                f"Failed to load mmdet module, env variable NETAPP_MMDET_PATH not set")
        if not os.path.exists(self.path_to_mmdet):
            raise RuntimeError(
                f"Failed to load mmdet module, path {self.path_to_mmdet} does not exist")
        elif not self.model_variant:
            raise RuntimeError(f"Failed to load model, env variable NETAPP_MODEL_VARIANT not set")
        config_file = os.path.join(self.path_to_mmdet, MODEL_VARIANTS[self.model_variant]['config_file'])
        checkpoint_file = os.path.join(self.path_to_mmdet, MODEL_VARIANTS[self.model_variant]['checkpoint_file'])
        self.with_masks = MODEL_VARIANTS[self.model_variant]['with_masks']
        self.model = init_detector(config_file, checkpoint_file, device=self.torch_device)

    def process_image(self, frame):
        """
        Detects the objects of selected classes the incoming frame and returns all detections.

        Args:
            frame (_type_): The passed image

        Returns:
            list(tuple(bbox[], score, class_id, class_name)): The list of detected objects,
            with bounding box (x1, y1, x2, y2, top-left bottom-right corners), score (0..1), 
            class_id and class_name.
        """

        if frame is not None:
            # gets results from detector
            result = inference_detector(self.model, frame)
            # convert results to the standard format
            return self._convert_mmdet_result(result, merged_data=False)
        else:
            raise RuntimeError("Image frame cannot be None.")

    def _convert_mmdet_result(self, result, dataset='coco', score_thr=0.5, merged_data=True):
        """Convert raw results from mmDet detector."""

        masks_raw = None
        if self.with_masks:
            masks_raw = result.pred_instances.masks.cpu().numpy()

        class_ids_raw = result.pred_instances.labels.cpu().numpy()
        bboxes_raw = result.pred_instances.bboxes.cpu().numpy()
        scores_raw = result.pred_instances.scores.cpu().numpy()

        # Filter by class id
        if self.class_id_filter is not None:
            class_filter_inds = np.where(class_ids_raw == self.class_id_filter)[0]
            bboxes_raw = bboxes_raw[class_filter_inds]
            scores_raw = scores_raw[class_filter_inds]
            # all class ids will be the same (but keep the data structure compatible)
            class_ids_raw = class_ids_raw[class_filter_inds]

        filtered_inds = np.where(scores_raw > score_thr)[0]
        bboxes = bboxes_raw[filtered_inds]
        scores = scores_raw[filtered_inds]

        all_class_names = get_classes(dataset)
        class_ids = [class_ids_raw[i] for i in filtered_inds]
        class_names = [all_class_names[i] for i in class_ids]

        if self.with_masks:
            filtered_masks = masks_raw[filtered_inds]  # filter by given confidence threshold

        if merged_data:
            if self.with_masks:
                converted_result = list(zip(bboxes, scores, class_ids, class_names, filtered_masks))
            else:
                converted_result = list(zip(bboxes, scores, class_ids, class_names))
        else:
            converted_result = {"bboxes": bboxes, "scores": scores, "class_ids": class_ids, "class_names": class_names}
            if self.with_masks:
                converted_result["masks"] = filtered_masks

        return converted_result
