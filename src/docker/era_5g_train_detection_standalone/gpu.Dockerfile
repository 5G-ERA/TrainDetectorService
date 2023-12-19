FROM but5gera/netapp_base_mmcv_gpu:0.2.0

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN apt-get update \
    && apt-get install -y \
    python3-pip \
    python-is-python3 \
    ffmpeg \
    wget

RUN wget -c https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_320_300e_coco/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth -O "/root/mmdetection/configs/yolo/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"

RUN wget -c https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -O "/root/mmdetection/configs/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"

RUN wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth -O "/root/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

ENTRYPOINT ["/root/start.sh"]

COPY python/era_5g_train_detection_standalone /root/era_5g_train_detection_standalone

RUN cd /root/era_5g_train_detection_standalone \
	&& pip3 install -r requirements.txt 
	
ENV PYTHONPATH=/root/era_5g_train_detection_standalone

COPY docker/era_5g_train_detection_standalone/start.sh /root/start.sh

RUN chmod +x /root/start.sh

ENV NETAPP_TORCH_DEVICE=cuda:0

EXPOSE 5896

ENV NETAPP_MODEL_VARIANT=yolov3_mobilenet

