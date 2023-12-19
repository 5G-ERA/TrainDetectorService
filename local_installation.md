
# Train Detector Service - Local Installation

This Net Application is mainly intended to be used in a form of a docker container. However, it can also be installed locally.

The repository was tested with CUDA 11.7.1 and cuDNN 8.5.0.96 on Ubuntu 20.04, with python 3.8 and pytorch 1.13.1, MMCV 2.1.0 and MMDetection 3.2.0.

Even though this repository can be used in CPU-only setting, it is recommended to install CUDA and use GPU.

All python packages are recommended to be installed inside a virtual environmnet.


## Local Installation

1. Install CUDA and cuDNN 

   - CUDA installation:
     ```
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-
     keyring_1.0-1_all.deb
     sudo dpkg -i cuda-keyring*.deb
     sudo apt-get update
     sudo apt-get install cuda-toolkit-11-7
     ``` 

   - cuDNN installation:
     1. Find the correct cuDNN version number (for the particular CUDA version) from: 
        [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
     2. and then the corresponding download link from:
        [https://developer.download.nvidia.com/compute/redist/cudnn/](https://developer.download.nvidia.com/compute/redist/cudnn/)
     3. Download the resultant file, e.i. for Ubuntu 20.04, CUDA 11.7 and cuDNN 8.5.0.96 it is: https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb
     4. Install it using sudo dpkg -i package_file.deb
        ```
        sudo dpkg -i cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb
        ```
    - Reboot the machine.
  
2. Install pytorch and MMCV 
   - CPU-only version (not recommended)
     ```
     pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
     pip install mmcv==2.1.0
     ```
    
   - GPU version
     ```
     pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
     pip install mmcv==2.1.0
     ```
     - Note: If MMCV was previously installed in CPU-only setting, it will most likely be necessary to rebuild it (pip uninstall it, clean the files cached by pip, and pip install again).
    
3. Install MMDetection (3.2.0)
   ```
   export MMDET_PATH="_your_selected_path_"
   git clone -b v3.2.0 https://github.com/open-mmlab/mmdetection.git "$MMDET_PATH"
   cd "$MMDET_PATH"
   pip install .
   ```

4. Download the weights of selected pretrained detector network (YOLO is used as the default option):
   ```
   wget -c https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_320_300e_coco/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth -O "$MMDET_PATH/configs/yolo/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
   ```

5. Install requirements 
   ```
   cd src/python/era_5g_train_detection_standalone/
   pip install -r requirements.txt
   ```
      
6. Set PYTHONPATH to include `src/python/era_5g_train_detection_standalone/`

