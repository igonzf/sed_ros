# sed_ros

The `sed_ros` package is an implementation in ROS 2 for real-time sound event detection using the ATST-SED model trained with the DESED dataset.

## Model Repository

This package uses the ATST-SED model, based on the work found in [ATST-SED GitHub repository](https://github.com/Audio-WestlakeU/ATST-SED/tree/main). For more details on the methodology and the model, refer to the related paper:

**Citation:**

Shao, Nian, Li, Xian, and Li, Xiaofei, "Fine-Tune the Pretrained ATST Model for Sound Event Detection", ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2024, pp. 911-915, doi: [10.1109/ICASSP48485.2024.10446159](https://doi.org/10.1109/ICASSP48485.2024.10446159).

## Installation

To install `sed_ros` in your ROS 2 workspace, follow these steps:

```bash
cd ~/ros2_ws/src
git clone https://github.com/igonzf/sed_ros.git
cd ~/ros2_ws/src/sed_ros/model
```

Download the pretained chekpoint files [atst_as2M.ckpt](https://drive.google.com/file/d/1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz/view), [stage_1.ckpt](https://drive.google.com/uc?id=1_sGve3FySPEqZQKYDO_DVntZ-VWVhtWN) and the [model](https://huggingface.co/igonzf/ATST-SED/blob/main/model.pth)

Modify the necessary paths in the 'confs/stage2.yaml' configuration file to point to your own input and output file locations.

```bash
cd ~/ros2_ws
colcon build
source ~/ros2_ws/install/setup.bash
```

## Usage

```bash
ros2 run sed_ros sed_detection
```
