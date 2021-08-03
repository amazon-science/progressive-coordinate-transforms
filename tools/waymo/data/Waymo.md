# Waymo Usage

For monocular 3D detection, we organize waymo dataset in kitti format.

## Setup

### Install Waymo Open Dataset precompiled packages.

Note: the following command assumes TensorFlow version 2.1.0. 
You may modify the command according to your TensorFlow version. 
See [official guide](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) 
for additional details.

```
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0 --user
```
If the waymo-open-dataset installation failed, you can download the suitable version from [here](https://pypi.org/project/waymo-open-dataset-tf-2-3-0/#files), and install it manually.

### Download Waymo open dataset and unzip data

Data can be downloaded from the [official website](https://waymo.com/open/download/).

Decompress the zip files into different directories.
```
ls *.tar | xargs -i tar xvf {} -C your_target_dir
```

Each directory should contain tfrecords.
Example:
```
waymo
├──ImageSets
├──── train.txt
├──── val.txt
├──raw_data
├──── training
├──── validation
├──── testing
```
### Convert dataset to KITTI format

```
python converter.py <load_dir> <save_dir> [--prefix prefix] [--num_proc num_proc]
```
- load_dir: directory to load Waymo Open Dataset tfrecords
- save_dir: directory to save converted KITTI-format data
- (optional) prefix: prefix to be added to converted file names
- (optional) num_proc: number of processes to spawn

The generated data are shown as:
```
waymo2kitti
├── training
├──── segment_id
├──────── calib
├──────── image_0
├──────── image_1
├──────── image_2
├──────── image_3
├──────── image_4
├──────── label_0
├──────── label_1
├──────── label_2
├──────── label_3
├──────── label_4
├──────── label_all
├──────── projected_points_0
├──────── velodyne

├── validation
├──── segment_id
├─────── ...
```

### Prepare for common training

For training like KITTI, we link the data using setup_split.py.
```
python setup_split.py
```
New imagesets and data will be shown as:
```
kitti
├── image
├──── 000000.png
├──── 000001.png
...
├── calib
├── depth
├── label
├── train.txt
├── val.txt
```

## Evaluation

We also conduct waymo evaluation in KITTI format. You can change the predefined data path to your own.
Example:
```
python waymo_eval.py
```

## Contributing

This code benefits from converter [Here](https://github.com/caizhongang/waymo_kitti_converter) and evaluation code [Here](https://github.com/GeorgeBohw/evaluate_waymo/tree/22f0c0eba1bea67caf096a019f8b9702948a52a1).
