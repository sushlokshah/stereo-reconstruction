# stereo-reconstruction
The Aim is to reconstruct a 3D structure, given the pair of stereo images.

# Methodology
  1. first compute calibration matrix,baseline and projection matrices using information given in calibration_info.txt .
  2. compute disparity from two images.
  3. compute 3d pointcloud using disparity map genrated previously using baseline calibration matrix found above.
# Run
```bash
    git clone https://github.com/sushlokshah/stereo-reconstruction.git
    cd stereo-reconstruction
      #Edit path.yaml file
      #image_1_path: [img_path]
      #image_2_path: [img_path]
      #calibration_info_path: [calibration_info path]
      #output_path: [folder path]
    python3 stereo_reconstruction.py
      #Result is obtained in output.ply at folder specified, which can be viewed using MeshLab, Open3D, or any other related software or library to view point clouds.
```
# Result
|Stereo Image 1| Stereo Image 2 | Disparity Map  | 3d reconstruction |
| -------- | -------- | --- | -------- |
| <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/bike/im0.png" width="250" height="250" />   | <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/bike/im1.png" width="250" height="250" />      |<img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/bike/Screenshot%202021-06-22%20090220.png" width="250" height="250" />  | <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/bike/output.png" width="250" height="250" />       |
| <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/cycle/imb0.png" width="250" height="250" />   | <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/cycle/imb1.png" width="250" height="250" />      |<img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/cycle/Screenshot%202021-06-22%20090548.png" width="250" height="250" />  | <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/cycle/output.png" width="250" height="250" />       |
| <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/umbrella/ima0.png" width="250" height="250" />   | <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/umbrella/ima1.png" width="250" height="250" />      |<img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/umbrella/Screenshot%202021-06-22%20091041.png" width="250" height="250" />  | <img src="https://github.com/sushlokshah/stereo-reconstruction/blob/main/umbrella/Screenshot%202021-06-22%20091008.png" width="250" height="250" />       |
