# 3D-Orientation-Field-Transform
3D Orientation Field Transform for Vascular Structure Enhancement  This repository contains the implementation of the 3D Orientation Field Transform (OFT) algorithm for enhancing vascular and tubular structures in noisy 3D images. 

This repository provides an implementation of the **3D Orientation Field Transform (OFT)** algorithm, which enhances vascular and tubular structures in noisy 3D images. The method is adapted from the research paper that introduces the 3D OFT as an effective solution for enhancing tubular structures in both synthetic and real-world datasets, including transmission electron microscopy (TEM) tomograms.

> **Note:** This repository is not original research but an implementation based on the algorithm described in the referenced paper. See the citation section for details.

## Features
- **Orientation Field Transform (OFT) Filter**: Enhances 3D tubular structures using the combination of the maximum, mean, and absolute deviation of line integrals and alignment integrals.
- **Vascular Enhancement in Noisy Data**: Handles noisy, oriented, and curved structures, performing well in low signal-to-noise ratio (SNR) conditions.
- **Data Preprocessing**: Includes noise reduction and intensity normalization to prepare images for vascular enhancement.
- **Applicable to 3D Volumes**: While primarily designed for 3D images, the algorithm can also be applied to 2D images with simplified settings.
- **Modular and Flexible**: Can be used in conjunction with other image processing techniques for tasks like segmentation and detection.

## Requirements
- Python 3.9
- Cython
- SciPy
- Additional dependencies as listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/3D-Orientation-Field-Transform.git
   cd 3D-Orientation-Field-Transform
2. Install the required Python packages:
   ```bash
    pip install -r requirements.txt
## Usage 
   Applying the OFT Filter:
   ```bash
   python oft_3d.py --input <preprocessed_image> --output <enhanced_image>


python oft_3d.py --input <preprocessed_image> --output <enhanced_image>

python oft_3d.py --input synthetic_volume.nii --output enhanced_volume.nii --noise_level 0.5 --tuning_params 3,1,0.5
```

# Vessel Generation with Bifurcations in 3D

This code generates a synthetic 3D vascular structure with bifurcations and outputs both a filled vessel representation and a noisy volume representation. The vessel model includes random walk behavior for vessel propagation and random orientation updates, with bifurcation points introduced to simulate natural branching. This README explains the components of the code and its functionality.

## Features
- Generate synthetic vessel datasets with configurable bifurcation count.
- Apply spatial and angular regularization.
- Add multiplicative noise with configurable mean and standard deviation.
- Save the generated dataset, including:
  - 3D volume with and without noise.
  - Vessel centerline.
  - Vessel radius.

## Usage
### Main Script
To run the dataset generation script, use the following command:
```bash
python <script_name>.py <Bif_Number> <D_33> <D_44> <Vessel_Length> <mean> <std>
```

### Arguments
| Argument         | Type      | Description                                | Default |
|------------------|-----------|--------------------------------------------|---------|
| `Bif_Number`     | `int`     | Number of bifurcations to generate.        | `10`    |
| `D_33`           | `float`   | Spatial regularization coefficient.        | `1.0`   |
| `D_44`           | `float`   | Angular regularization coefficient.        | `1.0`   |
| `Vessel_Length`  | `int`     | Length of each vessel.                     | `15`    |
| `mean`           | `float`   | Mean of the multiplicative noise.          | `0.0`   |
| `std`            | `float`   | Standard deviation of the noise.           | `3.0`   |

### Example
Here is an example of how to run the script:
```bash
python generate_vessel_data.py 10 1.0 1.0 15 0.0 3.0
```

## Output
The generated dataset is saved in the directory `Data_Folder/Synthatic_Data_Sets/Synthatic_Vol_<Bif_Number>` and includes the following files:
- `Volume_Syn_<Bif_Number>.npy`: 3D volume (with and without noise).
- `Vessel_Centerline_<Bif_Number>.npy`: Vessel centerline.
- `Vessel_Radius_Filled_<Bif_Number>.npy`: Vessel radius.

## Visulaiztion 

```bash
conda create -n paraview_env -c conda-forge paraview
conda activate paraview_env
```


## Main Functions

### 1. **`Rot_Mat_from_Rot_Axis_py(Rot_vec, angle)`**
   - Computes a rotation matrix for a given axis and angle using Rodrigues' rotation formula.

### 2. **`Rx(theta), Ry(theta), Rz(theta)`**
   - Generate standard rotation matrices about the X, Y, and Z axes, respectively.

### 3. **`Synthatic_Data_Random_Walk(...)`**
   - The core function for generating the vessel structure.
   - **Inputs**:
     - `Bifurcation_Num`: Number of bifurcations in the vessel.
     - `Init_Dir`: Initial direction of the vessel as a vector.
     - `D_33`: Vessel length scale factor.
     - `D_44`: Orientation randomness scale factor.
     - `N`: Number of points per segment.
     - `mean`, `std`: Parameters for adding noise to the vessel volume.
   - **Outputs**:
     - `Volume_returned`: 3D binary array representing the vessel structure.
     - `Volume_noisy`: 3D array with added noise to simulate realistic data.
     - `Synthatic_Vessel_out`: Vessel centerline coordinates.
     - `Synthatic_Vessel_Filled`: Coordinates representing the filled vessel structure.

---

## Process Overview

### **1. Initial Setup**
- Initializes vessel arrays and assigns the initial direction and positions.
- Sets up rotation matrices and scales.

### **2. Random Walk for Vessel Generation**
- Each segment of the vessel is generated by iteratively updating the position and direction:
  - Updates the position based on the last point and direction vector.
  - Updates the direction using random Euler rotations and cross-product transformations.
- Bifurcation points are introduced, and new branches are generated with random but constrained angles.

### **3. Filled Vessel Generation**
- For each point on the vessel centerline, generates a radial cross-section using null space computation to create orthogonal vectors and rotates them to form a cylindrical shape.

### **4. Volume Representation**
- Converts the vessel structure into a voxel-based representation.
- A noisy version of the volume is generated using Gaussian noise and random binary noise.

---




## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve functionality, add features, or enhance performance.

If you use this code in your research, please cite the original paper:

@article{Yeung:2024,
  title={3D orientation field transform},
  author={Yeung, W. C., Xiaohao L., Zizhen K., Byung-Ho},
  journal={Pattern Analysis and Applications},
  year={2024},
  volume={27}
}
