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

## Citation

If you use this code in your research, please cite the original paper:

@article{Yeung:2024,
  title={3D orientation field transform},
  author={Yeung, W. C., Xiaohao L., Zizhen K., Byung-Ho},
  journal={Pattern Analysis and Applications},
  year={2024},
  volume={27}
}
