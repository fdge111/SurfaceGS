# SurfaceGS: Dynamic Surface Gaussian Splatting for Urban Driving Scenes

## 🚀 Get started
### Environment
```bash
# Clone the repo.
git clone https://github.com/AutoLab-SAI-SJTU/SurfaceGS.git
cd SurfaceGS

# Make a conda environment.
conda create --name surfacegs python=3.10
conda activate surfacegs

# Install requirements.
pip install -r requirements.txt

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

### Data preparation

Create a directory for the data: `mkdir dataset`. Prepare the processed data locally using the instructions below.

<details> <summary>Prepare Waymo Open Dataset.</summary>

We provide the split file following [EmerNeRF](https://github.com/NVlabs/EmerNeRF). You can refer to this [document](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) for download details.

#### Preprocess the data

Preprocess the example scenes

```bash
python script/waymo/waymo_converter.py --root_dir TRAINING_SET_DIR --save_dir SAVE_DIR --split_file script/waymo/waymo_splits/demo.txt --segment_file script/waymo/waymo_splits/segment_list_train.txt
```

Generating LiDAR depth


```bash
python script/waymo/generate_lidar_depth.py --datadir DATA_DIR
```

Generating sky mask

Install GroundingDINO following this [repo](https://github.com/IDEA-Research/GroundingDINO) and download SAM checkpoint from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

```bash
python script/waymo/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CHECKPOINT
```

Generating intance segmentation
```bash
git clone https://github.com/xiao10ma/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
```
follow the instruction in the repo to install the dependencies.

Run the following command to generate the instance segmentation.
```bash
bash waymo_run.sh
```
</details>

### Training

## Novel View Synthesis
```
bash run/run_waymo_nvs.sh
```

## Reconstruction
```
bash run/run_waymo_reconstruction.sh
```


## 📜 BibTeX

``` bibtex
@inproceedings{ge2026surfacegs,
  title={SurfaceGS: Dynamic Surface Gaussian Splatting for Urban Driving Scenes},
  author={Ge, Fudong and Liu, Dingning and Wang, Hanshi and Zhang, Yiwei and Gao, Jin and Hu, Weiming and Zhang, Zhipeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={900--909},
  year={2026}
}
}
```