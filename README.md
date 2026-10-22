# SurfaceGS: Dynamic Surface Gaussian Splatting for Urban Driving Scenes

### [[Paper]](https://openaccess.thecvf.com/content/CVPR2026F/html/Ge_SurfaceGS_Dynamic_Surface_Gaussian_Splatting_for_Urban_Driving_Scenes_CVPRF_2026_paper.html)

> [**SurfaceGS: Dynamic Surface Gaussian Splatting for Urban Driving Scenes**](https://openaccess.thecvf.com/content/CVPR2026F/html/Ge_SurfaceGS_Dynamic_Surface_Gaussian_Splatting_for_Urban_Driving_Scenes_CVPRF_2026_paper.html)<br>
> Fudong Ge, Dingning Liu, Hanshi Wang, Yiwei Zhang, Jin Gao, Weiming Hu, Zhipeng Zhang<br>
> **State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), CASIA**<br>
> **AutoLab, School of Artificial Intelligence, Shanghai Jiao Tong University**<br>
> **CVPR 2026 Findings**

**Official implementation of "SurfaceGS: Dynamic Surface Gaussian Splatting for Urban Driving Scenes".**

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

# Install simple-knn.
pip install ./cuda_package/simple-knn

# Install the modified Gaussian splatting rasterizer (for feature rendering).
pip install ./cuda_package/diff-gaussian-rasterization

# Install nvdiffrast (for Envlight).
pip install ./cuda_package/nvdiffrast
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

Replace `TRAINING_SET_DIR` and `SAVE_DIR` with your local directories. Add
`--track_file TRACKING_RESULTS.json` if you have external tracking results.

Generating LiDAR depth


```bash
python script/waymo/generate_lidar_depth.py --datadir DATA_DIR
```

Generating sky mask

Install GroundingDINO following this [repo](https://github.com/IDEA-Research/GroundingDINO) and download SAM checkpoint from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

```bash
python script/waymo/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CHECKPOINT
```

Generating instance segmentation
```bash
git clone https://github.com/xiao10ma/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
```
Follow the instructions in the repo to install the dependencies.

Run the following command to generate the instance segmentation.
```bash
bash waymo_run.sh
```
</details>

### Training

Set `SOURCE_PATH` and `MODEL_PATH` in the run script to your local dataset and
output directories. The base configuration defaults to the repository's
`configs/base.yaml`; use `--base_config` to select a different file.

#### Novel View Synthesis
```bash
bash run/run_waymo_nvs.sh
```

#### Reconstruction
```bash
bash run/run_waymo_reconstruction.sh
```

### Evaluation

Pass your dataset and model directories explicitly. Evaluation uses the latest
checkpoint in `model_path`, or the file supplied through `checkpoint`.

```bash
python evaluate.py --config configs/waymo/nvs/017.yaml source_path=dataset/waymo/017 model_path=output/waymo_nvs/017
```

For monocular normal preprocessing with a separate DSINE checkout:

```bash
python script/waymo/generate_mono_normal.py --datadir dataset/waymo/017 --dsine_dir external/DSINE
```


### Acknowledgments

SurfaceGS uses code from a few open source repositories. Without the efforts of these folks (and their willingness to release their implementations), SurfaceGS would not be possible. We thank these authors for their efforts!

- [EmerNeRF](https://github.com/NVlabs/EmerNeRF)
- [PVG](https://github.com/fudan-zvg/PVG)
- [BezierGS](https://github.com/fudan-zvg/BezierGS)
- [OmniRe](https://github.com/ziyc/drivestudio)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)


## 📜 BibTeX

```bibtex
@inproceedings{ge2026surfacegs,
  title={SurfaceGS: Dynamic Surface Gaussian Splatting for Urban Driving Scenes},
  author={Ge, Fudong and Liu, Dingning and Wang, Hanshi and Zhang, Yiwei and Gao, Jin and Hu, Weiming and Zhang, Zhipeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={900--909},
  year={2026}
}
```
