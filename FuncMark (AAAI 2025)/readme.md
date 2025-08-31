# Usage

## Environment

Python Environment
```
conda env create -f environment.yml
conda activate test
cd backend
pip install .
```
## File Structure

```
dataset/data: ground truth sampling and values for F_\Theta
dataset/sdf: F_\Theta
dataset/wm-32: G_\Theta
```


## Data Preparation

Download [ShapeNetv2](https://shapenet.org/download/shapenetcore) and use [mesh-fusion](https://github.com/davidstutz/mesh-fusion) to create watertight meshes.

The finished watertight dataset should be in the following structure
```
| [root to shapenet_watertight]
---| 02691156
------| watertight
---------| *.off
---| 02828884
------| watertight
---------| *.off
```

After creating the watertight mesh
```
cd dataset
batch scripts.sh
```

## Create $F_\Theta$

```
python create_sdf.py --config config/create_sdf.yaml
```

Visualize created $F_\Theta$
```
python create_sdf.py --config config/create_sdf.yaml --debug
```

## Create $G_\Theta$

```
python watermark_sdf.py --config config/watermark_sdf.yaml
```

## Experiment reproduction

Run the following to visual the difference.
```
python watermark_sdf.py --config config/watermark_sdf.yaml --debug
```

```
# evaluation_metrics()                  <-------- uncomment this to reproduce the metrics in papers
# plot(sdf, visual_normal=False,        <-------- uncomment this to visual the G_\Theta with normal
#     #  screenshot="tmp.png"           
#      )                                
# plot_acc(sdf_wm)                      <--------
# learning_rotations(sdf_wm)            <-------- uncommnet this to reproduce affine robustness
# robustness(sdf_wm)                    <-------- uncommnet this to reproduce gauss robustness
# _save_wm_mesh()                       
# _decode_on_mesh()                     <-------- uncommnet this to reproduce affine robustness
view_diff(sdf_wm)                       <-------- uncommnet this to visualize difference
# plot_local(sdf_wm)
# decimation(sdf_wm)                    <-------- uncommnet this to reproduce decimate robustness
# smoothing(sdf_wm)                     <-------- uncommnet this to reproduce smooth robustness
# quantization(sdf_wm)                  <-------- uncommnet this to reproduce quantization robustness
# ablation()
```