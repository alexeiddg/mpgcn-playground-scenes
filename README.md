# **Challenge — Playground Scenes Using MP-GCN**

---

## **What We’ll Do and Why**

We’ll classify **one label per scene** in playground scenes (Transit, Social_People, Play_Object_Normal) using **2D skeletons** and a **person–object panoramic graph** (**MP-GCN**).

MP-GCN models **interactions**: _intra-person_ (body topology), _person↔object_ (hands↔swing/hill), and _inter-person_ (pelvis↔pelvis). It’s **lightweight**, **privacy-friendly**, and captures **risk/furniture use** better than a per-person attention model.
- **Reference repo:** [MP-GCN](https://github.com/mgiant/MP-GCN)
- **Paper:** [“Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph”](https://link.springer.com/chapter/10.1007/978-3-031-73202-7_15)

---
## **Our work**

- **Our Research:** [Playground Scene Understanding using
Multi-Person Graph Convolutional Networks](https://drive.google.com/file/d/1KF8tL09hIDGAQ8455WPxkUGc84YkJJFT/view?usp=sharing)

### **Knowledge bases created**
- **Presentation Slides:** [MP-GCN Playground](https://drive.google.com/file/d/1DcpNr8efTBVO34-lbSOfKgDQe-lTPmn4/view?usp=sharing)
- **Understanding MP-GCN:** [Skeleton-based Group Activity Recognition via
Spatial-Temporal Panoramic Graph](https://drive.google.com/file/d/1evxlVWJxJsMz0F9XLQI75ch02dkHrgRB/view?usp=sharing)
- **Understanding MP-GCN original Codebase:** [Understanding MP-GCN Codebase](https://drive.google.com/file/d/1WQdDZSIegiMXaEMVaLQW78xKmaY5c600/view?usp=sharing)
- **MPGCN Data construction:** [MPGCN Data construction](https://drive.google.com/file/d/1KDFllDOOB_muyGMPiBEuKBzzNfU4_fxn/view?usp=sharing)


### **Data Created**
- **.npy Pose-Only Data:** [NPY Files](https://drive.google.com/file/d/19MZTeHAVXPMMVdW9qbT6t5wmJjXsrKAh/view?usp=sharing)
- **Folds:** [FOLDS](https://drive.google.com/file/d/1L6aLr2j1sGeUcCoFLyacB7zNUH-QbgSh/view?usp=sharing)
- **Late-Fusion Generated Data:** [Late-fusion folds & eval](https://drive.google.com/file/d/1CCm8bu8vv25-GIoYRiAEU7g9ELWWPT6X/view?usp=sharing)
- **Full late-fusion workdir w/folds:** [Full fold data](https://drive.google.com/file/d/1HJlgjiJUWzFp_yAwayiAanozoTq7arhm/view?usp=sharing)
---

# Playground Scenes MP-GCN Customization

## Abstract

This project proposes a lightweight framework for Group Activity Recognition (GAR) in playground environments using the Multi-Person Graph Convolutional Network (MP-GCN) architecture. Unlike traditional RGB-based methods that struggle with occlusions, visual noise, and privacy concerns, MP-GCN relies solely on 2D skeletal keypoints and object centroids to model interactions through a panoramic spatial–temporal graph. This representation captures intra-person body structure, inter-person coordination, and person–object relationships, such as children interacting with swings or slides, allowing the system to classify six scene categories (Transit, Social_People, Play_Object_Normal, Play_Object_Risk, Adult_Assisting, and Negative_Contact) efficiently and ethically. The implementation includes pose estimation and tracking (YOLOv11-pose), graph construction, and training of an adapted MP-GCN pipeline on short multi-person video segments. By emphasizing geometry and motion over appearance, the project aims to deliver a reproducible, real-time activity recognition model that enhances safety and behavioral understanding in public playgrounds while maintaining computational efficiency and privacy protection.

## Repository overview

- `hello_mpgcn/`: vendor copy of MP-GCN kept intact with its `config/`, `src/`, and `main.py` entry for training/evaluation; use this when you need the canonical defaults or to compare against upstream.
- `src/`: playground-specific pipeline that wraps the vendor code. Highlights: `reader/` builds splits and tensors; `dataset/` defines graphs, augmentations, and the feeder; `model/MPGCN/` adds area embeddings; `processor.py` handles training/eval loops; `initializer.py` wires configs, samplers, and loss; `generator.py` is the CLI hook for data generation.
- `config/`: playground configs and metadata. `objects.yaml` holds camera/object centroids; `playground_gendata.yaml` drives tensor generation; `config/playground/*.yaml` are stream-specific train/eval configs (J, JM, B, BM, joint mixes).
- `data/`: raw CSVs, ROI geopackage, processed labels, and generated tensors in `data/npy/`; intermediate YOLO outputs stay under `data/temp/` and are ignored by Git.
- `script/`: utilities (`tensor_builder.py`, `cal_mean_std.py`, `ensemble_playground.py`) plus `script/notebooks/` organized by pipeline stage (`scene-`/`skeleton-`/`tensor-` prefixes per the notebooks README).
- `workdir/`: outputs from data generation and training (per-fold tensors, checkpoints, confusion matrices, notes). CI-like artifacts are kept here rather than under `data/`.
- Root files: `main.py` is the slim CLI entry that calls into `src/`; `automated_fusion_pipeline_v2.ipynb` and `late_fusion_model.ipynb` explore fusion/ensembling outside the main pipeline; `WORKPLAN.md` tracks experiment to-dos.

## Data flow and splits

- Source clips and detections live under `data/` (`videos.csv`/`videos-trimmed.csv`, YOLO JSONs in `temp/`, tensors in `npy/`). Notebook `script/notebooks/tensor-builder.ipynb` converts merged detections into `pose_data.npy` and `_object_data.npy`.
- `Playground_Reader` (`src/reader/playground_reader.py`) normalizes tensors to `[T, O|M, …]`, learns `n_obj_max`, filters unused labels, and writes fixed-shape pose/object/area arrays plus label pickles to `workdir/fold_{ID}`.
- Split strategy: **5×5 repeated stratified K-fold**, seeded for reproducibility, so every fold keeps class balance and per-camera variety. Fold IDs are stable across reruns; `{ID}` placeholders in configs resolve to the right `fold_{ID}` folder.
- People are ranked by motion energy via `select_top_m_people.py` to keep the top 4 movers (zero-padding the rest), aligning with the MP-GCN `M` limit and avoiding shape drift across splits.
- Regenerate tensors whenever labels, centroids, or sampling params change: `python hello_mpgcn/main.py -c config/playground_gendata.yaml -gd`.

## Model I/O at a glance

- Pose tensors come in as `pose_data.npy` shaped `[T, M, V, C]` with `T=48` frames, up to `M=4` people, `V=17` joints, `C=3` coords.
- Object tensors land in `*_object_data.npy` shaped `[T, O, C]` where `O` is per-fold `n_obj_max`; padding/truncation keeps time and objects static across clips.
- At train time, the feeder concatenates object nodes onto the joint dimension so the model sees `[C, T, V+O, M]` (and builds B/JM/BM streams from that).
- Graph adjacency uses COCO joints plus hands↔object links and optional pelvis↔pelvis inter-person links; area IDs thread through as embeddings to undo camera/site bias.

## Core customizations vs. the vendor copy (`hello_mpgcn/`)

- `src/reader/playground_reader.py`: end-to-end tensor builder for playground data. It tolerates both legacy `_object.npy` and new `_object_data.npy` names, normalizes arbitrary object array shapes to `[T, O, C]`, scans the dataset to learn `n_obj_max`, writes fixed-shape pose/object/area tensors per fold, and derives `class2idx`. The repeated stratified splits keep all classes present in every fold.
- `src/dataset/graphs.py`: new `playground` graph type extends COCO with `n_obj_max` object nodes. Each object connects to both hands `(9, obj_i)` and `(10, obj_i)` so the graph can reason about hand–object contact; multi-person variants add pelvis-pair links (configurable base joints) for inter-person context.
- `src/dataset/playground_feeder.py`: dataloader that merges pose and object tensors. It tiles object centroids across persons to fit the ST-GCN layout, supports per-stream selection (`J`, `JM`, `B`, `BM`, or `JVBM`), and enforces windowed shapes after augmentation. Area IDs are either read from `*_area.npy` or inferred from clip names (`columpios` vs. `unknown`) so downstream can compensate for camera bias.
- `src/dataset/augment.py`: augmentation keeps pose/object aligned—temporal jitter/crop/drop/speed-scale plus joint/object jitter, translation, and scaling—to fight overfitting while preserving hand–object geometry.
- `src/model/MPGCN/nets.py`: injects **area embeddings**. After the branch stack, an embedding vector per area is concatenated as extra channels before the main stream, letting the model undo static camera/site bias without hard-coding it into labels.
- `src/dataset/utils.py`: `multi_input` understands aliases (`JOINT`, `JM`, `B`, `BM`, `JVBM`) and builds stacked multi-stream tensors from the augmented joint+object graph. `graph_processing` handles person flattening while respecting the larger vertex count from the playground graph.
- `src/initializer.py`: resolves `{ID}` placeholders so `fold_{ID}` paths point at the correct split, infers `num_object` from saved tensors when not provided, builds **class-weighted losses** plus a **WeightedRandomSampler** to counter class imbalance, and optionally enables hard-example mining in `src/processor.py`.

## Running the pipeline

- Tensor generation: `python hello_mpgcn/main.py -c config/playground_gendata.yaml -gd` writes per-fold pose/object/area/label arrays under `workdir/`.
- Train or fine-tune: `python hello_mpgcn/main.py --config config/playground/mpgcn.yaml --gpus 0`; set `FOLD_ID` or `dataset_args.fold_id` to pick a split. Variants `mpgcn_J.yaml`, `mpgcn_B.yaml`, etc. select individual streams.
- Evaluate a checkpoint: `python hello_mpgcn/main.py --config config/playground/mpgcn.yaml --evaluate`.
- Notebook quick checks: `script/notebooks/tensor-builder.ipynb` for tensor sanity; `hello_mpgcn/hello_mpgcn.ipynb` to instantiate `Playground_Reader` and print `[C, T, V', M]` stats.
