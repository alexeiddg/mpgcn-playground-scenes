## **Workplan Plan**

### **Replicate the Paper (3 weeks)**
**Goal:** Understand MP-GCN and its _feeder_ before applying it to our data.
- Read the **paper** (graph idea, 4 streams) and the **repo README**.
- Clone the repo, create the environment, and **install** dependencies.
- **Prepare** a public dataset (Volleyball / NBA; optionally **Kinetics-400** via pyskl) as the repo describes.
- **Train or run inference** using the repo’s _config_ and **report** Top-1/confusion matrix.
- **Inspect the tensor shapes** from the _feeder_ ([C,T,V',M]) and print them in a notebook.

**Deliverables**
- Notebook “hello_mpgcn.ipynb” (installation + short training/inference + shape inspection)

---

### **Playground Pipeline (3 weeks)**
**Goal:** Build the **panoramic inputs** from our videos.
- **Filter videos** (ROI) using the database notebook (PostGIS) and **select ≥100 scenes**:
    video_id, camera, t_start, t_end, blob_path
    You can **pre-filter** with the **VLM** and/or by the **# of detections** in the database.
- **Skeletons + light tracking** (YOLO/RTM-pose + ByteTrack/DeepSORT) in those windows; **normalize** per person (hip to origin, scale by torso).
    Save per window: **[T, K_max, 17, 2]**.
- **Objects per camera:** manually annotate **centroids** (0..1) for swings/hills in configs/objects.yaml.
- **Panoramic graph:**
    - Expand **V → V' = 17 + n_obj** (replicate centroids per frame/person)
    - Add **obj↔hands** (intra) and **pelvis↔pelvis** (inter) edges
    - Generate **J/B/JM/BM** and matrices **A0/A_intra/A_inter**
    - **Test a forward pass** with the repo’s feeder (small batch)
**Deliverables**
- data/videos.csv (≥100 rows)
- data/npy/*.npy or .npz per window ([T,K_max,17,2] normalized)
- configs/objects.yaml (centroids per camera)
- Script/function **build_panoramic_graph** and proof of **successful forward pass**

---
### **Labeling, Training, and Results (4 weeks)**
**Goal:** Adapt MP-GCN to playground data and demonstrate the value of the person–object graph.
- **Automatic labeling with VLM** → _class scores_ → **argmax**; store conf_weight = score_max and apply **CORE-CLEAN** (e.g., score ≥ 0.8).
    _(If not using VLM: do light human curation for 3–5 key classes.)_
- **Light training:** freeze most of the backbone and train heads/layers.

**Deliverables**
- train.csv / val.csv (single-label; if VLM: include conf_weight)
- Checkpoint + **metrics**
- **Report** (2–4 pages)

---