# **Challenge — Playground Scenes Using MP-GCN**

---

## **What We’ll Do and Why**

We’ll classify **one label per scene** in playground scenes (Transit, Social_People, Play_Object_Normal, Play_Object_Risk, Adult_Assisting, Negative_Contact) using **2D skeletons** and a **person–object panoramic graph** (**MP-GCN**).

MP-GCN models **interactions**: _intra-person_ (body topology), _person↔object_ (hands↔swing/hill), and _inter-person_ (pelvis↔pelvis). It’s **lightweight**, **privacy-friendly**, and captures **risk/furniture use** better than a per-person attention model.
- **Reference repo:** [MP-GCN](https://github.com/mgiant/MP-GCN)
- **Paper:** [“Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph”](https://link.springer.com/chapter/10.1007/978-3-031-73202-7_15)

---

## **Practical Rules**

- **Processing FPS:** 12 → **T** ≈ 60 (sample at **T=48** for the model)
- **Max people per window (Kₘₐₓ):** 4
- **Input shape (ST-GCN/MP-GCN-style feeder):** X ∈ [C, T, V', M]
    - V' = 17 + n_obj (human joints + **object centroids** per camera)
    - **Streams:** J (joints), B (bones), JM=ΔJ, BM=ΔB
    - **Adjacencies:** A0 (self), A_intra (human + **obj↔hands**), A_inter (**pelvis↔pelvis**)

---