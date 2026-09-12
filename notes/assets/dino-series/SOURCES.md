# DINO series figure sources

The paired notes were written by Brench on September 12, 2026. Original diagrams are provided as editable SVGs in Chinese (`-zh.svg`) and English (`-en.svg`). Paper figures retain their original labels; attribution also appears beside each figure in the notes.

| Local figure | Original source |
|---|---|
| `dino-attention-figure-1.png` | Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), Figure 1 |
| `dino-training-figure-2.png` | Same paper, Figure 2 |
| `dinov2-pca-figure-1.png` | Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), Figure 1 |
| `dinov2-data-figure-3.png` | Same paper, Figure 3 |
| `registers-architecture-figure-6.png` | Darcet et al., [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), Figure 6 |
| `dinov3-dense-degradation-figure-5.png` | Siméoni et al., [DINOv3](https://arxiv.org/abs/2508.10104), Figure 5 |
| `dinov3-gram-ablation-figure-9.png` | Same paper, Figure 9 |
| `dinov3-gram-features-figure-10.png` | Same paper, Figure 10 |

The eight original SVG pairs explain series evolution, token outputs, teacher/student training, DINOv2 objectives, DINOv3 stages, Gram alignment, SimCLR pairs, and retrieval. They are teaching diagrams rather than reproductions of a full distributed training implementation. The SimCLR and retrieval diagrams follow the project at [cab5828c7c80d59f2b3d7b02c3e04131b7daa290](https://github.com/BrenchCC/dinov2_with_simclr/tree/cab5828c7c80d59f2b3d7b02c3e04131b7daa290). Solid arrows carry data or an explicitly labeled parameter update; dashed arrows carry detached teacher targets.
