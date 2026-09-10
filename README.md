## Spatially Enhanced Evolutionary Prompting for Class-Incremental Learning [ICONIP 2026]

[Official PyTorch code]

## Abstract
Prompt-based Continual Learning (PCL) adapts frozen Vision Transformers to sequential tasks with high parameter efficiency. Existing methods suffer from two fundamental limitations: token-only evolution methods discard fine-grained spatial structures, while static spatial prompt pools incur capacity saturation and cross-task interference as the number of tasks grows. We propose Spatially Enhanced Evolutionary Prompting (SEEP), a hierarchical framework that unifies evolutionary adaptation across both token and spatial dimensions. SEEP introduces an Evolutionary Local Prompt mechanism that instantiates a dedicated task-specific generator for each new task while freezing all previous generators, explicitly preventing catastrophic forgetting at the pixel level. A Cumulative Spatial Similarity Combination strategy then dynamically aggregates historical spatial prompts based on local structural alignment with the input, enabling fine-grained knowledge transfer without a fixed-capacity pool. Complemented by a
global frequency branch and a diversity-enhanced token-level prompt evolution, SEEP
eliminates the scalability bottleneck of prior spatial methods while retaining only
$4.2\%$ of their trainable parameters per incremental phase. Extensive experiments
on image classification (CIFAR-100, CUB-200, ImageNet-R) and video action
recognition (UCF-101, ActivityNet) benchmarks demonstrate that SEEP consistently
achieves state-of-the-art performance, improving accuracy by up to $3.36\%$ on
CIFAR-100 and $2.35\%$ on CUB-200 while reducing catastrophic forgetting by
over $50\%$ on long task sequences.