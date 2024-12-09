# KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning

This repository provides the implementation of **KDk**, a novel defense mechanism designed to protect against label inference attacks in vertical federated learning (VFL). The approach employs soft labels to mitigate the risks associated with label inference, ensuring enhanced privacy and security for federated learning models.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{arazzi2024kdk,
  title={KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning},
  author={Arazzi, Marco and Nicolazzo, Serena and Nocera, Antonino},
  journal={arXiv preprint arXiv:2404.12369},
  year={2024}
}
```

The code provided here is intended to be integrated into the repository FuChong-cyber/label-inference-attacks. Specifically, KDk generates soft labels before the vertical federated training begins.