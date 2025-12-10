# [OpenMOA](https://openmoa.net)

![Banner Image]()

[![PyPi Version](https://img.shields.io/pypi/v/openmoa)](https://pypi.org/project/openmoa/)
[![Join the Discord](https://img.shields.io/discord/1235780483845984367?label=Discord)](https://discord.gg/spd2gQJGAb)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://openmoa.net/docs/)
[![GitHub](https://img.shields.io/github/stars/ZW-SIYUAN/OpenMOA?style=social)](https://github.com/ZW-SIYUAN/OpenMOA)

OpenMOA is a unified machine learning library designed for Utilitarian Online Learning (UOL) in dynamic feature spaces.
It offers a clean Python API that integrates seamlessly with MOA (online learners), OpenMOA (stream learning backend), and PyTorch (deep models).
OpenMOA provides an efficient and extensible interface for executing state-of-the-art algorithms under evolving feature spaces, enabling reproducible, real-time learning and fair benchmarking across diverse streaming environments.

To setup OpenMOA, simply install it via pip. If you have any issues with the
installation (like not having Java installed) or if you want GPU support, please
refer to the [installation guide](https://openmoa.net/docs/getting-started/). Once installed take a
look at the [tutorials](https://openmoa.net/docs/guide/tutorials/) to get started.

```bash
# OpenMOA requires Java. This checks if you have it installed
java -version

# OpenMOA requires PyTorch. This installs the CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install OpenMOA and its dependencies
pip install openmoa

# Check that the install worked
python -c "import openmoa; print(openmoa.__version__)"
```

> **⚠️ WARNING**
>
> OpenMOA is still in the early stages of development. The API is subject to
> change until version 1.0.0. If you encounter any issues, please report
> them in [GitHub Issues](https://github.com/ZW-SIYUAN/OpenMOA/issues)
> or talk to us on [Discord](https://discord.gg/spd2gQJGAb).

---

![Benchmark Image]()
Our benchmark evaluates ten representative UOL and OL algorithms across 11 binary and 6 multi-class datasets under three dynamic feature-space paradigms.
Built upon OpenMOA’s unified API, all experiments follow standardized feature-evolution assumptions and prequential evaluation protocols, enabling fair, reproducible, and comprehensive comparison across diverse streaming environments.
You can find the code to reproduce this benchmark in
[`demo/demo_fesl_benchmark_binary.py`](https://github.com/ZW-SIYUAN/OpenMOA/blob/main/demo/demo_fesl_benchmark_binary.py).

## Cite Us

If you use OpenMOA in your research, please cite us using the following BibTeX item.

```
@misc{
    ZhiliWang2025OpenMOAAPythonLibraryforUtilitarianOnlineLearning,
    title={{OpenMOA}: A Python Library for Utilitarian Online Learning},
    author={Zhili Wang, Yi He},
    year={2025},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/},
}
```
