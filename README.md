# PyTorch Beginner Project Template

Welcome to the PyTorch Beginner Project Template! This repository is designed to help newcomers to PyTorch quickly get started with building, training, and testing machine learning models. It provides a skeletal template that includes essential scripts for training and testing, making it easier for you to dive into your first project.

## Features

- **Training Script:** A ready-to-use script to train your models.
- **Testing Script:** Easily evaluate your model's performance.
- **Modular Structure:** Clean and organized code structure to facilitate learning and development.
- **Customization:** Simple to customize for different types of machine learning tasks.

Whether you're just starting out or looking to streamline your workflow, this template will provide a solid foundation to build upon. Happy coding!

## Project directory structure is as follows:

```{md}
pytorch-project-skeleton
├── logs
├── models
├── README.md
├── LICENSE
├── setup.py
└── source
    ├── logger.py
    ├── metrics.py
    ├── models.py
    ├── test.py
    ├── train.py
    └── utils.py
```

## Setup

Requirements:

- [PyTorch 2.2.0](https://pytorch.org/)
- [numpy](https://github.com/numpy/numpy)
- [tqdm](https://github.com/tqdm/tqdm) 

Setting up the imports:

```{bash}
cd pytorch-project-skeleton
python setup.py develop
```

For training:
```{bash}
cd source
python train.py
```

For testing:
```{bash}
python test.py --exp_name exp_001
```