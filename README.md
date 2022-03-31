# PointNet.pytorch

这个仓库 [PointNet](https://arxiv.org/abs/1612.00593) 基于 pytorch 的实现，模型在 `pointnet/model.py` 中.

基于 pytorch-1.0 进行了测试。

# 数据下载与运行

```shell
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

下载和编译可视化工具

```shell
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

在Windows下编译可视化工具：

[Python调用CPP文件的方法](https://blog.csdn.net/qq_38939905/article/details/121961058)

使用 Visual Studio 2019，新建“C++，Windows，Library”中的动态链接库（DLL），在`pch.h`文件中声明外部调用的函数接口

```c++
#ifndef PCH_H
#define PCH_H
 
// 添加要在此处预编译的标头
#include "framework.h"
extern "C" _declspec(dllimport) void render_ball(int h, int w, unsigned char* show, int n, int* xyzs, float* c0, float* c1, float* c2, int r);
#endif //PCH_H
```

将 `render_balls_so.cpp` 中的内容拷贝到 `pch.cpp`，按照平台情况选择解决方案是`x86`还是`x64`，生成解决方案得到DLL文件，将之拷贝到`dll = np.ctypeslib.load_library('./utils/render_balls_so.dll', '.')` 对应的目录下即可。

训练模型

```shell
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

配置 `--feature_transform` 使用特征变换。

# 文件说明

- `pointnet/dataset.py`
  - `ShapeNetDataset`：ShapeNet 数据集类
  - `ModelNetDataset`：ModelNet 数据集类
- `pointnet/model.py`
  - `PointNetCls`：分类器类
  - `PointNetDenseCls`：分割器类
  - `PointNetFeat`：特征生成类
  - `STN3D`：T-Net, Input Transformer, 生成 3x3 的转换矩阵
  - `STNkd`：T-Net, Feature Transformer, 生成 64x64 的转换矩阵
- `utils/show_cls.py`：分类结果可视化
- `utils/show_seg.py`：分割结果可视化
- `utils/train_classification.py`：分类器的训练
- `utils/train_segmentation.py`：分割器的训练

# 性能

## 分类器的性能

数据集 ModelNet40:

|  | 所有精度 |
| :---: | :---: |
| 原始实现 | 89.2 |
| 仓库实现(w/o 特征变换) | 86.4 |
| 仓库实现(w/ 特征变换) | 87.0 |

数据集 [shapenet的子集](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | 所有精度 |
| :---: | :---: |
| 原始实现 | N/A |
| 仓库实现(w/o 特征变换) | 98.1 |
| 仓库实现(w/ 特征变换) | 97.7 |

## 分割器的性能

分割基于  [shapenet的子集](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.6 |  | | | | | | | | | |81.0|

注：这个实现单独训练每个类别，因此数据少的类别的性能比较参考实现要差。

样本分类结果:
![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)

# 链接

- [PointNet 主页](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow 实现](https://github.com/charlesq34/pointnet)
