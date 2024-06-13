# time_domain_maxwell
AI学习框架与科学计算大作业
# Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks

## 1. 论文介绍

### 1.1 背景介绍

偏微分方程（PDE）在科学和工程中非常重要，广泛用于描述物理问题，如声波传播、流体流动、电磁场等。传统数值方法（如有限差分法和有限元法）虽然被广泛应用，但随着问题规模和复杂性的增加，这些方法的计算成本变得非常高。

近年来，深度学习技术被用于解决PDE，其中物理信息神经网络（PINNs）因其在解决正问题和反问题中的潜力而受到关注。PINNs利用神经网络的通用逼近能力和深度学习框架中的自动微分技术来解决PDE。相比传统的PDE求解器，PINNs不仅可以处理参数化模拟，还能解决传统求解器无法处理的逆问题或数据同化问题。然而，当PDE包含点源（用Dirac delta函数表示）时，常规PINNs方法由于Dirac delta函数的奇异性问题无法收敛。

### 1.2 论文方法

**《Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks》** 提出了使用物理信息神经网络解决含点源PDE的方法。该方法提出了三项关键技术，解决了Dirac delta函数带来的奇异性问题。

**本文的主要方法和优势：**
- **平滑Dirac delta函数**：将Dirac delta函数建模为连续的概率密度函数，以消除其在PINNs训练中的奇异性问题。
- **不确定性加权算法**：提出了一种下界约束的不确定性加权算法，用于在点源区域和其他区域之间平衡PINNs的损失，保证在点源区域的损失不会被忽略，同时避免其他区域的损失被过度放大。
- **多尺度神经网络**：采用具有周期激活函数的多尺度深度神经网络（DNN），提高了PINNs方法的准确性和收敛速度。这种网络结构可以更有效地捕捉多尺度信息，提高解的精度。

**实验与结果**：
- 论文通过三个代表性PDE（包括但不限于Navier-Stokes方程、Schrödinger方程和Maxwell方程）进行了广泛的实验。实验结果表明，与现有的深度学习方法相比，所提出的方法在准确性、效率和通用性方面表现更佳。


### 1.3 结论

通过详细分析和实验验证，这篇论文展示了如何利用物理信息神经网络来有效解决传统方法无法处理的具有奇异性的PDE问题，为相关领域的研究提供了新的方向和方法。

## 2. 点源时域麦克斯韦方程AI求解

人工智能技术的蓬勃发展为科学计算提供了新的范式。MindSpore Elec套件提供了物理驱动和数据驱动的AI方法。物理驱动的AI方法结合物理方程和初边界条件进行模型的训练，相比于数据驱动而言，其优势在于无需监督数据。本课题重点实现物理驱动的AI方法求解点源时域麦克斯韦方程。


### 2.1 问题描述

案例模拟二维的TE波在矩形域的电磁场分布，高斯激励源位于矩形域的中心。该问题的控制方程以及初始和边界条件如下图所示:

![](https://github.com/ouyangzehong/time_domain_maxwell/blob/main/images/formulate3.png)

MindSpore Elec求解该问题的具体流程如下：

* 对求解域以及初边值条件进行随机采样，创建训练数据集。

* 定义控制方程以及定解条件，建立数据集与约束条件的映射关系。

* 构建神经网络。

* 网络训练与推理。

### 2.2 麦克斯韦方程组

有源麦克斯韦方程是电磁仿真的经典控制方程，它是一组描述电场、磁场与电荷密度、电流密度之间关系的偏微分方程组，具体形式如下：

![](https://github.com/ouyangzehong/time_domain_maxwell/blob/main/images/fomulate1.png)

其中，ε 和 μ 分别是介质的绝对介电常数和绝对磁导率。激励源 J 通常表现为端口脉冲的形式，这在数学意义上近似为Dirac函数形式所表示的点源，可以表示为：

![](https://github.com/ouyangzehong/time_domain_maxwell/blob/main/images/formulate2.png)

其中，x₀ 为激励源位置，g(t) 为脉冲信号的函数表达形式。
由于点源的空间分布是非连续的函数，使得源附近的物理场具有趋于无穷大的梯度。另外一个方面，激励源通常是多种频率信号的叠加。已有的基于物理信息神经网络的AI方法求解这种多尺度和奇异性问题通常无法收敛。在MindSpore Elec中，通过高斯分布函数平滑、多通道残差网络结合sin激活函数的网络结构以及自适应加权的多任务学习策略，使得针对该类问题的求解在精度和性能方面均明显优于其他框架及方法。

### 2.3 AI求解点源麦克斯韦方程组

AI求解点源麦克斯韦方程组的整体网络架构如下：

![](https://github.com/ouyangzehong/time_domain_maxwell/blob/main/images/image4.png)

以二维点源麦克斯韦方程组为例，网络输入为 `Ω = (x, y, t) ∈ [0,1]^3`，输出为方程的解 `u = (E_x, E_y, H_z)`。基于网络的输出和MindSpore框架的自动微分功能，训练损失函数来自于控制方程（PDE loss）、初始条件（IC loss）和边界条件（BC loss）三部分。这里我们采用电磁场为0的初始值，边界采用二阶Mur吸收边界条件。由于激励源的存在，我们将PDE loss的计算分为两部分：激励源附近区域 `Ω0` 与非激励源区域 `Ω1`。最终整体的损失函数可以表示为：

Ltotal=λsrcLsrc+λsrcicLsrcic+λnosrcLnosrc+λnosrcicLnosrcic+λbcLbc

其中λs表示各项损失函数的权重。为了降低权重选择的难度，采用了自适应加权的算法，具体参见论文。

### 2.4 数据集

* 训练数据：基于五个损失函数，分别对有源区域，无源区域，边界，初始时刻进行随机采点，作为网络的输入。
* 评估数据：基于传统的时域有限差分算法生成高精度的电磁场。

## 3 环境配置

### 3.1 MindSpore Elec框架搭建

* 版本：2.3.0-rc2
* 硬件平台：Ascend
* 操作系统：Linux-aarch64
* 编程语言：Python 3.7
  
安装命令：
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0rc2/MindSpore/unified/aarch64/mindspore-2.3.0rc2-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

采用自动安装的方式：
在使用自动安装脚本之前，需要确保系统正确安装了昇腾AI处理器配套软件包。

```bash
wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-pip.sh
# 安装MindSpore 2.3.0rc2和Python 3.7
# 默认LOCAL_ASCEND路径为/usr/local/Ascend
MINDSPORE_VERSION=2.3.0rc2 bash -i ./euleros-ascend-pip.sh
# 如需指定Python和MindSpore版本，以Python 3.9和MindSpore 1.6.0为例
# 且指定LOCAL_ASCEND路径为/home/xxx/Ascend，使用以下方式
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash -i ./euleros-ascend-pip.sh
```

在脚本执行完成后，需要重新打开终端窗口以使环境变量生效。

自动安装脚本会为MindSpore创建名为mindspore_pyXX的虚拟环境。其中XX为Python版本，如Python 3.7则虚拟环境名为mindspore_py37。执行以下命令查看所有虚拟环境。

```bash
conda env list
```
执行以下命令激活虚拟环境。

```bash
conda activate mindspore_py37
```

配置环境变量:

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

### 3.2 脚本说明

脚本及样例代码

```bash
.
└─Maxwell
  ├─README.md
  ├─docs                              # README示意图
  ├─src
    ├──dataset.py                     # 数据集配置
    ├──maxwell.py                     # 点源麦克斯韦方程定义
    ├──lr_scheduler.py                # 学习率下降方式
    ├──callback.py                    # 回调函数
    ├──sampling_config.py             # 随机采样数据集的参数配置文件
    ├──utils.py                       # 功能函数
  ├──train.py                         # 训练脚本
  ├──eval.py                          # 推理和评估脚本
  ├──config.json                      # 训练参数和评估数据参数
```

### 3.3 脚本参数

```bash
src_sampling_config = edict({         # 有源区域的采样配置
    'domain': edict({                 # 内部点空间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform'          # 随机采样方式
    }),
    'IC': edict({                     # 初始条件样本采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
    'time': edict({                   # 时间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
})

no_src_sampling_config = edict({      # 无源区域的采样配置
    'domain': edict({                 # 内部点空间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform'          # 随机采样方式
    }),
    'IC': edict({                     # 初始条件样本采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
    'time': edict({                   # 时间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
})

bc_sampling_config = edict({          # 边界区域的采样配置
    'BC': edict({                     # 边界点空间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
        'with_normal': False          # 是否需要边界法向向量
    }),
    'time': edict({                   # 时间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 65536,                # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
})
```
模型训练及控制参数在config.json文件中配置如下：

```bash
{
    "Description" : [ "PINNs for solve Maxwell's equations" ], # 案例描述
    "Case" : "2D_Mur_Src_Gauss_Mscale_MTL",                    # 案例标记
    "random_sampling" : true,                                  # 样本集是否通过随机采样生成，如果为false则加载离线生成的数据集
    "coord_min" : [0.0, 0.0],                                  # 矩形计算域x和y轴最小坐标
    "coord_max" : [1.0, 1.0],                                  # 矩形计算域x和y轴最大坐标
    "src_pos" : [0.4975, 0.4975],                              # 点源位置坐标
    "src_frq": 1e+9,                                           # 激励源主频率
    "range_t" : 4e-9,                                          # 模拟时长
    "input_scale": [1.0, 1.0, 2.5e+8],                         # 网络输入坐标的缩放系数
    "output_scale": [37.67303, 37.67303, 0.1],                 # 网络输出物理量的缩放系数
    "src_radius": 0.01,                                        # 高斯平滑后的点源范围半径大小
    "input_size" : 3,                                          # 网络输入维度
    "output_size" : 3,                                         # 网络输出维度
    "residual" : true,                                         # 网络结构是否包含残差模块
    "num_scales" : 4,                                          # 多尺度网络的子网数目
    "layers" : 7,                                              # 全连接网络层数(输入、输出加隐藏层)
    "neurons" : 64,                                            # 隐层神经元个数
    "amp_factor" : 10,                                         # 网络输入的放大因子
    "scale_factor" : 2,                                        # 多尺度网络的各子网放大系数
    "save_ckpt" : true,                                        # 训练中是否保存checkpoint信息
    "load_ckpt" : false,                                       # 是否加载权重进行增量训练
    "train_with_eval": false                                   # 是否边训练边推理
    "save_ckpt_path" : "./ckpt",                               # checkpoint保存路径
    "load_ckpt_path" : "",                                     # 加载checkpoint的文件路径
    "train_data_path" : "",                                    # 加载离线训练数据集的路径
    "test_data_path" : "",                                     # 加载离线测试数据集的路径
    "lr" : 0.002,                                              # 初始学习率
    "milestones" : [2000, 4000, 5000],                         # 学习率衰减的里程碑
    "lr_gamma" : 0.1,                                          # 学习率衰减系数
    "train_epoch" : 6000,                                      # 迭代训练数据集的次数
    "train_batch_size" : 8192,                                 # 网络训练的批数据大小
    "test_batch_size" : 32768,                                 # 网络推理的批数据大小
    "predict_interval" : 150,                                  # 边训练边推理的迭代间隔步数
    "vision_path" : "./vision",                                # 可视化结果保存路径
    "summary_path" : "./summary"                               # mindinsight summary结果保存路径
}
```

## 4.实验结果

python train.py

```bash
epoch: 1 step: 8, loss is 11.496931
epoch time: 185.432 s, per step time: 23178.955 ms
epoch: 2 step: 8, loss is 9.000967
epoch time: 0.511 s, per step time: 63.926 ms
epoch: 3 step: 8, loss is 8.101629
epoch time: 0.490 s, per step time: 61.248 ms
epoch: 4 step: 8, loss is 7.4107575
epoch time: 0.490 s, per step time: 61.230 ms
epoch: 5 step: 8, loss is 7.0657954
epoch time: 0.484 s, per step time: 60.477 ms
epoch: 6 step: 8, loss is 6.894913
epoch time: 0.482 s, per step time: 60.239 ms
epoch: 7 step: 8, loss is 6.6508193
epoch time: 0.482 s, per step time: 60.297 ms
epoch: 8 step: 8, loss is 6.316092
epoch time: 0.483 s, per step time: 60.343 ms
epoch: 9 step: 8, loss is 6.264338
epoch time: 0.484 s, per step time: 60.463 ms
epoch: 10 step: 8, loss is 6.113656
epoch time: 0.483 s, per step time: 60.392 ms
...
epoch: 5990 step: 8, loss is 0.7306183
epoch time: 0.485 s, per step time: 60.684 ms
epoch: 5991 step: 8, loss is 0.7217314
epoch time: 0.484 s, per step time: 60.559 ms
epoch: 5992 step: 8, loss is 0.7106861
epoch time: 0.483 s, per step time: 60.399 ms
epoch: 5993 step: 8, loss is 0.7238727
epoch time: 0.484 s, per step time: 60.520 ms
epoch: 5994 step: 8, loss is 0.72685266
epoch time: 0.486 s, per step time: 60.735 ms
epoch: 5995 step: 8, loss is 0.7518991
epoch time: 0.485 s, per step time: 60.613 ms
epoch: 5996 step: 8, loss is 0.7451218
epoch time: 0.482 s, per step time: 60.308 ms
epoch: 5997 step: 8, loss is 0.74497545
epoch time: 0.483 s, per step time: 60.313 ms
epoch: 5998 step: 8, loss is 0.72911096
epoch time: 0.483 s, per step time: 60.425 ms
epoch: 5999 step: 8, loss is 0.7317751
epoch time: 0.485 s, per step time: 60.591 ms
epoch: 6000 step: 8, loss is 0.71511096
epoch time: 0.485 s, per step time: 60.580 ms
==========================================================================================
l2_error, Ex:  0.03556711707787814 , Ey:  0.03434167989333677 , Hz:  0.022974221345851673
==========================================================================================
```

在网页打开收集的summary文件，随着训练的进行，Ex/Ey/Hz的误差曲线如下图所示：

```bash
```








