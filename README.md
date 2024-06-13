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




