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
