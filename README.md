**选自towardsdatascience**(README.md)

**作者：Mercy Markus**

**机器之心编辑部**

背靠 fast.ai 库与视频教程，首战 Kaggle 竞赛，新手在油棕种植园卫星图像识别中获得第三名。

背靠 fast.ai 库与视频教程，首战 Kaggle 竞赛，新手在油棕种植园卫星图像识别中获得第三名。

Women in Data Science 与合作伙伴共同发起了 WiDS 数据马拉松竞赛（WiDS datathon）。赛题是创建一个能够预测卫星图像上油棕种植园存在情况的模型。

Planet 和 Figure Eight 慷慨地提供了卫星图像的带注释数据集，而该卫星图像是最近由 Planet 卫星拍摄的。该数据集图像具有 3 米的空间分辨率，并且每一副图像都基于是否含油棕种植园来标记（0 意为没有油棕种植园，1 意为存在一个油棕种植园）。

竞赛任务是训练一个模型，该模型能够输入卫星图像，输出包含油棕种植园图像的似然预测。在模型开发中，竞赛举办者提供带标签的训练和测试数据集。

详细信息参见：[https://www.kaggle.com/c/widsdatathon2019](https://www.kaggle.com/c/widsdatathon2019)<br>

我和队友（Abdishakur、Halimah 和 Ifeoma Okoh）使用 fast.ai 框架来迎接这项赛题。非常感谢 Thomas Capelle 在 kaggle 上的启动内核，该内核为解决这个问题提供了诸多见解。

此外，还要感谢 fast.ai 团队，该团队创建了一门不可思议的深度学习课程，该课程简化了大量困难的深度学习概念。现在，深度学习初学者也可以赢得 kaggle 竞赛了。

课程地址：[https://course.fast.ai/](https://course.fast.ai/)<br>

从一个通俗易懂的深度学习指南开始

不要想着马上就能理解所有东西，这需要大量的练习。本指南旨在向深度学习初学者展示 fast.ai 的魅力。假定你了解一些 python 知识，也对机器学习稍有涉猎。这样的话，我们就走上了学习正轨。

（引用）本文展示的所有代码可在 Google Colaboratory 中找到：这是一个 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行。你可以通过 Colaboratory 编写和执行代码，保存和分享分析，访问大量的计算资源，所有这些都是免费的。

代码参见：[https://colab.research.google.com/drive/1PVaRPY1XZuPLtm01V2XxIWqhLrz3_rgX](https://colab.research.google.com/drive/1PVaRPY1XZuPLtm01V2XxIWqhLrz3_rgX)。<br>

导入 fast.ai 和我们将要使用的其他库

![image](https://github.com/11018339/113-2/blob/main/images/1.png)
