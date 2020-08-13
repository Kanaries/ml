# 介绍

<img src="https://ch-rath.oss-ap-northeast-1.aliyuncs.com/assets/kanaries-light-bg.png" alt="logo" width="280px" style="" />

`@kanaries/ml` 是一个skit-learn风格的javascript(ts)机器学习库。旨在提供可以被javascript开发者们直接使用的机器学习工具，它可以直接运行在浏览器和node环境中。

`@kanaries/ml`在设计开发上主要在参考skit-learn，根据情况适当做了微调。如果你熟悉skit-learn的API，那你可以很轻松的上手@kanaries/ml


`@kanaries/ml`目前还在开发中，如果你对它有什么建议，欢迎在issues中留言。

#### 开发者正在犹豫的事情
+ 与skit-learn的使用方式相比，由于欠缺一个类似numpy的js工具。这一点目前开发者还在犹豫，是否先提供一个矩阵转化工具，使得内部的所有计算都可以使用typedArray进行，来获取更高的性能。
