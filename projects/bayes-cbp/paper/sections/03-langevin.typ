#import "/template.typ": *

= 朗之万采样视角 <sec-langevin>

== 动力学

令势能 $U(theta) = -log p(D | theta) - log p(theta)$。
过阻尼朗之万方程为

$ upright(d) theta_t = -nabla U(theta_t) thin upright(d) t + sqrt(2 T) thin upright(d) W_t , $

其中 $W_t$ 是布朗运动，$T$ 是温度。
它的稳态分布是 $p_infinity prop exp(-U \/ T)$；$T = 1$ 时就是后验。
欧拉离散化得到#gloss[SGLD][随机梯度朗之万动力学：小批量梯度步加上方差为 $2 eta T$ 的高斯噪声 @welling2011sgld]：

$ theta_(k+1) = theta_k - eta nabla tilde(U)(theta_k) + sqrt(2 eta T) thin xi_k , quad xi_k tilde cal(N)(0, I). $

== 可塑性的度量

原文列出四个度量。逐条核对如下。

*响应幅度。* 在曲率 $kappa$ 的势阱里加一个梯度 $g$，平衡点移动 $Delta theta approx g \/ kappa$。正确。

*势垒逃逸。* #gloss[Kramers 公式][粒子靠热噪声翻过高度 $Delta U$ 的势垒，平均等待时间随 $Delta U \/ T$ 指数增长 @kramers1940]：
$tau tilde exp(Delta U \/ T)$。正确，省略了与曲率有关的前置因子。

*均方位移。* #fix
原文写 $"MSD"(t) tilde 2 d T t$。这只在平坦势能（自由扩散）下成立。
在曲率 $kappa$ 的二次势阱中，每个维度的方差是

$ "Var"(t) = T / kappa (1 - e^(-2 kappa t)). $

早期（$kappa t << 1$）它等于 $2 T t$；晚期饱和在 $T \/ kappa$。
原文的权衡式 $cal(P) tilde T \/ (kappa_"old" + beta)$ 正是这个饱和值，加上了 KL 惩罚的曲率 $beta$。
所以权衡式成立，但 MSD 式只描述最初的一小段。

== 两种困住方式：势垒与平台 #added

原文只讨论了势垒。
实际的可塑性丧失更常见另一种情形：*平台*。
一个 ReLU 单元对所有输入都输出 0，它的输入、输出权重的梯度都恰好是 0。
这里没有势垒，势能是平的。

三种动力学在平台上的表现完全不同：

- *SGD*：小批量梯度的每个样本分量都是 0，噪声也是 0。粒子*永远*不动。
  原文用常数 $sigma thin upright(d) W_t$ 建模 SGD 噪声，在这里不成立：SGD 噪声依赖状态，在死单元上消失。
- *朗之万*：注入的噪声与梯度无关，粒子做自由扩散。
  离开半宽为 $L$ 的平台，平均需要 $L^2 \/ (2 T)$。这是多项式时间，不是指数时间。
- *跳跃*：以速率 $lambda alpha$ 直接离开，与平台大小无关。

这个区分在 @sec-changepoint 还会出现：平台上的单元恰好是“数据不提供证据”的单元。
