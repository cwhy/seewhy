#import "/template.typ": *

= 任务切换与低效用单元 <sec-switch>

== 适应时间

任务从 $A$ 切到 $B$。
以 $A$ 的拉普拉斯后验为先验，新势能在 $theta_A$ 附近展开为

$ U_B (theta) approx U_B (theta_A) + g_B^top (theta - theta_A) + 1/2 (theta - theta_A)^top H_B (theta - theta_A), $

其中 $H_B$ 已包含旧任务的精度 $Lambda_A$。原文给出三种动力学的适应时间：

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    [动力学], [适应时间], [可塑性 $cal(P) tilde 1 \/ tau$],
    [SGD], [$1 \/ lambda_min (H_B)$], [$lambda_min (H_B)$],
    [朗之万], [$max(1 \/ lambda_min (H_B), thin e^(Delta U_B \/ T))$], [$min(lambda_min (H_B), thin e^(-Delta U_B \/ T))$],
    [Jump-CBP], [$1 \/ (lambda overline(alpha)_B (1 - pi_A)) + tau_"SGD" (xi)$], [$lambda overline(alpha)_B (1 - pi_A)$],
  ),
  caption: [三种动力学的适应时间（原文第 4 节；跳跃一行按 @sec-jump 的逃逸修正乘上 $1 - pi_A$）。时间单位为连续时间；换成步数要再除以学习率 $eta$。],
  kind: table,
)

作为数量级估计，这张表成立。补三点。

+ *平坦方向未必有害。* $lambda_min (H_B) approx 0$ 的方向，只有在 $g_B$ 有分量时才拖慢适应。
  而那时二次展开已经失效（势能沿该方向线性下降、无下界），需要更高阶项。
+ *平台不是势垒。* #added 死单元处 $g_B = 0$ 且 $H_B$ 在该方向为零。
  朗之万的时间应是 @sec-langevin 的扩散时间 $L^2 \/ (2 T)$，而 SGD 的时间是无穷。
+ *跳跃一行依赖 $overline(alpha)_B$ 在哪里取平均。* 它是被判为低效用的单元上的速率。
  对有用单元，CBP 不提供额外可塑性，也不应提供。

== 低效用的贝叶斯定义

原文定义：*低效用单元 = 对后验预测分布贡献可忽略的单元*，并列出四种“等价”刻画。

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    [名称], [定义], [含义],
    [信息增益], [$u_i^"info" = D_"KL" (q(theta_i | D) || p(theta_i))$], [后验离先验多远],
    [Fisher], [$u_i^"F" = F_(i i)$], [数据对该权重的约束强度],
    [预测贡献], [$u_i^"pred" = D_"KL" (p(y | x, D) || p_(-i)(y | x, D))$], [去掉该权重，预测变多少],
    [信噪比], [$u_i^"SNR" = mu_i^2 \/ sigma_i^2$], [后验均值是否显著不为零],
  ),
  caption: [原文的四种低效用度量。$p_(-i)$ 表示把第 $i$ 个权重（或单元）移除后的预测分布。],
  kind: table,
)

#fix 四者*不等价*。它们只在一个极端上一致：完全没被使用的权重。
此时 $F_(i i) = 0$，后验等于先验，四个量同时取最小值（信噪比在先验均值为零时为 0）。
离开这个极端就会分歧。例如平均场高斯后验 $cal(N)(mu_i, s_i^2)$、先验 $cal(N)(0, sigma^2)$ 时，

$ u_i^"info" = 1/2 [s_i^2 / sigma^2 + mu_i^2 / sigma^2 - 1 - log s_i^2 / sigma^2]. $

考虑一个均值为零、方差被数据压得很小的权重。
它的信噪比为 0；移除它等于把它置零，所以预测贡献也近 0。
但信息增益很高：数据确实约束了它——约束它等于零。
重置它会把它从零附近拉开，于是按信息增益不该重置，按预测贡献可以重置。
四种度量中，只有预测贡献 $u^"pred"$ 直接对应原文的定义；其余是它在特定近似下的替代品。

== 相对于哪个任务

原文提出，切换后应相对新任务 $B$ 衡量效用：

$ hat(u)_i^B approx F_(i i)^B (theta_A) = bb(E)_(D_B) [ (partial_(theta_i) log p(D_B | theta) |_(theta_A))^2 ]. $

#added 这里有一个张力。
只看 $F^B$，对 $A$ 至关重要、对 $B$ 无关的单元会被判为低效用，于是被重置，$A$ 被遗忘。
只看 $F^A$，则无法释放容量给 $B$。
任何实用的 $alpha$ 都必须同时用到两者。
@sec-changepoint 的变点后验自动做到这一点：
旧后验的精度（来自 $A$）与新数据的似然（来自 $B$）同时出现在边际似然之比中。
