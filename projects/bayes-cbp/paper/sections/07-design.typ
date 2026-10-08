#import "/template.typ": *

= 第一性原理的贝叶斯 CBP <sec-design>

== 原文的分层模型

原文为每个权重设一个分层先验：

$ theta_i tilde cal(N)(mu_i, sigma_i^2), quad mu_i tilde cal(N)(0, tau^2), quad sigma_i^2 tilde "InvGamma"(a, b). $

$mu_i$ 是重置中心，$sigma_i^2$ 控制可塑性。
超参数由#gloss[第二类极大似然][type-II ML / 经验贝叶斯：把参数积掉，选使数据边际概率最大的超参数]更新：

$ (mu^*, sigma^(2*)) = arg max_(mu, sigma^2) log integral p(D_t | theta) thin p(theta | D_(1:t-1), mu, sigma^2) thin upright(d) theta . $

#added 这里有一个与 @sec-bayes 相同的问题。
在静态模型中，$p(theta | D_(1:t-1), mu, sigma^2)$ 随 $t$ 增长被数据主导，对 $(mu, sigma^2)$ 的依赖越来越弱。
边际似然对超参数逐渐不敏感，经验贝叶斯也就越来越“学不动”。
要让超参数在后期仍起作用，模型里必须有让旧数据失效的机制。

== 重置强度 $1 - sigma^2_"old" \/ sigma^2_"new"$ 的问题 <sec-alpha>

原文定义 $alpha_i = 1 - sigma_i^(2("old")) \/ sigma_i^(2("new"))$。
#fix 它有三个问题。

*定义域。* $sigma_"new" < sigma_"old"$ 时 $alpha_i < 0$。至少需要 $max(0, dot)$。

*若 $sigma^2$ 指后验方差。* 静态模型下后验方差只减不增（@sec-bayes）。
于是 $sigma_"new" <= sigma_"old"$ 恒成立，截断后 $alpha_i equiv 0$，永不重置。

*若 $sigma^2$ 指先验超参数。* 对单个权重做第二类极大似然。
设数据对它的约束等价于一个观测 $hat(theta)_i$，方差 $v_i = 1 \/ F_i$。
则 $hat(theta)_i$ 的边际分布是 $cal(N)(mu_i, sigma_i^2 + v_i)$，最优解为

$ sigma_i^(2*) = max(0, thin (hat(theta)_i - mu_i)^2 - v_i). $

这就是#gloss[ARD][自动相关性判定：逐权重学先验方差，无用权重的先验方差被推到零而“剪掉” @tipping2001sparse]的结论。
数据约束弱（$v_i$ 大）的权重，$sigma_i^(2*) -> 0$，按原文公式 $alpha_i <= 0$，*最不会*被重置。
完全不受约束（$F_i = 0$）的死单元，边际似然与 $sigma_i^2$ 无关，$sigma_i^(2*)$ 不可辨识，$alpha_i$ 无定义。
这与原文“低效用单元 $alpha -> 1$”的设想方向相反。

*一个使公式成立的读法。* 若新先验是旧后验加一个独立随机游走，$sigma_"new"^2 = sigma_"old"^2 + q$，
则 $alpha = q \/ (sigma_"old"^2 + q)$，即“新先验方差中新鲜噪声所占的比例”。
但这个预测步保持均值不变，不会把 $theta$ 拉向 $mu^*$；与原文的软重置 $theta <- (1 - alpha) theta + alpha mu^*$ 不配套。

结论：原文的 $alpha$ 不是推出来的，按两种自然读法都不给出所需的行为。
下一节给出一个可以推出来的替代。

== 变点模型：CBP 的精确贝叶斯读法 <sec-changepoint>

#added 本节是全文的主要新增，原文没有。

*模型。* 给每个权重（或每个单元）一个变点先验：每一步，它以概率 $rho$ 被整个换掉，从 $p_0 = cal(N)(mu_i, sigma_i^2)$ 重新抽取；否则保持不变：

$ theta_(i,t) | theta_(i,t-1) tilde (1 - rho) thin delta(theta_(i,t) - theta_(i,t-1)) + rho thin p_0 (theta_(i,t)). $

*预测步 = CBP 跳跃。* 滤波的预测步把旧后验 $p_(t-1)$ 推进一步：

$ tilde(p)_t = (1 - rho) thin p_(t-1) + rho thin p_0 . $

令 $rho = lambda alpha thin Delta t$，得到 $(tilde(p)_t - p_(t-1)) \/ Delta t = lambda alpha (p_0 - p_(t-1))$。
这正是 @sec-jump 中跳跃部分的前向方程。
若 $rho$ 依赖 $theta_(t-1)$，同样的计算给出修正后的方程（流出 $alpha(theta) p$、流入 $overline(alpha) p_0$）。
所以 Jump-CBP 的跳跃项*就是*这个先验下的贝叶斯预测，不是近似。

*更新步 = 重置判决。* 乘上新数据的似然 $L(theta) = p(D_t | theta)$，后验是两个分量的混合：

$ p_t prop (1 - rho) thin L thin p_(t-1) + rho thin L thin p_0 . $

两个分量的权重正比于各自的边际似然
$Z_"stay" = integral L thin p_(t-1)$ 与 $Z_"fresh" = integral L thin p_0$。
“发生了变点”的后验概率为

$ r_i = (rho thin Z_"fresh") / (rho thin Z_"fresh" + (1 - rho) thin Z_"stay") . $

这就是重置概率的贝叶斯定义。它是第二类量（边际似然之比），这一点与原文的直觉一致；
但比较的是“留”与“换”两个假设，而不是新旧两个方差。

*闭式解。* 在 $theta_A$ 附近对新任务的对数似然取二阶近似：梯度 $g$，曲率 $F$（Fisher 信息）。
旧后验取 $cal(N)(theta_A, s^2)$，重置分布取 $cal(N)(mu, sigma^2)$，记 $delta = mu - theta_A$。高斯积分给出

$ log Z_"stay" = -1/2 log(1 + F s^2) + (g^2 s^2) / (2 (1 + F s^2)), $
$ log Z_"fresh" = -1/2 log(1 + F sigma^2) + (g^2 sigma^2 + 2 g delta - F delta^2) / (2 (1 + F sigma^2)). $

我们用数值积分核对过这个式子（相对误差小于 $10^(-11)$）。

#fig(include "/figures/changepoint_posterior.typ", caption: [
  单个权重的后验变点概率 $r$，闭式公式。
  设定：旧任务的权重值 $theta_A = 1$，旧后验方差 $s^2 = 0.01$；重置分布 $cal(N)(0, 1)$；先验变点率 $rho = 0.01$（虚线）。
  新任务的似然是以 $theta_B$ 为中心、精度为 $F$ 的高斯。
  $F = 0$（死单元）时 $r$ 恒等于 $rho$。
]) <fig-changepoint>

@fig-changepoint 显示三种情形。

- *死单元*（$g = 0$，$F = 0$）：$Z_"stay" = Z_"fresh" = 1$，$r = rho$。数据不提供任何证据，重置率退回先验。
- *新任务认同旧值*（$theta_B approx theta_A$，$F$ 大）：$Z_"stay" >> Z_"fresh"$，$r$ 低于 $rho$。单元受保护。
  在 $F = 100$ 时，$r$ 在 $theta_B = 1$ 处约为 $rho$ 的十分之一。
- *新任务否定旧值*（$theta_B$ 远离 $theta_A$，$F$ 大）：$r -> 1$。单元被重置。

这同时用到了 $A$（旧后验方差 $s^2$）与 $B$（似然的 $g, F$），回应了 @sec-switch 末尾的张力。

*与原文 $alpha$ 的对应。* 当 $rho$ 很小时，$r approx rho thin Z_"fresh" \/ Z_"stay"$。
对照 $rho = lambda alpha Delta t$，得到

$ alpha_i = Z_"fresh" \/ Z_"stay" , $

即*低效用程度就是支持“换掉”的贝叶斯因子*。
死单元 $alpha = 1$，与 CBP 的直觉吻合；有用的单元 $alpha -> 0$。
与原文不同，$alpha$ 可以大于 1：被新数据明确否定的单元，重置得比基础速率更快。

== 死单元的重置率学不到

上面的结果有一个直接推论。
死单元恰是最需要重置的单元，而它们的 $r$ 与数据无关，永远等于 $rho$。
所以*基础速率 $rho$（即 CBP 的 $lambda$）无法从这些单元自身的数据中推断*，只能作为先验设定。

一个可能的出路是跨单元汇总：活单元的变点证据可以用第二类极大似然估计一个共享的 $rho$，再用到死单元上。
这是否合理，取决于“活单元的变化频率”能否代表“死单元应被重抽的频率”。本文不作判断。

== 修订后的算法草案

按变点模型，原文第 7.4 节的算法可以改写为下面的形式。
它只是把上面的公式排成步骤，没有经过实验。

```python
for each step t:
    g, F = grad_and_diag_fisher(theta, batch)       # per weight (or per unit)
    log_bf = log_Z_fresh(g, F, mu, sigma2, theta) - log_Z_stay(g, F, s2)
    r = sigmoid(log(rho) - log1p(-rho) + log_bf)    # posterior change probability
    # pick ONE reset form (section 6.2): eps = 1 is a hard jump
    reset = random(shape) < r
    theta = where(reset, sample_fresh_posterior(g, F, mu, sigma2), theta)
    s2    = where(reset, fresh_posterior_var(F, sigma2), 1 / (1 / s2 + F))
    theta = theta - lr * g_total                    # ordinary gradient step
```

与原文算法的三处不同：
重置概率来自贝叶斯因子，而非方差比；
跳跃目标是“先验 $p_0$ 经新数据更新后的后验”，而非 $p_0$ 本身；
只保留一种重置形式，避免 @sec-methods 指出的重复计数。

== 这一读法的近似

- *逐坐标分解。* 闭式解假设似然在各权重间可分（对角 Fisher）。真实网络的权重强耦合。
- *二阶近似。* 似然在 $theta_A$ 附近的二次展开，离 $theta_A$ 远时失效；$Z_"fresh"$ 恰好要在远处积分。
- *单元还是权重。* CBP 重置整个单元，并把输出权重置零。按单元建模时，$g, F$ 变为向量与矩阵，公式形式不变；输出置零对应 $p_0$ 在输出权重上是点质量 $delta(0)$。
- *真实后验不可得。* 原文第 8 节已指出，带跳跃的动力学的稳态偏离真实后验。
  在变点读法下，这个“偏离”是对的：模型本就不再假设参数静止，目标后验已经变了。
