#import "/template.typ": *

= Jump-SDE 统一框架 <sec-jump>

== 三种动力学

原文把三种训练动力学写成同一方程的三种噪声项：

$ "SGD:" quad & upright(d) theta_t = -nabla U thin upright(d) t + sigma thin upright(d) W_t \
  "朗之万:" quad & upright(d) theta_t = -nabla U thin upright(d) t + sqrt(2 T) thin upright(d) W_t \
  "Jump-CBP:" quad & upright(d) theta_t = -nabla U thin upright(d) t + (xi_t - theta_(t^-)) thin upright(d) N_t $

最后一行里，$N_t$ 是计数过程，在时刻 $t$ 跳一次的速率为 $lambda alpha(theta_(t^-))$；
跳跃时参数被替换为新抽的 $xi_t tilde p_"init"$。
$alpha(theta) in [0, 1]$ 是“低效用程度”：没用的单元 $alpha -> 1$，有用的单元 $alpha -> 0$。

离散版本把跳跃换成伯努利开关 $B_k tilde "Bernoulli"(eta lambda alpha(theta_k))$：

$ theta_(k+1) = theta_k - eta nabla hat(U)(theta_k) + (xi_k - theta_k) B_k . $

这要求 $eta lambda alpha <= 1$，原文未写明。

对 SGD 一行的补充见 @sec-langevin：SGD 噪声依赖状态，在死单元上为零。
下文主要讨论跳跃项。

== 前向方程 #fix

原文给出的 CBP 前向方程是

$ partial_t p = -nabla dot (b p) + lambda alpha [p_"init"(theta) - p(theta, t)]. $

它只在 $alpha$ 是常数时成立。
当 $alpha$ 依赖状态时，从生成元出发：跳跃部分作用在测试函数 $f$ 上为
$lambda alpha(theta) integral [f(xi) - f(theta)] p_"init"(upright(d) xi)$。
取伴随，得到正确的方程：

$ partial_t p = -nabla dot (b p) + lambda [overline(alpha)(t) thin p_"init"(theta) - alpha(theta) thin p(theta, t)],
  quad overline(alpha)(t) = integral alpha thin p thin upright(d) theta . $

两项的含义不同。
*流出*项 $alpha(theta) p(theta)$ 只取决于本地：低效用的地方流出快。
*流入*项 $overline(alpha) p_"init"$ 取决于全体：所有流出的质量，按 $p_"init"$ 重新撒回。

忽略漂移时，稳态满足 $alpha(theta) p_infinity(theta) = overline(alpha) p_"init"(theta)$，即

$ p_infinity prop p_"init" \/ alpha . $

原文的方程则预测 $p_infinity = p_"init"$，与 $alpha$ 无关。
@fig-stationary 用二十万个粒子检验了两者：模拟结果与修正式重合，与原文式明显不同。

#fig(include "/figures/stationary_density.typ", caption: [
  状态依赖 $alpha$、无漂移时的稳态密度。
  设定：$p_"init" = cal(N)(0, 1.5^2)$；$|theta| < 1$ 处 $alpha = 0.1$，其余处 $alpha = 1$；$lambda = 1$。
  蒙特卡洛：20 万粒子，步长 0.01，运行到 $t = 40$（是最慢重置时间 10 的四倍）。
  高效用区（中间）的密度约为 $p_"init"$ 的 1.8 倍（$10 / 5.46$）。
]) <fig-stationary>

一个推论：若某区域上 $alpha = 0$ 而 $p_"init"$ 在那里有质量，
那么该区域是吸收的，长期看所有质量都会聚到那里。
这正是“有用的单元永不重置”的数学表达。

== 逃逸时间 #fix

原文对比两个时间尺度：

$ tau_L^"escape" tilde exp(Delta U \/ T), quad quad tau_J^"reset" = 1 / (lambda alpha). $

第二式是“等到第一次跳”的时间，不是“离开旧盆地”的时间。
新抽的 $xi$ 可能又落回旧盆地。设落回的概率是 $pi_A$，需要的跳跃次数服从几何分布，于是

$ tau_J^"escape" = 1 / (lambda alpha (1 - pi_A)) . $

结论不变：它不随势垒高度增长。
@fig-escape 画出两者：势垒超过约 $5 T$ 后，朗之万的等待时间超过跳跃。

#fig(include "/figures/escape_times.typ", caption: [
  两种逃离方式的等待时间（以 10 为底的对数），闭式公式，不是模拟。
  朗之万取 Kramers 公式并令前置因子为 1；跳跃取 $lambda alpha = 0.01$。
  “$1 - pi_A$”曲线取 $pi_A = 0.5$，即一半的新抽样落回旧盆地。
]) <fig-escape>

== 混合时间 #fix

原文写跳扩散的#gloss[谱隙][生成元最小非零特征值的绝对值，决定分布收敛到稳态的指数速率]
$lambda_1^"jump" approx lambda_1^"diff" + lambda overline(alpha)$，以及 $tau_"mix" approx 1 \/ (lambda overline(alpha))$。
两处都需修正。

*可加性。* 谱隙相加，需要扩散部分与“重抽”算子可交换。
这只在 $p_"init"$ 恰好是扩散部分的稳态分布时成立。
训练时 $p_"init"$ 是初始化分布，不是 $exp(-U\/T)$，所以一般不成立。

*应取下确界，不是均值。* 设 $alpha(theta) >= alpha_min > 0$ 处处成立。
把跳跃速率拆成两部分：与状态无关的 $lambda alpha_min$，和余下的 $lambda(alpha - alpha_min)$。
让两个副本共享第一部分的跳跃时刻与跳跃目标 $xi$；第一次共享跳跃之后，两者完全重合。
这是#gloss[Doeblin 耦合][两个从不同起点出发的副本，在某个随机时刻被强制合二为一；合一前的时间控制收敛速度 @meyn2009markov]，给出

$ norm(P_t (x, dot) - P_t (y, dot))_"TV" <= e^(-lambda alpha_min t). $

所以有限的、与势垒无关的混合时间需要 $alpha_min > 0$。
用均值 $overline(alpha)$ 不够：只要一部分状态的 $alpha$ 为零，就没有这个保证。

这其实是设计的一部分。
有用的单元 $alpha -> 0$，本来就*不该*快速混合。
“可塑性在有限时间内恢复”只对被判为低效用的单元成立。

== 方差注入 #fix

原文由 Itô 公式写 $Delta "Var"_"jump" = lambda alpha thin bb(E)[(xi - theta)^2]$。
右边其实是*平均跳跃距离的平方*，不是方差的变化率。
设 $alpha$ 为常数，$xi tilde cal(N)(mu_0, sigma_0^2)$，当前均值 $m$、方差 $V$。直接计算矩：

$ dot(m) = lambda alpha (mu_0 - m), quad quad
  dot(V) = lambda alpha [sigma_0^2 - V + (mu_0 - m)^2]. $

原文式等于 $lambda alpha [sigma_0^2 + V + (mu_0 - m)^2]$，多出 $2 lambda alpha V$。
差别是定性的：修正式中，当 $V > sigma_0^2$ 且均值已对齐时，跳跃*减少*方差。
重置是向 $p_"init"$ 收缩，不只是加噪声。
@fig-variance 用模拟检验。

#fig(include "/figures/jump_variance.typ", caption: [
  纯跳跃（无漂移）下参数方差随时间的变化。
  设定：$lambda alpha = 1$，$p_"init" = cal(N)(1, 0.5^2)$，初始分布 $cal(N)(-2, 2^2)$。
  蒙特卡洛：20 万粒子，步长 0.001。
  修正式与模拟重合；原文式无界增长。方差先升后降：均值移动先贡献方差，之后向 $sigma_0^2 = 0.25$ 收缩。
]) <fig-variance>

对照的朗之万注入率 $2 T$（每个维度）是噪声项的贡献，原文正确。
