#import "/template.typ": *

= 现有重置方法的统一解释 <sec-methods>

== 统一表

原文把四种方法放进同一个跳跃—扩散框架。下表保留原文的分类，并加一列核查状态。

#figure(
  table(
    columns: (auto, 1fr, auto, 1fr),
    align: (left, left, left, left),
    [方法], [原文：触发条件 / 跳跃形式], [SDE 中的角色], [核查],
    [CBP @dohare2024loss],
    [输出权重平方和（启发式）；硬重置到初始化],
    [状态依赖跳跃],
    [原文效用为“激活幅度 × 输出权重绝对值之和”的滑动平均，不是平方和；
     重置时输出权重置零；按固定替换率选效用最低者，而非逐单元独立抛硬币。],
    [SNR @farias2024snr],
    [基于放电率的统计检验；硬重置],
    [事件驱动跳跃],
    [大致一致：单元静默时长相对其历史放电率显著过长时重置。未逐式核对。],
    [G-CBP],
    [与梯度范数相关；硬重置],
    [跳跃强度随梯度变],
    [未找到出处，本文不评论。],
    [Auto-SPR @galashov2024soft],
    [无显式触发；向初始化的 OU 型软重置，漂移参数 $gamma_t$ 在线学习],
    [跳跃项的扩散极限],
    [方法描述大致一致。“扩散极限”的说法需修正，见下文。],
  ),
  caption: [原文第 6 节的方法表，加上核查列。“未逐式核对”表示只核对了方法的总体机制。],
  kind: table,
)

#added 关于 CBP 的两点细节会影响建模。
第一，CBP 重置的是整个单元：输入权重重新随机抽取，输出权重置零。
输出置零保证重置那一刻网络函数不变。逐权重的 $theta_i$ 记号看不到这一点。
第二，CBP 每步按固定比例替换效用最低的单元。
这是一个排序规则：$alpha$ 是“是否排在最后”的指示函数，而不是效用的连续函数。

== 软重置与硬重置是同一族 #fix

原文称 OU 型软重置是“跳跃项的扩散极限”。
硬重置把 $theta$ 整个换掉，每次跳跃是有限大小的，没有通常意义下的扩散极限。

更准确的说法是：二者属于同一个“部分重置”族。
取步长 $epsilon in (0, 1]$，令

$ theta' = mu + (1 - epsilon)(theta - mu) + sqrt(epsilon (2 - epsilon)) thin sigma thin z,
  quad z tilde cal(N)(0, 1). $

对任意 $epsilon$，这一步都保持 $cal(N)(mu, sigma^2)$ 不变。
$epsilon = 1$ 是硬重置。
以速率 $lambda alpha \/ epsilon$ 施加、令 $epsilon -> 0$，得到 OU 过程

$ upright(d) theta = lambda alpha (mu - theta) thin upright(d) t + sqrt(2 lambda alpha sigma^2) thin upright(d) W . $

#gloss[shrink-and-perturb][每隔一段时间把权重乘一个小于 1 的系数再加噪声 @ash2020warm] 是中间的 $epsilon$。

== 统一方程中的重复计数 #fix

原文的统一方程把三种重置写在一起：

$ upright(d) theta_i = -nabla_i U thin upright(d) t + lambda alpha_i (mu_i^* - theta_i) thin upright(d) t + sqrt(2 T lambda alpha_i) thin upright(d) W_i + (xi_i - theta_i) thin upright(d) N_i . $

第二、三项是 OU 软重置，第四项是硬重置，二者都以 $lambda alpha_i$ 为速率。
按上一节，它们是同一机制的两个极端；同时施加，重置速率实际翻倍。
此外，要让 OU 部分保持 $cal(N)(mu_i^*, sigma_i^(2*))$ 不变，噪声系数应为 $sqrt(2 lambda alpha_i sigma_i^(2*))$，即 $T = sigma_i^(2*)$，而不是独立的温度。
建议的写法是：选定一个 $epsilon$，只保留一种重置。
