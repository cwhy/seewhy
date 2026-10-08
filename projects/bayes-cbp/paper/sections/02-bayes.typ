#import "/template.typ": *

= 贝叶斯视角下的可塑性 <sec-bayes>

== 定义

把权重看成分布。数据分批到达：$D_1, D_2, dots$。
旧后验就是新先验：

$ p(theta | D_(1:t)) prop p(D_t | theta) thin p(theta | D_(1:t-1)). $

原文把可塑性定义为一步更新引起的后验变化：

$ cal(P)_t = D_"KL" (p(theta | D_(1:t)) || p(theta | D_(1:t-1))). $

这个定义合理。它衡量“新证据能把信念推多远”。

== 高斯情形：精度决定可塑性

先验 $cal(N)(mu_0, sigma_0^2)$，观测 $y$ 的噪声方差 $sigma_n^2$。
后验均值的移动量是

$ mu_"post" - mu_0 = underbrace(sigma_0^2 / (sigma_0^2 + sigma_n^2), K) (y - mu_0). $

系数 $K$ 就是卡尔曼增益。
先验方差大、观测噪声小，$K$ 接近 1，可塑性高。
先验精度 $sigma_0^(-2)$ 大，$K$ 接近 0，可塑性低。
这一节原文正确。

== 静态模型下，可塑性丧失是理性的 #added

原文没有指出一个后果：*如果模型假设参数永远不变，精确的贝叶斯更新本身就会丧失可塑性。*

在上面的高斯模型里连续观测 $t$ 次，后验方差为

$ sigma_t^2 = (sigma_0^(-2) + t sigma_n^(-2))^(-1) approx sigma_n^2 / t. $

增益 $K_t$ 随 $t$ 以 $1\/t$ 衰减。
第 1000 个观测对均值的影响，约为第 1 个的千分之一。
这不是缺陷：只要世界确实静止，这就是正确的推断。

所以，要从贝叶斯原理推出“重置”，必须先改模型：
先验里要写明“参数可能变化”。
@sec-changepoint 给出这样一个先验，并说明 CBP 的跳跃项恰好是它的预测步。
这正是 Kurle 等人 @kurle2020nonstationary 的#gloss[贝叶斯遗忘][Bayesian forgetting：每步把后验往先验方向“放松”一点，以表达参数可能漂移]
与 Adams 和 MacKay @adams2007bocpd 的在线变点检测所采用的思路。

== 变分推断与 EWC

原文的持续学习目标是#gloss[变分持续学习][VCL：每个任务用上一任务的近似后验作先验，做一次变分推断 @nguyen2018vcl]
的带温度版本：

$ cal(L) = bb(E)_(q_t) [log p(D_t | theta)] - beta thin D_"KL" (q_t || q_(t-1)). $

$beta$ 大，偏离旧后验的代价大，稳定。
对 $q_(t-1)$ 取拉普拉斯近似，KL 项变成二次惩罚，就得到
#gloss[EWC][弹性权重巩固：按每个权重的 Fisher 信息，惩罚它偏离旧值 @kirkpatrick2017ewc]：

$ cal(L) = cal(L)_"new" + lambda / 2 sum_i F_i (theta_i - theta_i^"old")^2. $

Fisher 大，旧后验精度高，权重不可塑。原文正确。

#callout(title: [注意一处记号混用 #fix])[
  原文第 8 节写“可塑性 = 后验方差 = 由边际似然推断的超参数 $sigma_i^2$”。
  这里有两个不同的量。
  *后验方差* $s_i^2 = (sigma_i^(-2) + F_i)^(-1)$ 由数据决定。
  *先验方差* $sigma_i^2$ 是超参数。
  二者只在数据对该权重完全无信息（$F_i = 0$）时相等。
  @sec-alpha 说明这个区分为什么影响重置强度的符号。
]
