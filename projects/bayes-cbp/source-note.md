<!-- The user's original note (2026-10-09), kept verbatim. The paper in paper/
     formalises and checks it; section numbers there cite §n of this file. -->

# 从贝叶斯第一性原理到 Jump-SDE：可塑性的统一视角与贝叶斯 CBP 设计

---

## 1. 贝叶斯视角下的可塑性

### 1.1 可塑性的定义

在贝叶斯神经网络中，权重是分布 \(p(\theta|D)\)，而非点估计。可塑性定义为**后验分布对新证据的更新能力**：

\[
\text{Plasticity} \sim D_{KL}\big(p(\theta|D_{1:t}) \,\|\, p(\theta|D_{1:t-1})\big)
\]

旧后验成为新先验：

\[
p(\theta|D_{1:t}) \propto p(D_t|\theta)\,p(\theta|D_{1:t-1})
\]

- 可塑性高：新似然能显著改变后验；
- 可塑性低：旧后验太强/太确定，新数据难以改变它；
- 稳定性高：旧后验精度高，KL 惩罚强。

### 1.2 高斯情形：精度决定可塑性

若先验 \(N(\mu_0,\sigma_0^2)\)，新观测噪声 \(\sigma_n^2\)，后验均值更新：

\[
\mu_{\text{post}}
=
\frac{\sigma_n^{-2} y + \sigma_0^{-2}\mu_0}{\sigma_n^{-2}+\sigma_0^{-2}}
\]

均值移动量：

\[
\mu_{\text{post}}-\mu_0
=
\frac{\sigma_0^2}{\sigma_0^2+\sigma_n^2}(y-\mu_0)
\]

因此：

- 先验方差 \(\sigma_0^2\) 大 → 可塑性高；
- 旧后验精度 \(\sigma_0^{-2}\) 大 → 可塑性低；
- 新数据噪声 \(\sigma_n^2\) 小、信息量大 → 可塑性高。

### 1.3 变分推断视角

ELBO 持续学习形式：

\[
\mathcal{L}
=
\mathbb{E}_{q_t}[\log p(D_t|\theta)]
-
\beta\,D_{KL}(q_t(\theta)\|q_{t-1}(\theta))
\]

- \(\beta\) 大 → 稳定性强，可塑性低；
- \(\beta\) 小 → 可塑性高，容易遗忘；
- \(q_{t-1}\) 方差大 → 可塑性高；
- \(q_{t-1}\) 方差小 → 已巩固，可塑性低。

EWC 是它的拉普拉斯近似：

\[
\mathcal{L}
=
\mathcal{L}_{\text{new}}
+
\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^{\text{old}})^2
\]

Fisher 大 = 旧后验精度高 = 不可塑；Fisher 小 = 可塑。

---

## 2. 朗之万采样视角

### 2.1 朗之万动力学

权重 \(\theta\) 看成粒子，后验对应势能：

\[
U(\theta)=-\log p(D|\theta)-\log p(\theta)
\]

过阻尼朗之万动力学：

\[
d\theta_t=-\nabla U(\theta_t)\,dt+\sqrt{2T}\,dW_t
\]

稳态分布：

\[
p_\infty(\theta)\propto \exp\left(-\frac{U(\theta)}{T}\right)
\]

离散化即 SGLD：

\[
\theta_{k+1}
=
\theta_k-\eta\nabla \tilde U(\theta_k)+\sqrt{2\eta T}\,\xi_k
\]

### 2.2 可塑性的朗之万度量

- 后验分布变化率：\(D_{KL}(p_t\|p_{t-1})\)
- 均方位移：\(\text{MSD}(t)\sim 2dTt\)
- 跨越势垒：Kramers 逃逸率
  \[
  k\sim \exp\left(-\frac{\Delta U}{T}\right)
  \]
- 对新势能的响应幅度：
  \[
  \Delta \theta \approx \frac{g}{\kappa},\qquad \kappa=U''\approx \text{旧后验精度}
  \]

可塑性—稳定性权衡：

\[
\text{可塑性}\sim \frac{T}{\kappa_{\text{old}}+\beta}
\]

---

## 3. Jump-SDE 统一框架

### 3.1 三种动力学的并排形式

\[
\boxed{
\begin{array}{ll}
\text{SGD:} & \displaystyle d\theta_t = -\nabla U(\theta_t)\,dt + \sigma\,dW_t \\[2mm]
\text{Langevin:} & \displaystyle d\theta_t = -\nabla U(\theta_t)\,dt + \sqrt{2T}\,dW_t \\[2mm]
\text{Jump-CBP:} & \displaystyle d\theta_t = -\nabla U(\theta_t)\,dt + (\xi_t-\theta_{t^-})\,dN_t
\end{array}
}
\]

其中：

\[
N_t \sim \text{Poisson}\left(\int_0^t \lambda\,\alpha(\theta_{s^-})\,ds\right),
\qquad
\xi_t\sim p_{\text{init}}
\]

\[
dN_t\in\{0,1\},
\qquad
\lambda_t=\lambda\,\alpha(\theta_{t^-})
\]

\(\alpha(\theta)\in[0,1]\) 是“低效用程度”：低效用单元 \(\alpha\to 1\)，高效用单元 \(\alpha\to 0\)。

### 3.2 离散版本

\[
\boxed{
\begin{aligned}
\text{SGD:}\quad 
\theta_{k+1} &= \theta_k-\eta \nabla \hat U(\theta_k)\\[1mm]
\text{SGLD:}\quad 
\theta_{k+1} &= \theta_k-\eta \nabla \hat U(\theta_k)+\sqrt{2\eta T}\,\xi_k\\[1mm]
\text{Jump-CBP:}\quad 
\theta_{k+1} &= \theta_k-\eta \nabla \hat U(\theta_k)+(\xi_k-\theta_k)B_k
\end{aligned}
}
\]

其中 \(B_k\sim \text{Bernoulli}(\eta\lambda\alpha(\theta_k))\)，\(\xi_k\sim p_{\text{init}}\)。

### 3.3 生成元与 Kolmogorov 前向方程

跳扩散生成元：

\[
\boxed{
\mathcal L f
=
b\cdot\nabla f
+
\frac12\sigma\sigma^T:\nabla^2 f
+
\int_{\mathcal Z}\big[f(x+c(x,z))-f(x)\big]\,\nu(dz)
}
\]

CBP 的 Fokker–Planck 方程：

\[
\partial_t p(\theta,t)
=
-\nabla\cdot(bp)
+
\lambda\alpha
\big[p_{\text{init}}(\theta)-p(\theta,t)\big]
\]

忽略漂移后：

\[
p(\theta,t)
=
p_{\text{init}}(\theta)
+
\big(p(\theta,0)-p_{\text{init}}(\theta)\big)e^{-\lambda\alpha t}
\]

重置时间：

\[
\tau_J^{\text{reset}}
=
\frac{1}{\lambda\alpha}
\]

### 3.4 逃逸时间对比

\[
\boxed{
\begin{aligned}
\text{Langevin:}\quad
&\tau_L^{\text{escape}}
\sim
\exp\left(\frac{\Delta U}{T}\right)
\\[2mm]
\text{Jump-CBP:}\quad
&\tau_J^{\text{reset}}
=
\frac{1}{\lambda\alpha}
\end{aligned}
}
\]

当 \(\Delta U/T\gg 1\) 时，\(\tau_L^{\text{escape}} \gg \tau_J^{\text{reset}}\)。

### 3.5 谱隙与混合时间

跳扩散谱隙：

\[
\lambda_1^{\text{jump}}
\simeq
\lambda_1^{\text{diff}}
+
\lambda\,\bar\alpha
\]

混合时间：

\[
\tau_{\text{mix}}^{\text{jump}}
\simeq
\frac{1}{\lambda\bar\alpha}
\]

有限且不依赖势垒。

### 3.6 方差注入

由 Itô 公式：

\[
\Delta\operatorname{Var}_{\text{jump}}
=
\lambda\alpha\,\mathbb E[(\xi-\theta)^2]
\]

对比 Langevin：

\[
\Delta\operatorname{Var}_{\text{diff}}=2T
\]

---

## 4. 任务切换后的可塑性分析

任务从 \(A\) 切到 \(B\)，新损失：

\[
U_B(\theta)=U_B^{\text{data}}(\theta)+\frac12(\theta-\theta_A)^T\Lambda_A(\theta-\theta_A)
\]

在 \(\theta_A\) 附近二阶展开：

\[
U_B(\theta)\approx U_B(\theta_A)+g_B^T(\theta-\theta_A)+\frac12(\theta-\theta_A)^T H_B(\theta-\theta_A)
\]

### 4.1 三种动力学的适应时间

\[
\boxed{
\begin{aligned}
\text{SGD:}\quad
&\tau_{\text{SGD}}\sim \frac{1}{\lambda_{\min}(H_B)}
\\[2mm]
\text{Langevin:}\quad
&\tau_{\text{Langevin}}\sim \max\left(\frac{1}{\lambda_{\min}(H_B)},\ \exp\left(\frac{\Delta U_B}{T}\right)\right)
\\[2mm]
\text{Jump-CBP:}\quad
&\tau_{\text{CBP}}\sim \frac{1}{\lambda\bar\alpha_B}+\tau_{\text{SGD}}(\xi)
\end{aligned}
}
\]

可塑性：

\[
\boxed{
\begin{aligned}
\mathcal P_{\text{SGD}}&\sim \lambda_{\min}(H_B)
\\[1mm]
\mathcal P_{\text{Langevin}}&\sim \min\left(\lambda_{\min}(H_B),\ e^{-\Delta U_B/T}\right)
\\[1mm]
\mathcal P_{\text{CBP}}&\sim \lambda\bar\alpha_B
\end{aligned}
}
\]

当 \(\lambda_{\min}(H_B)\approx 0\) 且 \(\Delta U_B/T\gg 1\) 时，SGD 和 Langevin 适应极慢，而 Jump-CBP 仍以 \(1/(\lambda\bar\alpha_B)\) 恢复可塑性。

---

## 5. 低效用单元的贝叶斯定义

\[
\boxed{\text{低效用单元} = \text{对后验预测分布贡献可忽略的单元}}
\]

几种等价刻画：

1. **信息增益小**：
   \[
   u_i^{\text{info}}
   =
   D_{KL}\big(q(\theta_i|D)\,\|\,p(\theta_i)\big)\approx 0
   \]

2. **Fisher 信息小**：
   \[
   u_i^{\text{Fisher}} = F_{ii}
   \]
   在拉普拉斯近似下 \(\Lambda_i \approx F_{ii}\)。

3. **对预测贡献可忽略**：
   \[
   u_i^{\text{pred}}
   =
   D_{KL}\big(p(y|x,D)\,\|\,p_{-i}(y|x,D)\big)\approx 0
   \]

4. **信噪比低**：
   \[
   u_i^{\text{SNR}}
   =
   \frac{\mu_i^2}{\sigma_i^2}
   \]

任务切换后，低效用相对于新任务 \(B\) 定义：

\[
\hat u_i^B
\approx
F_{ii}^B(\theta_A)
=
\mathbb E_{D_B}
\left[
\left(\frac{\partial \log p(D_B|\theta)}{\partial \theta_i}\bigg|_{\theta_A}\right)^2
\right]
\]

---

## 6. 现有 CBP 方法的统一解释

| 方法 | 触发条件 \(\alpha(\theta)\) | 跳跃强度 \(\lambda\) | 跳跃形式 | SDE 中的角色 |
| :--- | :--- | :--- | :--- | :--- |
| **硬 CBP** | 启发式（输出权重平方和） | 固定 | 硬重置到 \(\theta_{\text{init}}\) | 状态依赖的泊松跳跃 |
| **SNR** | 基于放电率的统计检验 | 固定，由检验门控 | 硬重置 | 精确的事件驱动跳跃 |
| **G-CBP** | 与梯度范数相关 | 自适应，与梯度耦合 | 硬重置 | 状态依赖的跳跃强度 |
| **Auto-SPR** | 无显式触发，持续作用 | 自适应漂移参数 \(\gamma_t\) | 软重置（OU 过程） | 跳跃项的扩散极限 |

统一形式：

\[
d\theta_i
=
-\nabla_i U\,dt
+
\lambda\alpha_i(\theta)(\mu_i^*-\theta_i)\,dt
+
\sqrt{2T\lambda\alpha_i(\theta)}\,dW_i
+
(\xi_i-\theta_i)\,dN_i
\]

---

## 7. First-Principle 贝叶斯 CBP 设计

### 7.1 分层贝叶斯模型

\[
\boxed{
\begin{aligned}
\theta_i &\sim N(\mu_i,\sigma_i^2)\\
\mu_i &\sim N(0,\tau^2)\\
\sigma_i^2 &\sim \text{InvGamma}(a,b)
\end{aligned}
}
\]

- \(\sigma_i^2\) 小：先验强，后验稳定，可塑性低；
- \(\sigma_i^2\) 大：先验弱，后验可移动，可塑性高；
- \(\mu_i\)：重置中心。

### 7.2 第二类贝叶斯（经验贝叶斯）

边际似然：

\[
\boxed{
p(D_t|D_{1:t-1},\mu,\sigma^2)
=
\int
p(D_t|\theta)\,
p(\theta|D_{1:t-1},\mu,\sigma^2)
\,d\theta
}
\]

超参数更新：

\[
\boxed{
(\mu^*,\sigma^{2*})
=
\arg\max_{\mu,\sigma^2}
\log p(D_t|D_{1:t-1},\mu,\sigma^2)
}
\]

### 7.3 重置强度的第一性原理定义

\[
\boxed{
\alpha_i
=
1-\frac{\sigma_i^{2(\text{old})}}{\sigma_i^{2(\text{new})}}
}
\]

- \(\alpha_i\to 1\)：需要重置；
- \(\alpha_i\to 0\)：需要稳定。

跳跃强度：

\[
\boxed{
\lambda_i
=
\lambda\left(
1-\frac{\sigma_i^{2(\text{old})}}{\sigma_i^{2(\text{new})}}
\right)
}
\]

跳跃目标：

\[
\xi_i\sim N(\mu_i^*,\sigma_i^{2*})
\]

### 7.4 算法流程

```python
for each step:
    forward/backward
    # 1. 后验更新（变分或采样）
    q_t = update_posterior(q_{t-1}, D_t)

    # 2. 经验贝叶斯更新超参数
    mu_star, sigma2_star = empirical_bayes(q_t, D_t)

    # 3. 计算重置强度
    alpha_i = 1 - sigma2_old_i / sigma2_new_i

    # 4. 软重置
    theta_i = (1 - alpha_i) * theta_i + alpha_i * mu_star_i

    # 5. 可选硬跳跃
    if random() < lambda * alpha_i:
        xi_i = sample(N(mu_star_i, sigma2_star_i))
        theta_i = xi_i
```

### 7.5 连续时间极限

\[
\boxed{
d\theta_i
=
-\nabla_i U\,dt
+
\lambda\alpha_i(\theta)(\mu_i^*-\theta_i)\,dt
+
\sqrt{2T\lambda\alpha_i(\theta)}\,dW_i
+
(\xi_i-\theta_i)\,dN_i
}
\]

其中 \(\alpha_i(\theta)\) 由边际似然推断，\(\mu_i^*\) 是学习到的重置中心。

---

## 8. 总结

\[
\boxed{
\begin{aligned}
&\text{可塑性}
&&=
\text{后验方差}
=
\text{由边际似然推断的超参数 } \sigma_i^2
\\[1mm]
&\text{重置}
&&=
\text{向学习到的后验中心 } \mu_i^* \text{ 收缩}
\\[1mm]
&\text{重置强度}
&&=
1-\frac{\sigma_i^{2(\text{old})}}{\sigma_i^{2(\text{new})}}
\\[1mm]
&\text{Jump-CBP 恢复可塑性}
&&\Longleftrightarrow\quad
\text{生成元非局部项 } \lambda\alpha \text{ 主导}
\\[1mm]
&\text{逃逸时间}
&&\Longleftrightarrow\quad
\tau\sim 1/(\lambda\alpha)\ \text{而非}\ e^{\Delta U/T}
\\[1mm]
&\text{代价}
&&\Longleftrightarrow\quad
\text{非平衡稳态，偏离真实后验}
\end{aligned}
}
\]

核心定理：**跳扩散的 Kramers 定律 + 大偏差原理 + 谱隙分析**。它们共同说明：只要跳跃强度 \(\lambda\alpha\) 足够大，可塑性就能在有限时间内恢复，而不受势垒指数限制。

第一性原理设计可以总结为一句话：

\[
\boxed{
\text{可塑性}
=
\text{后验方差}
=
\text{由边际似然推断的超参数 } \sigma_i^2
}
\]

\[
\boxed{
\text{重置}
=
\text{向学习到的后验中心 } \mu_i^* \text{ 收缩}
}
\]

\[
\boxed{
\text{重置强度}
=
1-\frac{\sigma_i^{2(\text{old})}}{\sigma_i^{2(\text{new})}}
}
\]

这就是从贝叶斯第一性原理出发，统一解释并超越现有 CBP 实现的设计理念。
