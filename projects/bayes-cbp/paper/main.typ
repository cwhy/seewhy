// Paper entry point. Formulation only — no experiments; figures are closed-form
// curves plus two Monte Carlo checks, built by scripts/gen_figures.py on the box.
//
//   preview:  uv run --no-sync python -m shared_lib.publish projects/bayes-cbp/paper --preview
//   publish:  uv run --no-sync python -m shared_lib.publish projects/bayes-cbp/paper

#import "/template.typ": *

#show: paper.with(
  title: "跳跃重置即贝叶斯变点滤波",
  subtitle: "可塑性、Jump-SDE 与贝叶斯 CBP 的形式化整理",
  date: "2026-10-09",
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    本文整理并核对一份笔记。笔记用贝叶斯后验与带跳跃的随机微分方程，统一解释“持续反向传播”（CBP）类重置方法。
    核心结论有三条。
    第一，CBP 的跳跃项不是对贝叶斯推断的近似，而是一个精确对象：
    它正是“参数随时可能被重新抽取”这一变点先验下，贝叶斯滤波的预测步。
    第二，在这个读法下，重置概率有唯一的贝叶斯定义：后验变点概率，即两个边际似然之比。
    对梯度恒为零的“死”单元，两个边际似然相等，重置概率退回先验速率 $rho$。
    因此最需要重置的单元，恰恰无法从数据里学到重置速率。
    第三，笔记有四处公式需要修正：
    状态依赖 $alpha$ 的前向方程、跳跃方差注入式、混合时间界（应取 $alpha$ 的下确界而非均值），
    以及重置强度 $1 - sigma^2_"old" \/ sigma^2_"new"$——它混淆了后验方差与先验超参数，
    而第二类极大似然恰好把无用权重的先验方差推向零，与笔记设想的方向相反。
    其余部分——高斯情形的可塑性分析、朗之万与跳跃的时间尺度对比、方法统一表——在加注条件后成立。
  ],
)

#include "/sections/01-introduction.typ"
#include "/sections/02-bayes.typ"
#include "/sections/03-langevin.typ"
#include "/sections/04-jump-sde.typ"
#include "/sections/05-task-switch.typ"
#include "/sections/06-methods.typ"
#include "/sections/07-design.typ"
#include "/sections/08-conclusion.typ"

#bibliography("/refs.bib", title: "参考文献", style: "ieee")
