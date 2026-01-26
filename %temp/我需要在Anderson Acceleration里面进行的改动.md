# 我需要在Anderson Acceleration里面进行的改动

1. 需要使用两套AA, 其中一套用于Jacobi, 一套用于full-cycle DL-HIM

   1. 对于Jacobi-AA, 他是性能的主要来源, 建议采用以下超参数
      1. damping采用cg-style damping 或者选取大的步长 (例如 $\alpha_k = 1.0$)
      2. regularization term 选取得小一点, 外推激进一点 (1e-15)
      3. 选取中或者大的history size $m$ (10-50)
   2. 对于Full-cycle-AA, 对他需要比较谨慎, 建议采用改以下超参数
      1. damping先尝试用cg-style dampling: $\bold{\alpha = \frac{(p_k^\top r_k)}{(p_k^\top Ap_k)}}$, 或者选取小的步长 (例如 $\alpha_k = 0.2-0.7$)
         1. 对于AA给出的方向p_k
         2. 先计算出 $r_{raw} = f - A * DON(r_k)$ (未经AA的原算法候选)
         3. 然后给出一个cg-style step size $\alpha_k$
         4. 如果新的 $r_{k+1}\leq (1+\eta)r_{raw}$, 就接受
         5. 否则$\alpha_k$减半, 如此持续至多n次, 或者直至  $r_{k+1}\leq (1+\eta)r_{raw}$
         6. 如果实在不行, 甚至可以考虑直接reset AA. 或者就做一个普通迭代 $u_{k+1}= u_{raw} = DON(r_k)$
      2. 使用较少的m (2-5或者2~10), 因为非线性太严重, m太大会把非平稳阶段的信息也混在一起
      3. 可以选取稍微大一点的damping (1e-8 ~ 1e-10)

2. 可以考虑自适应触发 hybrid update, 而不是fixed iterate

   1. 平台期触发, 根据局部收缩率 $\rho_k = r_k/r_{k-1}$ 变小则使用NN
   2.  AA 失效触发
      1. acceptance (AA 是否被 acceptance 接受) 失败率高
      2. beta (backtracking 后的最终步长）很小
      3. 或者AA的改进低于阈值 (和局部收缩率差不多的想法)
   3. 最小化 NN calls
      1. 尽量减小NN的调用次数和成本, 来增加效果
      2. 只有当 $r_k$ 在 $w$ 步内下降不到某个比例（例如 < 10%）才调用一次 NN

3. 叙事手法: 在混合迭代中，AA 不是替代 NN，而是让并行友好的基础平滑器（Jacobi）在 NN 插入的间隙里更快、更稳，从而减少 NN 调用次数/减少总循环成本。

4. 可以考虑使用Anderson mixing. 

   1. Anderson Acceleration的缺点

      - 对于一般的$Mf = M(b-Au)$的问题, 我们使用AA是因为以下原因:
        $$
        \begin{aligned}
        \sum \alpha_k Mf_k &= \sum\alpha_kM(b-Au_k) 
        \\&=M[b - A(\sum\alpha_ku_k)]
        \\&=M(b - A\tilde u)
        \end{aligned}
        $$

        - 因此 $\min \sum\alpha_kMf_k=\min M(b-A\tilde u)=\min M\tilde r$ 

      - 如果M是线性的且well-defined, 那么 我们有 $M\tilde r=0 \to r=0$

      - 但是用于 full-cycle DL-HIM这种nonlinear precoditioer, 我们首先就不会有
        $$
        \begin{aligned}
        \sum\alpha_k(Mb-MAu_k)&=M[b - A(\sum\alpha_ku_k)]
        
        \end{aligned}
        $$
        这个只能在很小的范围内成立, 因此我们需要减小m的数值

      - 除此之外, 即使我们满足上式, $M\tilde r=\tilde u_{pred}=0$ 也不一定能得出 $r=0$. 比如进入residual plateau或者DeepONet不善于处理的spectral domain的时候, 就会出现这种情况. 但是离达到tolerance依然很远.

   2. 对于这个错位, 我们可以考虑直接使用 $\tilde r$ 作为damping 或者 acceptance的标准, 而不是 $M\tilde r$. 

   3. 除此之外, 完全改写Anderson Acceleration也是一个不错的选择, 例如 Anderson mixing. 我们使用full cycle DLM作为一个preconditioner, 只用于外围给出prediction, 而内部则使用 $(b-Au_x)$ 和 $u_x$ 两个pool来执行 Anderson Acceleration

      1. **输入：** $u_0 = \mathbf{0}$ (Zero guess)。

      2. **计算残差：** $r_0 = b - A u_0 = b$。

      3. **检查历史：** 历史为空 ($m=0$)，无法进行最小二乘优化。

      4. **默认行为：** 直接信任全循环神经网络（预条件子）。

         $$u_1 = u_0 + \beta N(r_0) = N(b)$$

      5. **存入历史：** 现在你有了一对数据 $\{u_1, r_1\}$，存入池$\{u_i, r_i\}$中。

      6. **下一步 ($k=1$)：** 当你有了 $u_1$ 和 $r_1$ 后，历史池里有 2 个点，这时就可以开始做 Mixing 了

         - $$\min_{\gamma} \left\| \sum_{i=0}^m \gamma_i r_{k-m+i} \right\|_2$$

         - $$u^*_{mix} = \sum_{i=0}^m \gamma_i u_{k-m+i}$$
         - 我们认为 $u_{mix}$ 已经是AA能给出的最好的解. 然后我们用P进一步优化

         - $$u_{k+1} = u^*_{mix} + \beta \cdot P(r^*_{mix})$$
         - P 作为full cycle operator 给出一个建议方向, 可以加上acceptance/backtracking

      7. Repeat the above

   4. 上述Anderson mixing 算法建立在以下假设:

      - 与AA相同, $\min\sum\alpha_i r_i = \min\ b- A(\sum\alpha_iu_i)=\min b-A\tilde u$
      - 