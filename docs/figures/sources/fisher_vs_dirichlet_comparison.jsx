function FigureCard({ title, accent, bg, border, subtitle, children }) {
  return (
    <div
      style={{
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 22,
        padding: 16,
        display: "grid",
        gap: 10,
      }}
    >
      <div>
        <div style={{ fontSize: 22, fontWeight: 800, color: accent }}>{title}</div>
        <div style={{ marginTop: 4, fontSize: 13.5, lineHeight: 1.65, color: "#5d554b" }}>{subtitle}</div>
      </div>
      <div
        style={{
          background: "rgba(255,255,255,0.75)",
          border: "1px solid #eadfce",
          borderRadius: 18,
          padding: 12,
          minHeight: 220,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function FormulaCard({ title, accent, bg, border, principle, example, diff }) {
  return (
    <div
      style={{
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 22,
        padding: 16,
        display: "grid",
        gap: 12,
      }}
    >
      <div style={{ fontSize: 22, fontWeight: 800, color: accent }}>{title}</div>

      <div>
        <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 6, color: "#3f3a34" }}>原理公式</div>
        <div
          style={{
            background: "#fffdf8",
            border: "1px solid #e8dece",
            borderRadius: 14,
            padding: 12,
            fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
            fontSize: 12.5,
            lineHeight: 1.68,
            color: "#3e3832",
            whiteSpace: "pre-wrap",
          }}
        >
          {principle.join("\n")}
        </div>
      </div>

      <div>
        <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 6, color: "#3f3a34" }}>数字 Example</div>
        <div
          style={{
            background: "#f8f2e6",
            border: "1px solid #e5d5bc",
            borderRadius: 14,
            padding: 12,
            fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
            fontSize: 12.5,
            lineHeight: 1.68,
            color: "#3e3832",
            whiteSpace: "pre-wrap",
          }}
        >
          {example.join("\n")}
        </div>
      </div>

      <div>
        <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 6, color: "#3f3a34" }}>和 Gaussian 的区别</div>
        <div style={{ fontSize: 13.5, lineHeight: 1.75, color: "#4e473f" }}>{diff}</div>
      </div>
    </div>
  );
}

function GaussianFigure() {
  return (
    <svg viewBox="0 0 260 180" style={{ width: "100%", height: "auto", display: "block" }}>
      <circle cx="48" cy="92" r="24" fill="#dce8ff" stroke="#7fa2ea" strokeWidth="2.5" />
      <text x="30" y="97" fontSize="12.5" fontWeight="700" fill="#355c97">噪声</text>

      <circle cx="212" cy="92" r="24" fill="#ffe7d2" stroke="#e1a86e" strokeWidth="2.5" />
      <text x="193" y="97" fontSize="12.5" fontWeight="700" fill="#9a601e">目标</text>

      <path d="M72 92 C108 92, 150 92, 188 92" fill="none" stroke="#6a7d98" strokeWidth="4" strokeDasharray="10 6" />
      <polygon points="183,86 196,92 183,98" fill="#6a7d98" />

      <circle cx="128" cy="92" r="11" fill="#8c96aa" />
      <text x="91" y="135" fontSize="13" fill="#5a5348">中间态只是普通连续向量</text>
      <text x="74" y="154" fontSize="12" fill="#7a7267">不一定是概率分布</text>
    </svg>
  );
}

function DirichletFigure() {
  return (
    <svg viewBox="0 0 260 180" style={{ width: "100%", height: "auto", display: "block" }}>
      <polygon points="30,150 130,28 230,150" fill="#fff5ea" stroke="#cb7a35" strokeWidth="3" />
      <circle cx="30" cy="150" r="5" fill="#cb7a35" />
      <circle cx="130" cy="28" r="5" fill="#cb7a35" />
      <circle cx="230" cy="150" r="5" fill="#cb7a35" />
      <text x="14" y="168" fontSize="12" fill="#815732">p1</text>
      <text x="122" y="18" fontSize="12" fill="#815732">p2</text>
      <text x="214" y="168" fontSize="12" fill="#815732">p3</text>

      <circle cx="106" cy="108" r="9" fill="#cb7a35" />
      <path d="M106 108 C112 90,120 73,127 46" fill="none" stroke="#cb7a35" strokeWidth="4" strokeDasharray="8 6" />
      <polygon points="121,49 129,37 133,51" fill="#cb7a35" />

      <text x="64" y="133" fontSize="12.5" fill="#6a5a47">概率质量在三角形里</text>
      <text x="76" y="151" fontSize="12.5" fill="#6a5a47">逐步朝目标顶点集中</text>
    </svg>
  );
}

function FisherFigure() {
  return (
    <svg viewBox="0 0 260 180" style={{ width: "100%", height: "auto", display: "block" }}>
      <polygon points="20,145 76,72 132,145" fill="#effbf7" stroke="#1f7d7a" strokeWidth="3" />
      <text x="32" y="164" fontSize="12" fill="#476264">单纯形</text>

      <path d="M138 88 C154 70,168 60,184 56" fill="none" stroke="#1f7d7a" strokeWidth="3" strokeDasharray="8 5" />
      <polygon points="177,52 188,55 180,63" fill="#1f7d7a" />

      <circle cx="208" cy="102" r="42" fill="none" stroke="#1f7d7a" strokeWidth="3" />
      <path d="M184 132 A42 42 0 0 1 232 85" fill="none" stroke="#1f7d7a" strokeWidth="4" strokeDasharray="8 6" />
      <polygon points="226,82 235,81 231,90" fill="#1f7d7a" />

      <text x="146" y="86" fontSize="12.5" fill="#4a6667">开方到球面</text>
      <text x="166" y="153" fontSize="12.5" fill="#4a6667">沿最短弧线走</text>
    </svg>
  );
}

function GumbelFigure() {
  return (
    <svg viewBox="0 0 260 180" style={{ width: "100%", height: "auto", display: "block" }}>
      <rect x="24" y="48" width="58" height="42" rx="12" fill="#f1eaff" stroke="#bfa8f1" strokeWidth="2" />
      <text x="38" y="74" fontSize="12.5" fontWeight="700" fill="#6e4ba7">logits</text>

      <rect x="102" y="48" width="58" height="42" rx="12" fill="#fff0ea" stroke="#efb69d" strokeWidth="2" />
      <text x="116" y="74" fontSize="12.5" fontWeight="700" fill="#a35a2c">+ 噪声</text>

      <rect x="180" y="48" width="58" height="42" rx="12" fill="#fff5ea" stroke="#e4bf8d" strokeWidth="2" />
      <text x="189" y="68" fontSize="12.5" fontWeight="700" fill="#9a601e">softmax</text>
      <text x="198" y="84" fontSize="11.5" fill="#9a601e">/ 温度</text>

      <path d="M82 69 H102" stroke="#91887a" strokeWidth="3" />
      <path d="M160 69 H180" stroke="#91887a" strokeWidth="3" />

      <path d="M36 136 Q130 112 224 46" fill="none" stroke="#cf6d43" strokeWidth="4" strokeDasharray="10 6" />
      <circle cx="82" cy="125" r="7" fill="#cf6d43" />
      <circle cx="156" cy="92" r="7" fill="#cf6d43" />
      <circle cx="224" cy="46" r="7" fill="#cf6d43" />
      <text x="28" y="156" fontSize="12" fill="#6a5846">高温: 更软</text>
      <text x="176" y="156" fontSize="12" fill="#6a5846">低温: 更尖</text>
    </svg>
  );
}

export default function FisherVsDirichletComparison() {
  const figureCols = {
    display: "grid",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    gap: 16,
    marginTop: 18,
  };

  return (
    <div
      style={{
        width: "100%",
        maxWidth: 1360,
        margin: "0 auto",
        padding: 24,
        boxSizing: "border-box",
        background:
          "radial-gradient(circle at top left, #fff9ee 0%, #f7f2e8 48%, #efe7d7 100%)",
        color: "#1f1b16",
        fontFamily: '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif',
      }}
    >
      <div
        style={{
          background: "rgba(255,255,255,0.76)",
          border: "1px solid #ddd2bf",
          borderRadius: 28,
          padding: 24,
          boxSizing: "border-box",
          boxShadow: "0 12px 30px rgba(80, 60, 30, 0.06)",
        }}
      >
        <div style={{ display: "grid", gridTemplateColumns: "1.15fr 0.85fr", gap: 20, alignItems: "start" }}>
          <div>
            <div style={{ fontSize: 34, fontWeight: 800, letterSpacing: "-0.03em" }}>四种 Flow 的直觉对比</div>
            <div style={{ marginTop: 10, fontSize: 16, lineHeight: 1.8, color: "#5d554b" }}>
              上面先看形象图。下面每一列都对应同一方法的单独公式推导，里面同时给原理公式和数字例子。
            </div>
          </div>
          <div
            style={{
              background: "#f8f2e6",
              border: "1px solid #e7dac3",
              borderRadius: 20,
              padding: 18,
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 800, marginBottom: 8 }}>你看图时只要盯住这 3 件事</div>
            <div style={{ fontSize: 13.5, lineHeight: 1.8, color: "#5a5145" }}>
              1. 中间状态是不是始终还能解释成“概率分布”。<br />
              2. 它是在原来的概率坐标里走，还是先换坐标再走。<br />
              3. 它更偏理论几何，还是更偏高 K 的工程稳定性。
            </div>
          </div>
        </div>

        <div style={figureCols}>
          <FigureCard
            title="Gaussian FM"
            accent="#365b92"
            bg="#f3f6fc"
            border="#c8d6ef"
            subtitle="在普通欧氏空间里，从高斯噪声连续走向目标向量。"
          >
            <GaussianFigure />
          </FigureCard>

          <FigureCard
            title="Dirichlet"
            accent="#8c541f"
            bg="#fdf5ea"
            border="#ebc79f"
            subtitle="直接在概率三角形里把质量推向目标顶点。"
          >
            <DirichletFigure />
          </FigureCard>

          <FigureCard
            title="Fisher"
            accent="#156260"
            bg="#edf9f6"
            border="#bfe3da"
            subtitle="先换到球面坐标，再沿球面的最短弧线走。"
          >
            <FisherFigure />
          </FigureCard>

          <FigureCard
            title="Gumbel-Softmax"
            accent="#a24d2e"
            bg="#fdf1ed"
            border="#e8c2b4"
            subtitle="用噪声和温度退火，把软分类逐渐变成接近硬分类。"
          >
            <GumbelFigure />
          </FigureCard>
        </div>

        <div style={{ marginTop: 30, fontSize: 28, fontWeight: 800, letterSpacing: "-0.02em" }}>下面开始对应推公式</div>
        <div style={{ marginTop: 8, fontSize: 15, lineHeight: 1.8, color: "#5d554b" }}>
          下面四列和上面一一对应。统一采用 3 类例子，目标类别设成第 2 类，也就是目标 one-hot
          `e₂ = [0, 1, 0]`。
        </div>

        <div style={{ ...figureCols, marginTop: 16 }}>
          <FormulaCard
            title="Gaussian FM"
            accent="#365b92"
            bg="#f3f6fc"
            border="#c8d6ef"
            principle={[
              "状态空间: 普通欧氏空间 R^3",
              "取起点 z0, 终点 z1",
              "线性路径: z_t = (1-s) z0 + s z1",
              "这里 s ∈ [0,1]",
            ]}
            example={[
              "取 z0 = [0.2, -1.0, 0.5]",
              "取 z1 = e₂ = [0, 1, 0]",
              "取 s = 0.25",
              "",
              "z_t = 0.75*z0 + 0.25*z1",
              "    = [0.15, -0.50, 0.375]",
              "",
              "检查:",
              "第 2 项 = -0.50",
              "总和 = 0.025",
            ]}
            diff="中间状态只是普通连续向量，不保证非负，也不保证总和为 1。所以它不像“类别概率分配”，更像在连续编码空间里做 flow。"
          />

          <FormulaCard
            title="Dirichlet"
            accent="#8c541f"
            bg="#fdf5ea"
            border="#ebc79f"
            principle={[
              "目标类别 c = 2",
              "目标 one-hot: e₂ = [0,1,0]",
              "条件路径: x_t ~ Dir(1 + t e₂)",
              "",
              "如果 alpha = [a1,a2,a3]",
              "则均值 E[x_t] = alpha / (a1+a2+a3)",
            ]}
            example={[
              "取 K = 3, t = 4",
              "alpha = 1 + 4*e₂ = [1,5,1]",
              "alpha0 = 7",
              "",
              "E[x_t] = [1/7, 5/7, 1/7]",
              "       ≈ [0.143, 0.714, 0.143]",
              "",
              "例如一次采样可接近",
              "[0.10, 0.80, 0.10]",
            ]}
            diff="和 Gaussian 最大的区别是：它从一开始就把状态定义成合法概率分布。问题在于同一个 t 在不同 K 下不再代表同样尖锐的状态。"
          />

          <FormulaCard
            title="Fisher"
            accent="#156260"
            bg="#edf9f6"
            border="#bfe3da"
            principle={[
              "先在概率空间里取 p",
              "开方映射: u = sqrt(p)",
              "于是 sum(u_i^2) = 1, 落到球面上",
              "再在球面上做最短弧线插值",
              "最后映回: p_t = u_t^2",
            ]}
            example={[
              "取 p0 = [1/3,1/3,1/3]",
              "取 p1 = e₂ = [0,1,0]",
              "",
              "u0 = sqrt(p0) ≈ [0.577,0.577,0.577]",
              "u1 = sqrt(p1) = [0,1,0]",
              "",
              "取中间 s = 0.5",
              "球面插值后可得到近似",
              "u_t ≈ [0.325, 0.888, 0.325]",
              "",
              "映回:",
              "p_t = u_t^2 ≈ [0.106,0.789,0.106]",
            ]}
            diff="和 Gaussian 不同，它不是在普通平直空间里走，而是先换到更适合概率分布的几何坐标里，再沿那个几何里的最短路走。"
          />

          <FormulaCard
            title="Gumbel-Softmax"
            accent="#a24d2e"
            bg="#fdf1ed"
            border="#e8c2b4"
            principle={[
              "先有 logits",
              "再加 Gumbel noise",
              "最后做带温度的 softmax:",
              "x_t[j] = softmax((logit_j + g_j)/tau)",
              "",
              "tau 大 -> 分布更软",
              "tau 小 -> 分布更尖",
            ]}
            example={[
              "取 logits = [0.3, 1.1, -0.2]",
              "取 noise  = [0.1, 0.0, -0.4]",
              "相加得    = [0.4, 1.1, -0.6]",
              "",
              "tau = 1.0 时:",
              "x_t ≈ [0.296, 0.595, 0.109]",
              "",
              "tau = 0.3 时:",
              "x_t ≈ [0.088, 0.909, 0.003]",
            ]}
            diff="和 Gaussian 不同，它不会把中间状态变成任意连续向量，而是始终保持成合法概率分布，只是通过温度让它从“软分类”逐渐变成“硬分类”。"
          />
        </div>
      </div>
    </div>
  );
}
