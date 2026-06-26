import { useState } from "react";

const PATHS = [
  { key: "cropped_rgb", label: "Cropped RGB", color: "#7F77DD", bg: "#EEEDFE", border: "#AFA9EC", textDark: "#3C3489" },
  { key: "cropped_mask", label: "Cropped mask", color: "#1D9E75", bg: "#E1F5EE", border: "#5DCAA5", textDark: "#085041" },
  { key: "full_rgb", label: "Full RGB", color: "#378ADD", bg: "#E6F1FB", border: "#85B7EB", textDark: "#0C447C" },
  { key: "full_mask", label: "Full mask", color: "#BA7517", bg: "#FAEEDA", border: "#EF9F27", textDark: "#633806" },
];

const STEPS = [
  {
    id: "raw",
    title: "用户原始输入",
    subtitle: "2 个东西",
    detail: "用户只需提供原图 RGB 和一个二值 mask（标出目标物体）。这两个输入将被 preprocessor 派生出 4 路。",
    dims: {
      cropped_rgb: null,
      cropped_mask: null,
      full_rgb: null,
      full_mask: null,
    },
    rawInput: true,
  },
  {
    id: "preprocess",
    title: "Preprocessor",
    subtitle: "派生 4 路",
    detail: "从原始输入派生出 4 路：cropped 路按 mask bbox 裁出物体并抹黑背景；full 路保留整张图。所有路 pad 成正方形后 Resize 到 518×518。mask 此时仍为 1 通道。",
    dims: {
      cropped_rgb: "(B, 3, 518, 518)",
      cropped_mask: "(B, 1, 518, 518)",
      full_rgb: "(B, 3, 518, 518)",
      full_mask: "(B, 1, 518, 518)",
    },
  },
  {
    id: "repeat",
    title: "Mask repeat",
    subtitle: "1 通道 → 3 通道",
    detail: "DINOv2 的 patch embedding 是 Conv2d(in_channels=3)，写死吃 3 通道。mask 只有 1 通道，所以 repeat(1,3,1,1) 复制成 3 通道。DINOv2 不知道也不在乎这是 mask——对它来说就是一张灰度图。",
    dims: {
      cropped_rgb: "(B, 3, 518, 518)",
      cropped_mask: "(B, 3, 518, 518)",
      full_rgb: "(B, 3, 518, 518)",
      full_mask: "(B, 3, 518, 518)",
    },
    highlight: ["cropped_mask", "full_mask"],
  },
  {
    id: "patch",
    title: "Patch embedding",
    subtitle: "切 patch + 投影",
    detail: "把 518×518 切成 14×14 的小方块：518÷14=37，共 37×37=1369 个 patch。每个 patch 展平后 Linear(588→768)，再拼上 1 个可学习的 CLS token = 1370 个 token。",
    dims: {
      cropped_rgb: "(B, 1370, 768)",
      cropped_mask: "(B, 1370, 768)",
      full_rgb: "(B, 1370, 768)",
      full_mask: "(B, 1370, 768)",
    },
    math: "518÷14 = 37 → 37×37 = 1369 patch → +1 CLS = 1370",
  },
  {
    id: "transformer",
    title: "12 层 Transformer",
    subtitle: "DINOv2 主干",
    detail: "12 层标准 ViT Block（Self-Attention + FFN），形状不变。token 的语义从底层（边缘、颜色）逐渐抽象为高层（物体类别、边界）。整个 DINOv2 权重冻结，不参与训练。",
    dims: {
      cropped_rgb: "(B, 1370, 768)",
      cropped_mask: "(B, 1370, 768)",
      full_rgb: "(B, 1370, 768)",
      full_mask: "(B, 1370, 768)",
    },
    frozen: true,
  },
  {
    id: "projection",
    title: "Projection",
    subtitle: "768 → D",
    detail: "每路各一个独立的 projection net：LayerNorm(768) → FeedForward(768→4D→D)。把 DINOv2 的 768 维对齐到 Flow Transformer 的隐藏维度 D。这部分是可训练的。",
    dims: {
      cropped_rgb: "(B, 1370, D)",
      cropped_mask: "(B, 1370, D)",
      full_rgb: "(B, 1370, D)",
      full_mask: "(B, 1370, D)",
    },
    highlight: ["cropped_rgb", "cropped_mask", "full_rgb", "full_mask"],
  },
  {
    id: "pathid",
    title: "+ Path-ID",
    subtitle: "可学习路别嵌入",
    detail: "每路加一个独有的可学习 D 维向量，广播加到该路所有 1370 个 token 上。让下游能分辨 cat 后的 token 来自哪一路。类似位置编码，但编码的是「身份」而非「位置」。梯度来自 Flow Transformer 的去噪 loss。",
    dims: {
      cropped_rgb: "(B, 1370, D)",
      cropped_mask: "(B, 1370, D)",
      full_rgb: "(B, 1370, D)",
      full_mask: "(B, 1370, D)",
    },
    pathId: true,
  },
  {
    id: "cat",
    title: "Concatenate",
    subtitle: "cat(dim=1)",
    detail: "4 路沿 token 维度拼接：1370×4 = 5480 个 token。这就是 Flow Transformer 做 cross-attention 时的完整条件序列。每个 shape token 可以同时看到所有 5480 个 condition token。",
    dims: null,
    concat: true,
    finalDim: "(B, 5480, D)",
  },
];

function Arrow({ vertical = false }) {
  if (vertical) {
    return (
      <div style={{ display: "flex", justifyContent: "center", padding: "6px 0", color: "#B4B2A9" }}>
        <svg width="16" height="20" viewBox="0 0 16 20">
          <path d="M8 0 L8 14 M3 10 L8 16 L13 10" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", alignItems: "center", padding: "0 2px", color: "#B4B2A9" }}>
      <svg width="20" height="16" viewBox="0 0 20 16">
        <path d="M0 8 L14 8 M10 3 L16 8 L10 13" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  );
}

function DimBadge({ dim, highlight, pathInfo }) {
  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', monospace",
      fontSize: 12,
      padding: "5px 10px",
      borderRadius: 6,
      background: highlight ? pathInfo.bg : "#F1EFE8",
      color: highlight ? pathInfo.textDark : "#5F5E5A",
      border: `1px solid ${highlight ? pathInfo.border : "#D3D1C7"}`,
      textAlign: "center",
      lineHeight: 1.4,
      transition: "all 0.25s ease",
    }}>
      {dim}
    </div>
  );
}

function PathIdBadge({ pathInfo, index }) {
  return (
    <div style={{
      fontSize: 11,
      fontFamily: "'IBM Plex Mono', monospace",
      padding: "3px 8px",
      borderRadius: 6,
      background: pathInfo.bg,
      color: pathInfo.textDark,
      border: `1px dashed ${pathInfo.border}`,
      textAlign: "center",
      marginTop: 4,
    }}>
      + path_id_{index}
    </div>
  );
}

function TokenBar({ pathInfo, count = 20, height = 16 }) {
  return (
    <div style={{ display: "flex", gap: 1, height, alignItems: "flex-end", marginTop: 6 }}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} style={{
          flex: 1,
          height: height,
          background: pathInfo.border,
          borderRadius: 1.5,
          minWidth: 1,
          opacity: 0.7,
        }} />
      ))}
    </div>
  );
}

function ConcatBar() {
  const count = 14;
  return (
    <div style={{
      background: "#fff",
      border: "1px solid #D3D1C7",
      borderRadius: 10,
      padding: "14px 16px 10px",
    }}>
      <div style={{
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 13,
        color: "#5F5E5A",
        marginBottom: 8,
        fontWeight: 500,
      }}>
        (B, 5480, D)
      </div>
      <div style={{ display: "flex", gap: 3, height: 22, alignItems: "flex-end" }}>
        {PATHS.map((p) => (
          <div key={p.key} style={{ display: "flex", gap: 1, flex: 1 }}>
            {Array.from({ length: count }).map((_, i) => (
              <div key={i} style={{
                flex: 1, height: 22,
                background: p.border,
                borderRadius: 1.5,
                minWidth: 0,
                opacity: 0.75,
              }} />
            ))}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 3, marginTop: 6 }}>
        {PATHS.map((p) => (
          <div key={p.key} style={{
            flex: 1,
            textAlign: "center",
            fontSize: 10,
            color: p.textDark,
            fontFamily: "'IBM Plex Mono', monospace",
          }}>
            1370
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SAM3DImageEncoder() {
  const [activeStep, setActiveStep] = useState(0);
  const step = STEPS[activeStep];

  return (
    <div style={{
      fontFamily: "'Instrument Sans', 'Helvetica Neue', sans-serif",
      maxWidth: 720,
      margin: "0 auto",
      padding: "2rem 0",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />

      <div style={{ marginBottom: 28 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600, margin: "0 0 4px", color: "#2C2C2A", letterSpacing: -0.3 }}>
          Image encoder — 4 路维度变换
        </h2>
        <p style={{ fontSize: 13, color: "#888780", margin: 0 }}>
          点击每一步查看维度变化
        </p>
      </div>

      <div style={{
        display: "flex",
        gap: 6,
        flexWrap: "wrap",
        marginBottom: 24,
      }}>
        {STEPS.map((s, i) => (
          <button
            key={s.id}
            onClick={() => setActiveStep(i)}
            style={{
              padding: "6px 12px",
              borderRadius: 8,
              border: activeStep === i ? "1.5px solid #534AB7" : "1px solid #D3D1C7",
              background: activeStep === i ? "#EEEDFE" : "transparent",
              color: activeStep === i ? "#3C3489" : "#888780",
              fontSize: 12,
              fontWeight: 500,
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "all 0.15s ease",
            }}
          >
            {s.title}
          </button>
        ))}
      </div>

      <div style={{
        background: "#FAFAF8",
        borderRadius: 14,
        padding: "20px 20px 16px",
        border: "1px solid #E8E7E3",
        marginBottom: 16,
      }}>
        <div style={{ marginBottom: 12 }}>
          <span style={{
            fontSize: 15,
            fontWeight: 600,
            color: "#2C2C2A",
          }}>
            {step.title}
          </span>
          <span style={{
            fontSize: 13,
            color: "#888780",
            marginLeft: 10,
          }}>
            {step.subtitle}
          </span>
        </div>

        {step.rawInput ? (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <div style={{
              flex: 1, minWidth: 160,
              padding: "14px 16px",
              borderRadius: 10,
              background: "#fff",
              border: "1px solid #D3D1C7",
            }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "#2C2C2A", marginBottom: 6 }}>Image (原图)</div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, color: "#5F5E5A" }}>(H, W, 3)</div>
              <div style={{ fontSize: 11, color: "#888780", marginTop: 4 }}>任意尺寸 RGB</div>
            </div>
            <div style={{
              flex: 1, minWidth: 160,
              padding: "14px 16px",
              borderRadius: 10,
              background: "#fff",
              border: "1px solid #D3D1C7",
            }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "#2C2C2A", marginBottom: 6 }}>Mask (二值)</div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, color: "#5F5E5A" }}>(H, W)</div>
              <div style={{ fontSize: 11, color: "#888780", marginTop: 4 }}>0/1 标出目标物体</div>
            </div>
          </div>
        ) : step.concat ? (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 8 }}>
              {PATHS.map((p, i) => (
                <div key={p.key} style={{
                  padding: "10px 12px",
                  borderRadius: 8,
                  background: p.bg,
                  border: `1px solid ${p.border}`,
                }}>
                  <div style={{ fontSize: 12, fontWeight: 500, color: p.textDark }}>{p.label}</div>
                  <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: p.textDark, marginTop: 2 }}>(B, 1370, D)</div>
                  <TokenBar pathInfo={p} count={16} height={12} />
                </div>
              ))}
            </div>
            <Arrow vertical />
            <ConcatBar />
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {PATHS.map((p, i) => {
              const dim = step.dims[p.key];
              const hl = step.highlight?.includes(p.key);
              return (
                <div key={p.key} style={{
                  padding: "10px 12px",
                  borderRadius: 8,
                  background: "#fff",
                  border: `1px solid ${hl ? p.border : "#E8E7E3"}`,
                  transition: "all 0.2s ease",
                }}>
                  <div style={{
                    fontSize: 12,
                    fontWeight: 500,
                    color: p.textDark,
                    marginBottom: 6,
                  }}>
                    {p.label}
                  </div>
                  {dim && <DimBadge dim={dim} highlight={hl} pathInfo={p} />}
                  {step.id !== "preprocess" && step.id !== "repeat" && dim && (
                    <TokenBar pathInfo={p} count={16} height={14} />
                  )}
                  {step.pathId && <PathIdBadge pathInfo={p} index={i} />}
                </div>
              );
            })}
          </div>
        )}

        {step.math && (
          <div style={{
            marginTop: 12,
            padding: "8px 12px",
            borderRadius: 8,
            background: "#EEEDFE",
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 12,
            color: "#3C3489",
            textAlign: "center",
          }}>
            {step.math}
          </div>
        )}

        {step.frozen && (
          <div style={{
            marginTop: 12,
            padding: "8px 12px",
            borderRadius: 8,
            background: "#FCEBEB",
            fontSize: 12,
            color: "#791F1F",
            textAlign: "center",
          }}>
            DINOv2 全部冻结 — requires_grad = False
          </div>
        )}
      </div>

      <div style={{
        padding: "14px 16px",
        borderRadius: 10,
        background: "#F5F5F2",
        fontSize: 13,
        color: "#5F5E5A",
        lineHeight: 1.7,
      }}>
        {step.detail}
      </div>

      <div style={{
        marginTop: 24,
        padding: "16px",
        borderRadius: 10,
        border: "1px solid #E8E7E3",
        background: "#fff",
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#2C2C2A", marginBottom: 10 }}>
          完整维度对照表
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{
            width: "100%",
            borderCollapse: "collapse",
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11,
          }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #E8E7E3" }}>
                <th style={{ textAlign: "left", padding: "6px 8px", color: "#888780", fontWeight: 500, fontFamily: "'Instrument Sans', sans-serif" }}>步骤</th>
                {PATHS.map(p => (
                  <th key={p.key} style={{ textAlign: "center", padding: "6px 4px", color: p.textDark, fontWeight: 500, fontSize: 10, fontFamily: "'Instrument Sans', sans-serif" }}>{p.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {STEPS.filter(s => !s.rawInput && !s.concat).map((s, idx) => (
                <tr key={s.id} style={{
                  borderBottom: "1px solid #F1EFE8",
                  background: activeStep === idx + 1 ? "#FAFAF8" : "transparent",
                }}>
                  <td style={{ padding: "6px 8px", color: "#5F5E5A", fontFamily: "'Instrument Sans', sans-serif", fontSize: 12 }}>{s.title}</td>
                  {PATHS.map(p => (
                    <td key={p.key} style={{
                      textAlign: "center",
                      padding: "6px 4px",
                      color: s.highlight?.includes(p.key) ? p.textDark : "#888780",
                      fontWeight: s.highlight?.includes(p.key) ? 500 : 400,
                    }}>
                      {s.dims[p.key]}
                    </td>
                  ))}
                </tr>
              ))}
              <tr style={{ background: "#FAFAF8" }}>
                <td style={{ padding: "6px 8px", color: "#5F5E5A", fontFamily: "'Instrument Sans', sans-serif", fontSize: 12 }}>cat output</td>
                <td colSpan={4} style={{ textAlign: "center", padding: "6px 4px", color: "#3C3489", fontWeight: 500 }}>(B, 5480, D)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
