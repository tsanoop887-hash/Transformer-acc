import React, { useState, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from "recharts";
import { Cpu, HardDrive, Gauge, AlertTriangle, Zap, Layers } from "lucide-react";

// ---------- theme ----------
const COLOR = {
  bg: "#0A1420",
  panel: "#0F1F2E",
  panelAlt: "#132738",
  line: "rgba(148,197,255,0.14)",
  lineStrong: "rgba(148,197,255,0.28)",
  fg: "#E7F1FF",
  fgMuted: "#7E97B2",
  fgFaint: "#4C6480",
  cyan: "#6FE3D0",
  amber: "#F3A85C",
  violet: "#9C93F5",
  rose: "#F17B85",
};

const DTYPE_BYTES = { fp32: 4, fp16: 2, int8: 1 };

const HARDWARE_PRESETS = [
  { id: "edge", name: "Edge / laptop GPU", vramGB: 8, bandwidthGBs: 400, flopsTFLOPS: 20 },
  { id: "consumer", name: "Consumer — RTX 4090", vramGB: 24, bandwidthGBs: 1000, flopsTFLOPS: 82 },
  { id: "a100", name: "Datacenter — A100 80GB", vramGB: 80, bandwidthGBs: 2000, flopsTFLOPS: 312 },
  { id: "h100", name: "Datacenter — H100 SXM", vramGB: 80, bandwidthGBs: 3350, flopsTFLOPS: 990 },
  { id: "custom", name: "Custom", vramGB: 24, bandwidthGBs: 1000, flopsTFLOPS: 82 },
];

const VOCAB_OPTIONS = [16000, 32000, 50257, 100000, 152000];
const HIDDEN_OPTIONS = [256, 512, 768, 1024, 1536, 2048, 4096];
const HEADS_OPTIONS = [4, 8, 16, 32, 64];
const FFN_MULT_OPTIONS = [2, 3, 4, 8];
const PROMPT_STEPS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];

function formatBytes(b) {
  if (b < 1e6) return (b / 1e3).toFixed(1) + " KB";
  if (b < 1e9) return (b / 1e6).toFixed(1) + " MB";
  return (b / 1e9).toFixed(2) + " GB";
}
function formatNum(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}
function formatMs(ms) {
  if (ms < 1) return (ms * 1000).toFixed(0) + " µs";
  if (ms < 1000) return ms.toFixed(1) + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

// ---------- analytical model ----------
// Standard roofline approximation: time = max(compute-bound time, memory-bound time).
// Decode is treated as memory-bandwidth bound (reading all weights + KV cache every step),
// which is the well-known bottleneck for autoregressive, batch-1 generation.
function computeModel(cfg, hw) {
  const headDim = cfg.hiddenDim / cfg.nHeads;
  const ffnDim = cfg.hiddenDim * cfg.ffnMult;
  const bpp = DTYPE_BYTES[cfg.dtype];

  const embedParams = cfg.vocabSize * cfg.hiddenDim;
  const perLayerParams = 4 * cfg.hiddenDim * cfg.hiddenDim + 2 * cfg.hiddenDim * ffnDim;
  const layerParams = perLayerParams * cfg.nLayers;
  const outProjParams = cfg.hiddenDim * cfg.vocabSize;
  const totalParams = embedParams + layerParams + outProjParams;
  const weightsBytes = totalParams * bpp;

  const kvBytesPerToken = 2 * cfg.nLayers * cfg.hiddenDim * bpp;
  const finalContext = cfg.promptLen + cfg.genTokens;
  const kvCacheBytes = kvBytesPerToken * finalContext * cfg.batchSize;
  const activationBytes = cfg.batchSize * cfg.promptLen * cfg.hiddenDim * bpp * 6;

  const bandwidthBps = hw.bandwidthGBs * 1e9;
  const flopsPs = hw.flopsTFLOPS * 1e12;

  function prefillTimeMs(S, kernel) {
    const flops = S * 2 * totalParams + 2 * cfg.nLayers * cfg.hiddenDim * S * S;
    const computeTime = flops / flopsPs;
    const attnBytes =
      kernel === "gemm"
        ? cfg.nLayers * cfg.nHeads * S * S * bpp * 4
        : cfg.nLayers * cfg.nHeads * S * headDim * bpp * 8;
    const memTime = (weightsBytes + attnBytes) / bandwidthBps;
    return Math.max(computeTime, memTime) * 1000;
  }

  function decodeTokensPerSec(contextLen) {
    const kvBytes = kvBytesPerToken * contextLen * cfg.batchSize;
    const bytesPerStep = weightsBytes + kvBytes;
    return bandwidthBps / bytesPerStep;
  }

  const prefillTimeCurrent = prefillTimeMs(cfg.promptLen, cfg.kernel);
  const decodeTPS = decodeTokensPerSec(finalContext);
  const totalGenMs = prefillTimeCurrent + (cfg.genTokens / decodeTPS) * 1000;
  const vramBytes = hw.vramGB * 1e9;
  const totalMemoryBytes = weightsBytes + kvCacheBytes + activationBytes;

  return {
    headDim, ffnDim, totalParams, weightsBytes, kvCacheBytes, activationBytes,
    totalMemoryBytes, vramBytes, prefillTimeCurrent, decodeTPS, totalGenMs,
    prefillTimeMs, decodeTokensPerSec, finalContext,
  };
}

// ---------- small UI atoms ----------
function Panel({ title, icon, children }) {
  return (
    <div style={{ background: COLOR.panel, border: `1px solid ${COLOR.line}`, borderRadius: 10, padding: "16px 18px" }}>
      {title && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
          {icon}
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: COLOR.fgMuted }}>
            {title}
          </span>
        </div>
      )}
      {children}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: COLOR.fgMuted, marginBottom: 6, display: "flex", justifyContent: "space-between" }}>
        {label}
      </div>
      {children}
    </div>
  );
}

const selectStyle = {
  width: "100%", background: COLOR.panelAlt, color: COLOR.fg,
  border: `1px solid ${COLOR.lineStrong}`, borderRadius: 6,
  padding: "7px 8px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12.5,
  outline: "none",
};
const rangeStyle = { width: "100%", accentColor: COLOR.cyan };

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: COLOR.panel, border: `1px solid ${COLOR.line}`, borderRadius: 10, padding: "14px 16px", flex: 1, minWidth: 150 }}>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, letterSpacing: "0.08em", textTransform: "uppercase", color: COLOR.fgMuted }}>{label}</div>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 600, color: accent || COLOR.fg, marginTop: 4 }}>{value}</div>
      {sub && <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: COLOR.fgFaint, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ---------- VRAM gauge (signature element) ----------
function VramGauge({ weightsBytes, kvCacheBytes, activationBytes, vramBytes }) {
  const total = weightsBytes + kvCacheBytes + activationBytes;
  const overflow = total > vramBytes;
  const gaugeMax = Math.max(vramBytes, total) * 1.08;
  const pct = (b) => (b / gaugeMax) * 100;
  const capacityPct = pct(vramBytes);

  const segments = [
    { label: "Model weights", bytes: weightsBytes, color: COLOR.cyan },
    { label: "KV cache", bytes: kvCacheBytes, color: COLOR.amber },
    { label: "Activations (est.)", bytes: activationBytes, color: COLOR.violet },
  ];

  return (
    <div>
      <div style={{ position: "relative", height: 34, background: COLOR.panelAlt, borderRadius: 6, overflow: "hidden", border: `1px solid ${COLOR.lineStrong}` }}>
        <div style={{ display: "flex", height: "100%" }}>
          {segments.map((s) => (
            <div key={s.label} style={{ width: `${pct(s.bytes)}%`, background: s.color, height: "100%" }} title={s.label} />
          ))}
        </div>
        {/* tick marks */}
        {[0, 25, 50, 75, 100].map((t) => (
          <div key={t} style={{ position: "absolute", left: `${t}%`, top: 0, bottom: 0, width: 1, background: "rgba(0,0,0,0.25)" }} />
        ))}
        {/* capacity marker */}
        <div style={{ position: "absolute", left: `${Math.min(capacityPct, 100)}%`, top: -3, bottom: -3, width: 2, background: overflow ? COLOR.rose : COLOR.fg }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: COLOR.fgFaint }}>
        <span>0 GB</span>
        <span>{(gaugeMax / 1e9).toFixed(0)} GB</span>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12 }}>
        {segments.map((s) => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11.5, color: COLOR.fgMuted }}>
            <span style={{ width: 9, height: 9, background: s.color, borderRadius: 2, display: "inline-block" }} />
            {s.label}: <span style={{ color: COLOR.fg }}>{formatBytes(s.bytes)}</span>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12, padding: "9px 12px", borderRadius: 6, display: "flex", alignItems: "center", gap: 8,
        background: overflow ? "rgba(241,123,133,0.1)" : "rgba(111,227,208,0.08)",
        border: `1px solid ${overflow ? "rgba(241,123,133,0.35)" : "rgba(111,227,208,0.25)"}` }}>
        {overflow ? <AlertTriangle size={15} color={COLOR.rose} /> : <Gauge size={15} color={COLOR.cyan} />}
        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11.5, color: overflow ? COLOR.rose : COLOR.cyan }}>
          {overflow
            ? `Exceeds selected VRAM by ${formatBytes(total - vramBytes)}`
            : `Fits with ${formatBytes(vramBytes - total)} of VRAM to spare`}
        </span>
      </div>
    </div>
  );
}

// ---------- chart tooltip ----------
function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div style={{ background: COLOR.panelAlt, border: `1px solid ${COLOR.lineStrong}`, borderRadius: 6, padding: "8px 10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11.5 }}>
      <div style={{ color: COLOR.fgMuted, marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>{p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}{unit}</div>
      ))}
    </div>
  );
}

// ---------- main component ----------
export default function TransformerCostExplorer() {
  const [vocabSize, setVocabSize] = useState(32000);
  const [hiddenDim, setHiddenDim] = useState(1024);
  const [nHeads, setNHeads] = useState(16);
  const [nLayers, setNLayers] = useState(24);
  const [ffnMult, setFfnMult] = useState(4);
  const [dtype, setDtype] = useState("fp16");
  const [kernel, setKernel] = useState("flash");
  const [promptLen, setPromptLen] = useState(2048);
  const [genTokens, setGenTokens] = useState(256);
  const [batchSize, setBatchSize] = useState(1);
  const [hwPreset, setHwPreset] = useState("consumer");
  const [customHw, setCustomHw] = useState({ vramGB: 24, bandwidthGBs: 1000, flopsTFLOPS: 82 });
  const [tab, setTab] = useState("memory");

  const headOptions = HEADS_OPTIONS.filter((h) => hiddenDim % h === 0);
  const hw = hwPreset === "custom" ? customHw : HARDWARE_PRESETS.find((h) => h.id === hwPreset);

  const cfg = { vocabSize, hiddenDim, nHeads, nLayers, ffnMult, dtype, kernel, promptLen, genTokens, batchSize };
  const model = useMemo(() => computeModel(cfg, hw), [vocabSize, hiddenDim, nHeads, nLayers, ffnMult, dtype, kernel, promptLen, genTokens, batchSize, hw]);

  const memoryByDtype = useMemo(() => {
    return ["fp32", "fp16", "int8"].map((dt) => {
      const m = computeModel({ ...cfg, dtype: dt }, hw);
      return { dtype: dt.toUpperCase(), "Model weights": m.weightsBytes / 1e9, "KV cache": m.kvCacheBytes / 1e9, "Activations": m.activationBytes / 1e9 };
    });
  }, [vocabSize, hiddenDim, nLayers, ffnMult, nHeads, promptLen, genTokens, batchSize, hw]);

  const prefillByLength = useMemo(() => {
    const lens = PROMPT_STEPS.filter((s) => s <= Math.max(promptLen * 4, 4096));
    return lens.map((S) => ({
      len: S >= 1024 ? `${S / 1024}K` : String(S),
      Flash: model.prefillTimeMs(S, "flash"),
      GEMM: model.prefillTimeMs(S, "gemm"),
    }));
  }, [model, promptLen]);

  const decodeThroughput = useMemo(() => {
    const points = 12;
    const out = [];
    for (let i = 0; i <= points; i++) {
      const ctx = Math.round(promptLen + (genTokens * i) / points);
      out.push({ step: ctx, "Tokens/sec": model.decodeTokensPerSec(ctx) });
    }
    return out;
  }, [model, promptLen, genTokens]);

  return (
    <div style={{
      background: COLOR.bg, minHeight: 640, padding: 24, fontFamily: "'Inter', sans-serif",
      backgroundImage: `linear-gradient(${COLOR.line} 1px, transparent 1px), linear-gradient(90deg, ${COLOR.line} 1px, transparent 1px)`,
      backgroundSize: "28px 28px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
        input[type=range] { height: 4px; }
        select:focus { outline: 1px solid ${COLOR.cyan}; }
      `}</style>

      {/* header */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, letterSpacing: "0.15em", color: COLOR.cyan, textTransform: "uppercase" }}>
          Inference Cost Explorer
        </div>
        <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 600, color: COLOR.fg, marginTop: 2 }}>
          Transformer memory &amp; latency, before you provision hardware
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20, alignItems: "start" }}>
        {/* ---------- sidebar ---------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Panel title="Architecture" icon={<Layers size={13} color={COLOR.fgMuted} />}>
            <Field label="Vocab size">
              <select style={selectStyle} value={vocabSize} onChange={(e) => setVocabSize(+e.target.value)}>
                {VOCAB_OPTIONS.map((v) => <option key={v} value={v}>{formatNum(v)}</option>)}
              </select>
            </Field>
            <Field label="Hidden dim">
              <select style={selectStyle} value={hiddenDim} onChange={(e) => {
                const v = +e.target.value;
                setHiddenDim(v);
                if (v % nHeads !== 0) setNHeads(HEADS_OPTIONS.filter((h) => v % h === 0).slice(-1)[0] || 4);
              }}>
                {HIDDEN_OPTIONS.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </Field>
            <Field label="Attention heads">
              <select style={selectStyle} value={nHeads} onChange={(e) => setNHeads(+e.target.value)}>
                {headOptions.map((v) => <option key={v} value={v}>{v} (head dim {hiddenDim / v})</option>)}
              </select>
            </Field>
            <Field label={`Layers: ${nLayers}`}>
              <input type="range" min={1} max={64} value={nLayers} onChange={(e) => setNLayers(+e.target.value)} style={rangeStyle} />
            </Field>
            <Field label="FFN size">
              <select style={selectStyle} value={ffnMult} onChange={(e) => setFfnMult(+e.target.value)}>
                {FFN_MULT_OPTIONS.map((v) => <option key={v} value={v}>{v}× hidden ({v * hiddenDim})</option>)}
              </select>
            </Field>
          </Panel>

          <Panel title="Precision & kernel" icon={<Cpu size={13} color={COLOR.fgMuted} />}>
            <Field label="Weight dtype">
              <select style={selectStyle} value={dtype} onChange={(e) => setDtype(e.target.value)}>
                <option value="fp32">FP32</option>
                <option value="fp16">FP16</option>
                <option value="int8">INT8</option>
              </select>
            </Field>
            <Field label="Attention kernel">
              <select style={selectStyle} value={kernel} onChange={(e) => setKernel(e.target.value)}>
                <option value="flash">Flash (tiled, streaming)</option>
                <option value="gemm">GEMM (naive, materializes scores)</option>
              </select>
            </Field>
          </Panel>

          <Panel title="Workload" icon={<Zap size={13} color={COLOR.fgMuted} />}>
            <Field label={`Prompt length: ${promptLen.toLocaleString()} tok`}>
              <input type="range" min={128} max={16384} step={128} value={promptLen} onChange={(e) => setPromptLen(+e.target.value)} style={rangeStyle} />
            </Field>
            <Field label={`New tokens to generate: ${genTokens}`}>
              <input type="range" min={1} max={2048} step={1} value={genTokens} onChange={(e) => setGenTokens(+e.target.value)} style={rangeStyle} />
            </Field>
            <Field label={`Batch size: ${batchSize}`}>
              <input type="range" min={1} max={64} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} style={rangeStyle} />
            </Field>
          </Panel>

          <Panel title="Hardware" icon={<HardDrive size={13} color={COLOR.fgMuted} />}>
            <Field label="Preset">
              <select style={selectStyle} value={hwPreset} onChange={(e) => setHwPreset(e.target.value)}>
                {HARDWARE_PRESETS.map((h) => <option key={h.id} value={h.id}>{h.name}</option>)}
              </select>
            </Field>
            {hwPreset === "custom" ? (
              <>
                <Field label={`VRAM: ${customHw.vramGB} GB`}>
                  <input type="range" min={4} max={192} value={customHw.vramGB} onChange={(e) => setCustomHw({ ...customHw, vramGB: +e.target.value })} style={rangeStyle} />
                </Field>
                <Field label={`Bandwidth: ${customHw.bandwidthGBs} GB/s`}>
                  <input type="range" min={100} max={4000} step={50} value={customHw.bandwidthGBs} onChange={(e) => setCustomHw({ ...customHw, bandwidthGBs: +e.target.value })} style={rangeStyle} />
                </Field>
                <Field label={`Compute: ${customHw.flopsTFLOPS} TFLOPS`}>
                  <input type="range" min={5} max={1200} step={5} value={customHw.flopsTFLOPS} onChange={(e) => setCustomHw({ ...customHw, flopsTFLOPS: +e.target.value })} style={rangeStyle} />
                </Field>
              </>
            ) : (
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: COLOR.fgFaint, lineHeight: 1.8 }}>
                {hw.vramGB} GB · {hw.bandwidthGBs.toLocaleString()} GB/s · {hw.flopsTFLOPS} TFLOPS
              </div>
            )}
          </Panel>
        </div>

        {/* ---------- main ---------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <StatCard label="Parameters" value={formatNum(model.totalParams)} sub={`head dim ${model.headDim}`} />
            <StatCard label="Time to first token" value={formatMs(model.prefillTimeCurrent)} sub={`${promptLen.toLocaleString()} tok prefill, ${kernel}`} />
            <StatCard label="Decode throughput" value={`${model.decodeTPS.toFixed(1)}/s`} accent={COLOR.cyan} sub={`at ${model.finalContext.toLocaleString()} tok context`} />
            <StatCard label="Total generation" value={formatMs(model.totalGenMs)} sub={`${genTokens} new tokens`} />
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            {["memory", "speed"].map((t) => (
              <button key={t} onClick={() => setTab(t)} style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: 11.5, letterSpacing: "0.08em", textTransform: "uppercase",
                padding: "8px 16px", borderRadius: 6, cursor: "pointer",
                background: tab === t ? COLOR.cyan : "transparent",
                color: tab === t ? "#06201C" : COLOR.fgMuted,
                border: `1px solid ${tab === t ? COLOR.cyan : COLOR.lineStrong}`,
              }}>{t}</button>
            ))}
          </div>

          {tab === "memory" ? (
            <>
              <Panel title="VRAM footprint vs selected hardware" icon={<Gauge size={13} color={COLOR.fgMuted} />}>
                <VramGauge weightsBytes={model.weightsBytes} kvCacheBytes={model.kvCacheBytes} activationBytes={model.activationBytes} vramBytes={model.vramBytes} />
              </Panel>

              <Panel title="Memory footprint by precision (current architecture)">
                <div style={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={memoryByDtype} barGap={4}>
                      <CartesianGrid stroke={COLOR.line} vertical={false} />
                      <XAxis dataKey="dtype" tick={{ fill: COLOR.fgMuted, fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: COLOR.lineStrong }} tickLine={false} />
                      <YAxis tick={{ fill: COLOR.fgMuted, fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: COLOR.lineStrong }} tickLine={false} label={{ value: "GB", angle: -90, position: "insideLeft", fill: COLOR.fgMuted, fontSize: 11 }} />
                      <Tooltip content={<ChartTooltip unit=" GB" />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                      <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11, color: COLOR.fgMuted }} />
                      <Bar dataKey="Model weights" stackId="a" fill={COLOR.cyan} radius={[0, 0, 0, 0]} />
                      <Bar dataKey="KV cache" stackId="a" fill={COLOR.amber} />
                      <Bar dataKey="Activations" stackId="a" fill={COLOR.violet} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </>
          ) : (
            <>
              <Panel title="Prefill time vs. prompt length — kernel comparison">
                <div style={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={prefillByLength}>
                      <CartesianGrid stroke={COLOR.line} vertical={false} />
                      <XAxis dataKey="len" tick={{ fill: COLOR.fgMuted, fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: COLOR.lineStrong }} tickLine={false} />
                      <YAxis tick={{ fill: COLOR.fgMuted, fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: COLOR.lineStrong }} tickLine={false} label={{ value: "ms", angle: -90, position: "insideLeft", fill: COLOR.fgMuted, fontSize: 11 }} />
                      <Tooltip content={<ChartTooltip unit=" ms" />} />
                      <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11, color: COLOR.fgMuted }} />
                      <Line type="monotone" dataKey="Flash" stroke={COLOR.cyan} strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="GEMM" stroke={COLOR.rose} strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: COLOR.fgFaint, marginTop: 8 }}>
                  GEMM materializes the full attention score matrix in memory each layer — its cost grows much faster with context length than Flash's tiled, streaming approach.
                </div>
              </Panel>

              <Panel title="Decode throughput as the KV cache grows">
                <div style={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={decodeThroughput}>
                      <CartesianGrid stroke={COLOR.line} vertical={false} />
                      <XAxis dataKey="step" tick={{ fill: COLOR.fgMuted, fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: COLOR.lineStrong }} tickLine={false} label={{ value: "context length (tokens)", position: "insideBottom", offset: -4, fill: COLOR.fgMuted, fontSize: 11 }} />
                      <YAxis tick={{ fill: COLOR.fgMuted, fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: COLOR.lineStrong }} tickLine={false} label={{ value: "tok/s", angle: -90, position: "insideLeft", fill: COLOR.fgMuted, fontSize: 11 }} />
                      <Tooltip content={<ChartTooltip unit=" tok/s" />} />
                      <Line type="monotone" dataKey="Tokens/sec" stroke={COLOR.amber} strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: COLOR.fgFaint, marginTop: 8 }}>
                  Decode is memory-bandwidth bound: every step re-reads all weights plus the full KV cache, so throughput falls as the conversation gets longer.
                </div>
              </Panel>
            </>
          )}

          <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: COLOR.fgFaint, lineHeight: 1.6 }}>
            Analytical roofline estimate (compute-bound vs. memory-bandwidth-bound), not a measured benchmark. Hardware figures are illustrative.
          </div>
        </div>
      </div>
    </div>
  );
}
