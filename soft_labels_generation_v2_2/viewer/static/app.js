const state = {
  files: [],
  selectedPath: null,
  entry: null,
  layer: "segmentation",
  variant: "mass",
  classIndex: 0,
  sliceIndex: 0,
  mapPayload: null,
  spaxel: null,
  selectedPixel: null,
};

const fileSelect = document.getElementById("file-select");
const layerButtons = document.getElementById("layer-buttons");
const variantButtons = document.getElementById("variant-buttons");
const classChipRow = document.getElementById("class-chip-row");
const pipe3dSelect = document.getElementById("pipe3d-select");
const toggleValidOverlay = document.getElementById("toggle-valid-overlay");
const sliceSlider = document.getElementById("slice-slider");
const sliceLabel = document.getElementById("slice-label");
const mapCanvas = document.getElementById("map-canvas");
const mapCtx = mapCanvas.getContext("2d");
const mapCaption = document.getElementById("map-caption");
const mapLegend = document.getElementById("map-legend");
const metaGrid = document.getElementById("meta-grid");
const selectionCaption = document.getElementById("selection-caption");
const selectionMeta = document.getElementById("selection-meta");
const probSvg = document.getElementById("prob-svg");
const spectrumSvg = document.getElementById("spectrum-svg");

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const abs = Math.abs(Number(value));
  if ((abs > 0 && abs < 0.01) || abs >= 10000) {
    return Number(value).toExponential(2);
  }
  return Number(value).toFixed(digits);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function createMetaCards(container, entries) {
  container.innerHTML = "";
  entries.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "meta-card";
    card.innerHTML = `<label>${label}</label><strong>${value}</strong>`;
    container.appendChild(card);
  });
}

function turboColor(value) {
  const stops = [
    [0.0, [31, 58, 95]],
    [0.25, [42, 157, 143]],
    [0.5, [137, 201, 109]],
    [0.75, [244, 162, 97]],
    [1.0, [231, 111, 81]],
  ];
  const x = Math.min(1, Math.max(0, value));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [x0, c0] = stops[i];
    const [x1, c1] = stops[i + 1];
    if (x <= x1) {
      const t = (x - x0) / (x1 - x0);
      return `rgb(${Math.round(c0[0] + (c1[0] - c0[0]) * t)}, ${Math.round(
        c0[1] + (c1[1] - c0[1]) * t,
      )}, ${Math.round(c0[2] + (c1[2] - c0[2]) * t)})`;
    }
  }
  return "rgb(231, 111, 81)";
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [255, 255, 255];
}

function drawSelectedPixel(cellW, cellH) {
  if (!state.selectedPixel) {
    return;
  }
  mapCtx.strokeStyle = "rgba(255,255,255,0.95)";
  mapCtx.lineWidth = 2;
  mapCtx.strokeRect(state.selectedPixel.x * cellW, state.selectedPixel.y * cellH, cellW, cellH);
}

function drawCategorical(payload) {
  const idx = payload.class_index;
  const ny = idx.length;
  const nx = idx[0].length;
  const cellW = mapCanvas.width / nx;
  const cellH = mapCanvas.height / ny;
  // origen inferior-izquierdo (convención astronómica): fila 0 abajo
  mapCtx.fillStyle = "#05070c";
  mapCtx.fillRect(0, 0, mapCanvas.width, mapCanvas.height);

  for (let y = 0; y < ny; y += 1) {
    const row = ny - 1 - y;
    for (let x = 0; x < nx; x += 1) {
      const c = idx[y][x];
      if (c < 0) {
        continue;
      }
      const [r, g, b] = hexToRgb(payload.colors[c]);
      let alpha = 0.25 + 0.75 * Math.min(1, payload.max_prob[y][x]);
      if (toggleValidOverlay.checked && !payload.valid[y][x]) {
        alpha *= 0.18;
      }
      mapCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
      mapCtx.fillRect(x * cellW, row * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
  if (state.selectedPixel) {
    mapCtx.strokeStyle = "rgba(255,255,255,0.95)";
    mapCtx.lineWidth = 2;
    mapCtx.strokeRect(
      state.selectedPixel.x * cellW,
      (ny - 1 - state.selectedPixel.y) * cellH,
      cellW,
      cellH,
    );
  }

  mapLegend.innerHTML = payload.class_names
    .map(
      (name, c) =>
        `<span><span class="legend-swatch" style="background:${payload.colors[c]}"></span>${name}</span>`,
    )
    .join("");
  mapLegend.innerHTML +=
    '<span style="margin-left:auto">opacidad ∝ P máx · atenuado = fuera de M_valid</span>';
}

function drawContinuous(payload) {
  const data = payload.data;
  const nanMask = payload.nan_mask;
  const ny = data.length;
  const nx = data[0].length;
  const cellW = mapCanvas.width / nx;
  const cellH = mapCanvas.height / ny;
  const span = payload.vmax - payload.vmin || 1;

  mapCtx.fillStyle = "#05070c";
  mapCtx.fillRect(0, 0, mapCanvas.width, mapCanvas.height);
  for (let y = 0; y < ny; y += 1) {
    const row = ny - 1 - y;
    for (let x = 0; x < nx; x += 1) {
      if (nanMask && nanMask[y][x]) {
        continue;
      }
      const norm = (data[y][x] - payload.vmin) / span;
      mapCtx.fillStyle = turboColor(norm);
      mapCtx.fillRect(x * cellW, row * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
  if (state.selectedPixel) {
    mapCtx.strokeStyle = "rgba(255,255,255,0.95)";
    mapCtx.lineWidth = 2;
    mapCtx.strokeRect(
      state.selectedPixel.x * cellW,
      (ny - 1 - state.selectedPixel.y) * cellH,
      cellW,
      cellH,
    );
  }

  mapLegend.innerHTML = `<span>${formatNumber(payload.vmin)}</span><div class="legend-bar"></div><span>${formatNumber(payload.vmax)}</span>`;
}

function drawMap() {
  if (!state.mapPayload) {
    mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
    return;
  }
  if (state.mapPayload.kind === "categorical") {
    drawCategorical(state.mapPayload);
  } else {
    drawContinuous(state.mapPayload);
  }
}

function renderProbBars() {
  if (!state.spaxel) {
    probSvg.innerHTML =
      '<text class="empty-state" x="50%" y="50%" text-anchor="middle">Sin spaxel seleccionado</text>';
    return;
  }
  const names = state.spaxel.class_names;
  const colors = state.entry.class_colors;
  const probs = state.spaxel.probs[state.variant] || [];
  const rows = names
    .map((name, i) => {
      const y = 18 + i * 38;
      const width = Math.max(1, 270 * Math.min(1, probs[i] || 0));
      return `
        <text class="prob-name" x="6" y="${y + 14}">${name}</text>
        <rect x="78" y="${y}" width="270" height="20" rx="4" fill="#121826" stroke="#273250" />
        <rect x="78" y="${y}" width="${width}" height="20" rx="4" fill="${colors[i]}" />
        <text class="prob-value" x="356" y="${y + 14}">${formatNumber(probs[i], 3)}</text>`;
    })
    .join("");
  probSvg.innerHTML = rows;
}

function renderSpectrum() {
  if (!state.spaxel) {
    spectrumSvg.innerHTML =
      '<text class="empty-state" x="50%" y="50%" text-anchor="middle">Sin spaxel seleccionado</text>';
    return;
  }
  const xs = state.spaxel.wave;
  const flux = state.spaxel.flux;
  const plot = { left: 52, top: 16, width: 388, height: 170 };
  let minY = Infinity;
  let maxY = -Infinity;
  flux.forEach((v) => {
    if (Number.isFinite(v)) {
      minY = Math.min(minY, v);
      maxY = Math.max(maxY, v);
    }
  });
  if (!Number.isFinite(minY)) {
    minY = 0;
    maxY = 1;
  }
  if (maxY <= minY) {
    maxY = minY + 1;
  }
  const minX = xs[0];
  const maxX = xs[xs.length - 1];
  const xScale = (v) => plot.left + ((v - minX) / (maxX - minX || 1)) * plot.width;
  const yScale = (v) => plot.top + plot.height - ((v - minY) / (maxY - minY)) * plot.height;
  // submuestrear para no inflar el SVG
  const step = Math.max(1, Math.floor(xs.length / 900));
  const pts = [];
  for (let i = 0; i < xs.length; i += step) {
    pts.push(`${xScale(xs[i]).toFixed(1)},${yScale(flux[i]).toFixed(1)}`);
  }
  const ticks = [];
  for (let i = 0; i <= 4; i += 1) {
    const frac = i / 4;
    const xv = minX + (maxX - minX) * frac;
    ticks.push(
      `<text class="tick-label" x="${plot.left + plot.width * frac}" y="${plot.top + plot.height + 18}" text-anchor="middle">${formatNumber(xv, 0)}</text>`,
    );
    const yv = minY + (maxY - minY) * (1 - frac);
    ticks.push(
      `<text class="tick-label" x="${plot.left - 8}" y="${plot.top + plot.height * frac + 4}" text-anchor="end">${formatNumber(yv, 2)}</text>`,
    );
  }
  spectrumSvg.innerHTML = `
    <line class="axis" x1="${plot.left}" y1="${plot.top + plot.height}" x2="${plot.left + plot.width}" y2="${plot.top + plot.height}" />
    <line class="axis" x1="${plot.left}" y1="${plot.top}" x2="${plot.left}" y2="${plot.top + plot.height}" />
    <polyline class="flux-line" points="${pts.join(" ")}" />
    ${ticks.join("")}
    <text class="axis-label" x="${plot.left + plot.width / 2}" y="232" text-anchor="middle">λ [Å]</text>`;
}

function renderSpaxelMeta() {
  if (!state.spaxel) {
    createMetaCards(selectionMeta, []);
    return;
  }
  const s = state.spaxel;
  const entries = [
    ["Spaxel", `(${s.x}, ${s.y})`],
    ["M_valid", s.valid ? "sí" : "no"],
    ["N_eff", formatNumber(s.n_eff, 1)],
  ];
  Object.entries(s.pipe3d).forEach(([key, value]) => {
    entries.push([`p3d ${key}`, formatNumber(value, 2)]);
  });
  createMetaCards(selectionMeta, entries);
}

async function loadMap() {
  if (!state.selectedPath) {
    return;
  }
  let url = `/api/map?path=${encodeURIComponent(state.selectedPath)}&layer=${state.layer}&variant=${state.variant}`;
  if (state.layer === "class_prob") {
    url += `&class_index=${state.classIndex}`;
  }
  if (state.layer === "cube_slice") {
    url += `&index=${state.sliceIndex}`;
  }
  state.mapPayload = await fetchJson(url);
  mapCaption.textContent = state.mapPayload.label || state.layer;
  drawMap();
}

async function loadSpaxel(x, y) {
  state.spaxel = await fetchJson(
    `/api/spaxel?path=${encodeURIComponent(state.selectedPath)}&x=${x}&y=${y}`,
  );
  selectionCaption.textContent = `Spaxel (${x}, ${y}) · variante ${state.variant}`;
  renderProbBars();
  renderSpectrum();
  renderSpaxelMeta();
}

function rebuildClassChips() {
  classChipRow.innerHTML = "";
  state.entry.class_names.forEach((name, i) => {
    const chip = document.createElement("button");
    chip.className = `chip${i === state.classIndex ? " active" : ""}`;
    chip.textContent = name;
    chip.style.borderColor = state.entry.class_colors[i];
    if (i === state.classIndex) {
      chip.style.background = state.entry.class_colors[i];
    }
    chip.addEventListener("click", async () => {
      state.classIndex = i;
      rebuildClassChips();
      if (state.layer === "class_prob") {
        await loadMap();
      }
    });
    classChipRow.appendChild(chip);
  });
}

async function loadEntry(path) {
  state.selectedPath = path;
  state.selectedPixel = null;
  state.spaxel = null;

  const entry = await fetchJson(`/api/entry?path=${encodeURIComponent(path)}`);
  state.entry = entry;
  state.sliceIndex = Math.floor(entry.shape.n_wave / 2);
  sliceSlider.max = entry.shape.n_wave - 1;
  sliceSlider.value = state.sliceIndex;
  sliceLabel.textContent = `#${state.sliceIndex}`;

  pipe3dSelect.innerHTML = '<option value="">— ninguno —</option>';
  entry.pipe3d_maps.forEach((item) => {
    const option = document.createElement("option");
    option.value = `pipe3d:${item.name}`;
    option.textContent = item.label;
    pipe3dSelect.appendChild(option);
  });

  rebuildClassChips();

  const qa = entry.qa || {};
  const cards = [
    ["Galaxia", entry.galaxy_id],
    ["Vista", String(entry.view_id)],
    ["Grilla", `${entry.shape.ny} × ${entry.shape.nx}`],
    ["Spaxels válidos", String(entry.n_valid)],
    ["QA status", String(qa.status || "-")],
    ["Conserv. masa", formatNumber(qa.mass_conservation_error, 6)],
    ["Barra detectada", qa.bar_detected === undefined ? "-" : String(qa.bar_detected)],
    ["Crestas de brazo", qa.n_arm_crests === undefined ? "-" : String(qa.n_arm_crests)],
  ];
  if (qa.flags) {
    const flags = Array.isArray(qa.flags) ? qa.flags : [String(qa.flags)];
    cards.push(["Flags", flags.length ? flags.join(", ") : "ninguno"]);
  }
  createMetaCards(metaGrid, cards);

  await loadMap();
  renderProbBars();
  renderSpectrum();
  renderSpaxelMeta();
}

fileSelect.addEventListener("change", (event) => {
  loadEntry(event.target.value).catch(showError);
});

layerButtons.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", async () => {
    layerButtons.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    pipe3dSelect.value = "";
    state.layer = button.dataset.layer;
    classChipRow.classList.toggle("hidden", state.layer !== "class_prob");
    await loadMap();
  });
});

variantButtons.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", async () => {
    variantButtons.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    state.variant = button.dataset.variant;
    if (["segmentation", "class_prob"].includes(state.layer)) {
      await loadMap();
    }
    renderProbBars();
    if (state.spaxel) {
      selectionCaption.textContent = `Spaxel (${state.spaxel.x}, ${state.spaxel.y}) · variante ${state.variant}`;
    }
  });
});

pipe3dSelect.addEventListener("change", async (event) => {
  if (!event.target.value) {
    return;
  }
  layerButtons.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
  classChipRow.classList.add("hidden");
  state.layer = event.target.value;
  await loadMap();
});

toggleValidOverlay.addEventListener("change", drawMap);

sliceSlider.addEventListener("input", async (event) => {
  state.sliceIndex = Number(event.target.value);
  sliceLabel.textContent = `#${state.sliceIndex}`;
  if (state.layer === "cube_slice") {
    await loadMap();
  }
});

mapCanvas.addEventListener("click", async (event) => {
  if (!state.entry) {
    return;
  }
  const rect = mapCanvas.getBoundingClientRect();
  const cx = ((event.clientX - rect.left) / rect.width) * mapCanvas.width;
  const cy = ((event.clientY - rect.top) / rect.height) * mapCanvas.height;
  const nx = state.entry.shape.nx;
  const ny = state.entry.shape.ny;
  const x = Math.max(0, Math.min(nx - 1, Math.floor((cx / mapCanvas.width) * nx)));
  const row = Math.max(0, Math.min(ny - 1, Math.floor((cy / mapCanvas.height) * ny)));
  const y = ny - 1 - row; // canvas dibuja con origen abajo
  state.selectedPixel = { x, y };
  drawMap();
  await loadSpaxel(x, y).catch(showError);
});

function showError(error) {
  console.error(error);
  mapCaption.textContent = `Error: ${error.message}`;
}

async function bootstrap() {
  const payload = await fetchJson("/api/files");
  state.files = payload.files;
  if (state.files.length === 0) {
    fileSelect.innerHTML = "<option>No hay dataset entries</option>";
    mapCaption.textContent =
      "Corre `python -m aperturenet_labels.cli.main run --pilot` para generarlos";
    return;
  }
  fileSelect.innerHTML = "";
  state.files.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.label;
    fileSelect.appendChild(option);
  });
  await loadEntry(state.files[0].path);
}

bootstrap().catch(showError);
