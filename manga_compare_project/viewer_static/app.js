const state = {
  files: [],
  selectedPath: null,
  observation: null,
  currentMapMode: "overview",
  currentSlice: 0,
  selectedCubePixel: null,
  selectedFiber: null,
  mapPayload: null,
  spectrumPayload: null,
};

const fileSelect = document.getElementById("file-select");
const sliceSlider = document.getElementById("slice-slider");
const sliceLabel = document.getElementById("slice-label");
const overviewButton = document.getElementById("overview-button");
const sliceButton = document.getElementById("slice-button");
const mapCanvas = document.getElementById("map-canvas");
const mapCaption = document.getElementById("map-caption");
const selectionCaption = document.getElementById("selection-caption");
const metaGrid = document.getElementById("meta-grid");
const selectionMeta = document.getElementById("selection-meta");
const spectrumSvg = document.getElementById("spectrum-svg");
const toggleError = document.getElementById("toggle-error");
const toggleGas = document.getElementById("toggle-gas");
const toggleMask = document.getElementById("toggle-mask");
const mapCtx = mapCanvas.getContext("2d");

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

function createMetaCards(container, entries) {
  container.innerHTML = "";
  entries.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "meta-card";
    card.innerHTML = `<label>${label}</label><strong>${value}</strong>`;
    container.appendChild(card);
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
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
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * t);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * t);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return "rgb(231, 111, 81)";
}

function drawImageMap(payload) {
  const data = payload.data;
  const ny = data.length;
  const nx = data[0].length;
  const cellWidth = mapCanvas.width / nx;
  const cellHeight = mapCanvas.height / ny;
  const span = payload.vmax - payload.vmin || 1;

  mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const raw = data[y][x];
      const norm = (raw - payload.vmin) / span;
      mapCtx.fillStyle = turboColor(norm);
      mapCtx.fillRect(x * cellWidth, y * cellHeight, cellWidth + 1, cellHeight + 1);
    }
  }

  if (state.selectedCubePixel) {
    mapCtx.strokeStyle = "rgba(255,255,255,0.95)";
    mapCtx.lineWidth = 2;
    mapCtx.strokeRect(
      state.selectedCubePixel.x * cellWidth,
      state.selectedCubePixel.y * cellHeight,
      cellWidth,
      cellHeight,
    );
  }
}

function drawScatterMap(payload) {
  const { xmin, xmax, ymin, ymax } = payload.bounds;
  const span = payload.vmax - payload.vmin || 1;

  const xScale = (value) => ((value - xmin) / (xmax - xmin || 1)) * (mapCanvas.width - 40) + 20;
  const yScale = (value) => mapCanvas.height - (((value - ymin) / (ymax - ymin || 1)) * (mapCanvas.height - 40) + 20);

  mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  mapCtx.fillStyle = "white";
  mapCtx.fillRect(0, 0, mapCanvas.width, mapCanvas.height);

  payload.points.forEach((point) => {
    const norm = (point.value - payload.vmin) / span;
    mapCtx.beginPath();
    mapCtx.fillStyle = turboColor(norm);
    mapCtx.arc(xScale(point.x), yScale(point.y), 5, 0, Math.PI * 2);
    mapCtx.fill();

    if (state.selectedFiber === point.index) {
      mapCtx.beginPath();
      mapCtx.strokeStyle = "rgba(255,255,255,0.95)";
      mapCtx.lineWidth = 2;
      mapCtx.arc(xScale(point.x), yScale(point.y), 8, 0, Math.PI * 2);
      mapCtx.stroke();
    }
  });
}

function drawMap() {
  if (!state.mapPayload) {
    mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
    return;
  }
  if (state.mapPayload.kind === "image") {
    drawImageMap(state.mapPayload);
  } else {
    drawScatterMap(state.mapPayload);
  }
}

function computePlotScales(xs, seriesList, plot) {
  const minX = xs[0];
  const maxX = xs[xs.length - 1];
  let minY = Infinity;
  let maxY = -Infinity;

  seriesList.forEach((series) => {
    if (!series) {
      return;
    }
    series.forEach((value) => {
      if (Number.isFinite(value)) {
        minY = Math.min(minY, value);
        maxY = Math.max(maxY, value);
      }
    });
  });

  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
    minY = -1;
    maxY = 1;
  }
  if (maxY <= minY) {
    maxY = minY + 1;
  }

  const xScale = (value) => plot.left + ((value - minX) / (maxX - minX || 1)) * plot.width;
  const yScale = (value) => plot.top + plot.height - ((value - minY) / (maxY - minY)) * plot.height;
  return { minX, maxX, minY, maxY, xScale, yScale };
}

function seriesToPoints(xs, ys, scales) {
  return ys
    .map((value, index) => `${scales.xScale(xs[index]).toFixed(2)},${scales.yScale(value).toFixed(2)}`)
    .join(" ");
}

function buildMaskBands(mask, xs, xScale, yBottom, bandHeight) {
  if (!mask || !toggleMask.checked) {
    return "";
  }
  const bands = [];
  let start = null;
  for (let i = 0; i < mask.length; i += 1) {
    const flagged = Number(mask[i]) !== 0;
    if (flagged && start === null) {
      start = i;
    }
    if ((!flagged || i === mask.length - 1) && start !== null) {
      const end = flagged && i === mask.length - 1 ? i : i - 1;
      const x0 = xScale(xs[start]);
      const x1 = xScale(xs[end]);
      bands.push(`<rect class="mask-band" x="${x0}" y="${yBottom}" width="${Math.max(1, x1 - x0)}" height="${bandHeight}" />`);
      start = null;
    }
  }
  return bands.join("");
}

function renderSpectrum() {
  if (!state.spectrumPayload) {
    spectrumSvg.innerHTML = `<text class="empty-state" x="50%" y="50%" text-anchor="middle">Selecciona un spaxel o fibra en el mapa</text>`;
    createMetaCards(selectionMeta, []);
    return;
  }

  const payload = state.spectrumPayload;
  const xs = payload.wave;
  const flux = payload.flux;
  const plot = { left: 60, right: 20, top: 24, bottom: 54, width: 780, height: 320 };
  const activeSeries = [flux];
  if (toggleError.checked && payload.error) {
    activeSeries.push(payload.error);
  }
  if (toggleGas.checked && payload.gas) {
    activeSeries.push(payload.gas);
  }
  const scales = computePlotScales(xs, activeSeries, plot);
  const fluxPoints = seriesToPoints(xs, flux, scales);
  const errorPoints = payload.error ? seriesToPoints(xs, payload.error, scales) : null;
  const gasPoints = payload.gas ? seriesToPoints(xs, payload.gas, scales) : null;
  const sliceX = scales.xScale(xs[state.currentSlice]);

  const horizontalTicks = 4;
  const verticalTicks = 5;
  const grid = [];
  for (let i = 0; i <= horizontalTicks; i += 1) {
    const y = plot.top + (plot.height / horizontalTicks) * i;
    grid.push(`<line class="grid-line" x1="${plot.left}" y1="${y}" x2="${plot.left + plot.width}" y2="${y}" />`);
  }
  for (let i = 0; i <= verticalTicks; i += 1) {
    const x = plot.left + (plot.width / verticalTicks) * i;
    grid.push(`<line class="grid-line" x1="${x}" y1="${plot.top}" x2="${x}" y2="${plot.top + plot.height}" />`);
  }

  const xTickLabels = [];
  for (let i = 0; i <= verticalTicks; i += 1) {
    const frac = i / verticalTicks;
    const value = scales.minX + (scales.maxX - scales.minX) * frac;
    const x = plot.left + plot.width * frac;
    xTickLabels.push(`<text class="tick-label" x="${x}" y="${plot.top + plot.height + 24}" text-anchor="middle">${formatNumber(value, 1)}</text>`);
  }

  const yTickLabels = [];
  for (let i = 0; i <= horizontalTicks; i += 1) {
    const frac = 1 - i / horizontalTicks;
    const value = scales.minY + (scales.maxY - scales.minY) * frac;
    const y = plot.top + plot.height * (1 - frac);
    yTickLabels.push(`<text class="tick-label" x="${plot.left - 10}" y="${y + 4}" text-anchor="end">${formatNumber(value, 3)}</text>`);
  }

  const maskBands = buildMaskBands(payload.mask, xs, scales.xScale, plot.top + plot.height + 8, 18);

  spectrumSvg.innerHTML = `
    ${grid.join("")}
    <line class="axis" x1="${plot.left}" y1="${plot.top + plot.height}" x2="${plot.left + plot.width}" y2="${plot.top + plot.height}" />
    <line class="axis" x1="${plot.left}" y1="${plot.top}" x2="${plot.left}" y2="${plot.top + plot.height}" />
    <polyline class="flux-line" points="${fluxPoints}" />
    ${toggleError.checked && errorPoints ? `<polyline class="error-line" points="${errorPoints}" />` : ""}
    ${toggleGas.checked && gasPoints ? `<polyline class="gas-line" points="${gasPoints}" />` : ""}
    <line class="slice-indicator" x1="${sliceX}" y1="${plot.top}" x2="${sliceX}" y2="${plot.top + plot.height}" />
    ${maskBands}
    ${xTickLabels.join("")}
    ${yTickLabels.join("")}
    <text class="axis-label" x="${plot.left + plot.width / 2}" y="408" text-anchor="middle">Longitud de onda</text>
    <text class="axis-label" x="20" y="${plot.top + plot.height / 2}" text-anchor="middle" transform="rotate(-90,20,${plot.top + plot.height / 2})">Flujo</text>
  `;

  const entries = [
    ["Seleccion", payload.label],
    ["Flujo medio", formatNumber(payload.stats.mean, 4)],
    ["Flujo total", formatNumber(payload.stats.sum, 4)],
    ["Mascara > 0", String(payload.stats.masked_count)],
  ];
  Object.entries(payload.coords).forEach(([key, value]) => {
    entries.push([key, formatNumber(value, 3)]);
  });
  createMetaCards(selectionMeta, entries);
}

async function loadSpectrumForSelection(selection) {
  if (!state.selectedPath) {
    return;
  }
  let url = `/api/spectrum?path=${encodeURIComponent(state.selectedPath)}`;
  if (selection.kind === "spaxel") {
    url += `&x=${selection.x}&y=${selection.y}`;
  } else {
    url += `&index=${selection.index}`;
  }
  state.spectrumPayload = await fetchJson(url);
  selectionCaption.textContent = state.spectrumPayload.label;
  renderSpectrum();
}

async function loadMap(mode, index = state.currentSlice) {
  state.currentMapMode = mode;
  let url = `/api/map?path=${encodeURIComponent(state.selectedPath)}&mode=${mode}`;
  if (mode === "slice") {
    url += `&index=${index}`;
  }
  state.mapPayload = await fetchJson(url);
  drawMap();
  mapCaption.textContent = state.mapPayload.label || (mode === "slice" ? "Slice actual" : "Mapa medio");
}

async function loadObservation(path) {
  state.selectedPath = path;
  state.selectedCubePixel = null;
  state.selectedFiber = null;
  state.spectrumPayload = null;

  const payload = await fetchJson(`/api/observation?path=${encodeURIComponent(path)}`);
  state.observation = payload;
  state.currentSlice = payload.default_slice_index;

  sliceSlider.max = payload.shape.n_wave - 1;
  sliceSlider.value = payload.default_slice_index;
  sliceLabel.textContent = `#${payload.default_slice_index} · ${formatNumber(payload.wave.min + ((payload.wave.max - payload.wave.min) * payload.default_slice_index / Math.max(1, payload.shape.n_wave - 1)), 1)} A`;

  const cards = [
    ["Archivo", payload.path],
    ["Tipo", `${payload.source} / ${payload.shape.mode}`],
    ["Wave Range", `${formatNumber(payload.wave.min, 1)} - ${formatNumber(payload.wave.max, 1)} A`],
    ["N wave", String(payload.shape.n_wave)],
  ];

  if (payload.shape.mode === "cube") {
    cards.push(["Shape", `${payload.shape.n_wave} × ${payload.shape.ny} × ${payload.shape.nx}`]);
  } else {
    cards.push(["RSS Shape", `${payload.shape.n_sample} × ${payload.shape.n_wave}`]);
  }
  cards.push(["Error", payload.has_error ? "Si" : "No"]);
  cards.push(["Mascara", payload.has_mask ? "Si" : "No"]);
  cards.push(["Gas", payload.has_gas ? "Si" : "No"]);
  cards.push(["Extensiones", payload.extnames.join(", ")]);
  Object.entries(payload.header_summary).forEach(([key, value]) => cards.push([key, String(value)]));
  createMetaCards(metaGrid, cards);

  await loadMap("overview");
  renderSpectrum();
}

function findNearestFiber(payload, canvasX, canvasY) {
  const { xmin, xmax, ymin, ymax } = payload.bounds;
  const xScale = (value) => ((value - xmin) / (xmax - xmin || 1)) * (mapCanvas.width - 40) + 20;
  const yScale = (value) => mapCanvas.height - (((value - ymin) / (ymax - ymin || 1)) * (mapCanvas.height - 40) + 20);

  let best = null;
  let bestD2 = Infinity;
  payload.points.forEach((point) => {
    const dx = xScale(point.x) - canvasX;
    const dy = yScale(point.y) - canvasY;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = point;
    }
  });
  if (bestD2 <= 20 * 20) {
    return best;
  }
  return null;
}

fileSelect.addEventListener("change", async (event) => {
  await loadObservation(event.target.value);
});

sliceSlider.addEventListener("input", async (event) => {
  state.currentSlice = Number(event.target.value);
  sliceLabel.textContent = `#${state.currentSlice}`;
  if (state.currentMapMode === "slice") {
    await loadMap("slice", state.currentSlice);
  }
  renderSpectrum();
});

overviewButton.addEventListener("click", async () => {
  overviewButton.classList.add("active");
  sliceButton.classList.remove("active");
  await loadMap("overview");
});

sliceButton.addEventListener("click", async () => {
  sliceButton.classList.add("active");
  overviewButton.classList.remove("active");
  await loadMap("slice", state.currentSlice);
});

mapCanvas.addEventListener("click", async (event) => {
  if (!state.observation || !state.mapPayload) {
    return;
  }

  const rect = mapCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * mapCanvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * mapCanvas.height;

  if (state.observation.shape.mode === "cube" && state.mapPayload.kind === "image") {
    const px = Math.floor((x / mapCanvas.width) * state.observation.shape.nx);
    const py = Math.floor((y / mapCanvas.height) * state.observation.shape.ny);
    state.selectedCubePixel = {
      x: Math.max(0, Math.min(state.observation.shape.nx - 1, px)),
      y: Math.max(0, Math.min(state.observation.shape.ny - 1, py)),
    };
    drawMap();
    await loadSpectrumForSelection({ kind: "spaxel", x: state.selectedCubePixel.x, y: state.selectedCubePixel.y });
    return;
  }

  if (state.observation.shape.mode === "rss" && state.mapPayload.kind === "scatter") {
    const nearest = findNearestFiber(state.mapPayload, x, y);
    if (!nearest) {
      return;
    }
    state.selectedFiber = nearest.index;
    drawMap();
    await loadSpectrumForSelection({ kind: "fiber", index: nearest.index });
  }
});

[toggleError, toggleGas, toggleMask].forEach((node) => {
  node.addEventListener("change", renderSpectrum);
});

async function bootstrap() {
  const payload = await fetchJson("/api/files");
  state.files = payload.files;

  if (state.files.length === 0) {
    fileSelect.innerHTML = `<option>No se encontraron FITS compatibles</option>`;
    mapCaption.textContent = "Descarga un MaNGA LOGCUBE/LOGRSS o copia un mock cube dentro de data/";
    renderSpectrum();
    return;
  }

  fileSelect.innerHTML = "";
  state.files.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.label;
    fileSelect.appendChild(option);
  });

  await loadObservation(state.files[0].path);
}

bootstrap().catch((error) => {
  console.error(error);
  mapCaption.textContent = `Error: ${error.message}`;
  spectrumSvg.innerHTML = `<text class="empty-state" x="50%" y="50%" text-anchor="middle">${error.message}</text>`;
});
