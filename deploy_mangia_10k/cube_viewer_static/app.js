const state = {
  files: [],
  selectedPath: null,
  cubeMeta: null,
  currentMapMode: "overview",
  currentSlice: 0,
  selectedSpaxel: null,
  spaxelPayload: null,
};

const cubeSelect = document.getElementById("cube-select");
const sliceSlider = document.getElementById("slice-slider");
const sliceLabel = document.getElementById("slice-label");
const overviewButton = document.getElementById("overview-button");
const sliceButton = document.getElementById("slice-button");
const mapCanvas = document.getElementById("map-canvas");
const mapCaption = document.getElementById("map-caption");
const spaxelCaption = document.getElementById("spaxel-caption");
const cubeMeta = document.getElementById("cube-meta");
const spaxelMeta = document.getElementById("spaxel-meta");
const spectrumSvg = document.getElementById("spectrum-svg");
const toggleError = document.getElementById("toggle-error");
const toggleGas = document.getElementById("toggle-gas");
const toggleMask = document.getElementById("toggle-mask");

const mapCtx = mapCanvas.getContext("2d");

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
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

function drawMap(imagePayload) {
  if (!state.cubeMeta) {
    return;
  }

  const data = imagePayload.data;
  const ny = data.length;
  const nx = data[0].length;
  const cellWidth = mapCanvas.width / nx;
  const cellHeight = mapCanvas.height / ny;

  mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);

  const span = imagePayload.vmax - imagePayload.vmin || 1;
  for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const raw = data[y][x];
      const norm = (raw - imagePayload.vmin) / span;
      mapCtx.fillStyle = turboColor(norm);
      mapCtx.fillRect(x * cellWidth, y * cellHeight, cellWidth + 1, cellHeight + 1);
    }
  }

  if (state.selectedSpaxel) {
    mapCtx.strokeStyle = "rgba(255,255,255,0.95)";
    mapCtx.lineWidth = 2;
    mapCtx.strokeRect(
      state.selectedSpaxel.x * cellWidth,
      state.selectedSpaxel.y * cellHeight,
      cellWidth,
      cellHeight,
    );
  }
}

function computePlotScales(xs, seriesList, plot) {
  const n = xs.length;
  const minX = xs[0];
  const maxX = xs[n - 1];
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
  if (!state.spaxelPayload) {
    spectrumSvg.innerHTML = `<text class="empty-state" x="50%" y="50%" text-anchor="middle">Selecciona un spaxel en el mapa</text>`;
    createMetaCards(spaxelMeta, []);
    return;
  }

  const payload = state.spaxelPayload;
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
    yTickLabels.push(`<text class="tick-label" x="${plot.left - 10}" y="${y + 4}" text-anchor="end">${formatNumber(value, 2)}</text>`);
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
    ["Spaxel", `x=${payload.x}, y=${payload.y}`],
    ["X Grid", payload.coords.x_grid === null ? "-" : formatNumber(payload.coords.x_grid, 3)],
    ["Y Grid", payload.coords.y_grid === null ? "-" : formatNumber(payload.coords.y_grid, 3)],
    ["Flujo medio", formatNumber(payload.stats.mean, 4)],
    ["Flujo total", formatNumber(payload.stats.sum, 4)],
    ["Mascara > 0", String(payload.stats.masked_count)],
  ];
  createMetaCards(spaxelMeta, entries);
}

async function loadSpaxel(x, y) {
  state.selectedSpaxel = { x, y };
  drawMap(getCurrentMapPayload());
  const payload = await fetchJson(`/api/spaxel?path=${encodeURIComponent(state.selectedPath)}&x=${x}&y=${y}`);
  state.spaxelPayload = payload;
  spaxelCaption.textContent = `Spaxel seleccionado: (${x}, ${y})`;
  renderSpectrum();
}

async function loadSlice(index) {
  state.currentSlice = Number(index);
  const payload = await fetchJson(`/api/slice?path=${encodeURIComponent(state.selectedPath)}&index=${state.currentSlice}`);
  state.slicePayload = payload;
  sliceLabel.textContent = `#${payload.index} · ${formatNumber(payload.wavelength, 1)} A`;
  if (state.currentMapMode === "slice") {
    drawMap(payload.image);
    mapCaption.textContent = `Slice actual en ${formatNumber(payload.wavelength, 1)} A`;
  }
  renderSpectrum();
}

function getCurrentMapPayload() {
  if (state.currentMapMode === "slice" && state.slicePayload) {
    return state.slicePayload.image;
  }
  return state.cubeMeta.overview_map;
}

async function loadCube(path) {
  state.selectedPath = path;
  state.selectedSpaxel = null;
  state.spaxelPayload = null;
  state.slicePayload = null;

  const payload = await fetchJson(`/api/cube?path=${encodeURIComponent(path)}`);
  state.cubeMeta = payload;
  state.currentSlice = payload.default_slice_index;

  sliceSlider.max = payload.shape.n_wave - 1;
  sliceSlider.value = payload.default_slice_index;

  const headerCards = [
    ["Archivo", payload.path],
    ["Shape", `${payload.shape.n_wave} × ${payload.shape.ny} × ${payload.shape.nx}`],
    ["Wave Range", `${formatNumber(payload.wave.min, 1)} - ${formatNumber(payload.wave.max, 1)} A`],
    ["Error", payload.has_error ? "Si" : "No"],
    ["Mascara", payload.has_mask ? "Si" : "No"],
    ["Gas", payload.has_gas ? "Si" : "No"],
    ["Extensiones", payload.extnames.join(", ")],
  ];

  Object.entries(payload.header_summary).forEach(([key, value]) => {
    headerCards.push([key, String(value)]);
  });
  createMetaCards(cubeMeta, headerCards);

  await loadSlice(payload.default_slice_index);
  drawMap(payload.overview_map);
  mapCaption.textContent = "Mapa medio del cubo";
  spaxelCaption.textContent = "Haz click en el mapa para cargar un espectro";
  renderSpectrum();
}

async function bootstrap() {
  const payload = await fetchJson("/api/files");
  state.files = payload.files;

  if (state.files.length === 0) {
    cubeSelect.innerHTML = `<option>No se encontraron cubos 3D</option>`;
    mapCaption.textContent = "Reconstruye un cubo y vuelve a cargar esta pagina";
    renderSpectrum();
    return;
  }

  cubeSelect.innerHTML = "";
  state.files.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.label;
    cubeSelect.appendChild(option);
  });

  await loadCube(state.files[0].path);
}

cubeSelect.addEventListener("change", async (event) => {
  await loadCube(event.target.value);
});

sliceSlider.addEventListener("input", async (event) => {
  await loadSlice(event.target.value);
});

overviewButton.addEventListener("click", () => {
  state.currentMapMode = "overview";
  overviewButton.classList.add("active");
  sliceButton.classList.remove("active");
  drawMap(state.cubeMeta.overview_map);
  mapCaption.textContent = "Mapa medio del cubo";
});

sliceButton.addEventListener("click", () => {
  state.currentMapMode = "slice";
  sliceButton.classList.add("active");
  overviewButton.classList.remove("active");
  if (state.slicePayload) {
    drawMap(state.slicePayload.image);
    mapCaption.textContent = `Slice actual en ${formatNumber(state.slicePayload.wavelength, 1)} A`;
  }
});

mapCanvas.addEventListener("click", async (event) => {
  if (!state.cubeMeta) {
    return;
  }
  const rect = mapCanvas.getBoundingClientRect();
  const x = Math.floor(((event.clientX - rect.left) / rect.width) * state.cubeMeta.shape.nx);
  const y = Math.floor(((event.clientY - rect.top) / rect.height) * state.cubeMeta.shape.ny);
  await loadSpaxel(
    Math.max(0, Math.min(state.cubeMeta.shape.nx - 1, x)),
    Math.max(0, Math.min(state.cubeMeta.shape.ny - 1, y)),
  );
});

[toggleError, toggleGas, toggleMask].forEach((node) => {
  node.addEventListener("change", renderSpectrum);
});

bootstrap().catch((error) => {
  console.error(error);
  mapCaption.textContent = `Error: ${error.message}`;
  spectrumSvg.innerHTML = `<text class="empty-state" x="50%" y="50%" text-anchor="middle">${error.message}</text>`;
});
