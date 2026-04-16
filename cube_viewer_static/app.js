const state = {
  files: [],
  selectedPath: null,
  cubeMeta: null,
  labelMeta: null,
  currentMapMode: "overview",
  currentSlice: 0,
  selectedSpaxel: null,
  spaxelPayload: null,
  slicePayload: null,
  overlayMode: "off",
  overlayClassIndex: 0,
  overlayOpacity: 0.65,
  overlayPayload: null,
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
const labelModeSelect = document.getElementById("label-mode-select");
const labelClassSelect = document.getElementById("label-class-select");
const overlayOpacitySlider = document.getElementById("overlay-opacity");
const overlayOpacityValue = document.getElementById("overlay-opacity-value");
const labelStatus = document.getElementById("label-status");
const labelSummary = document.getElementById("label-summary");
const spaxelMeta = document.getElementById("spaxel-meta");
const spaxelLabelMeta = document.getElementById("spaxel-label-meta");
const spectrumSvg = document.getElementById("spectrum-svg");
const toggleError = document.getElementById("toggle-error");
const toggleGas = document.getElementById("toggle-gas");
const toggleMask = document.getElementById("toggle-mask");

const mapCtx = mapCanvas.getContext("2d");

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  const numeric = Number(value);
  const abs = Math.abs(numeric);
  if (abs !== 0 && (abs >= 1e4 || abs < 1e-3)) {
    return numeric.toExponential(2);
  }
  return numeric.toFixed(digits);
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function friendlyClassName(name) {
  return {
    no_valido: "No valido",
    bulbo: "Bulbo",
    disco: "Disco",
    barra: "Barra",
    brazos: "Brazos",
    other: "Other",
    incierto: "Incierto",
    incierto_otro: "Incierto/Otro",
  }[name] || name;
}

function thresholdLabel(key) {
  const numeric = Number(key);
  if (Number.isFinite(numeric)) {
    return (numeric / 100).toFixed(2);
  }
  return String(key);
}

function friendlyModeLabel(key) {
  if (key === "off") return "Sin overlay";
  if (key === "soft_mass") return "Soft mass";
  if (key === "soft_light") return "Soft light";
  if (key === "hard_mass") return "Hard mass default";
  if (key === "hard_light") return "Hard light default";
  if (key.startsWith("hard_mass_")) {
    return `Hard mass ${thresholdLabel(key.replace("hard_mass_", ""))}`;
  }
  if (key.startsWith("hard_light_")) {
    return `Hard light ${thresholdLabel(key.replace("hard_light_", ""))}`;
  }
  return key;
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
      return [r, g, b];
    }
  }
  return [231, 111, 81];
}

function rgba(color, alpha) {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function drawMap(imagePayload) {
  if (!state.cubeMeta || !imagePayload) {
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
      const color = turboColor(norm);
      mapCtx.fillStyle = rgba(color, 1.0);
      mapCtx.fillRect(x * cellWidth, y * cellHeight, cellWidth + 1, cellHeight + 1);
    }
  }

  if (state.overlayPayload) {
    const overlay = state.overlayPayload.image.data;
    const overlaySpan = state.overlayPayload.image.vmax - state.overlayPayload.image.vmin || 1;
    const overlayColor = state.overlayPayload.color;
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const raw = overlay[y][x];
        if (!Number.isFinite(raw) || raw <= 0) {
          continue;
        }
        const norm = Math.min(
          1,
          Math.max(0, (raw - state.overlayPayload.image.vmin) / overlaySpan),
        );
        const alpha = state.overlayOpacity * Math.sqrt(norm);
        if (alpha <= 0.01) {
          continue;
        }
        mapCtx.fillStyle = rgba(overlayColor, alpha);
        mapCtx.fillRect(x * cellWidth, y * cellHeight, cellWidth + 1, cellHeight + 1);
      }
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
    createMetaCards(spaxelLabelMeta, []);
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

  const labelEntries = [];
  if (payload.labels && payload.labels.available) {
    labelEntries.push(["Valido", payload.labels.valid ? "Si" : "No"]);
    labelEntries.push(["Hard Mass", friendlyClassName(payload.labels.hard_mass.name)]);
    labelEntries.push(["Conf. Mass", formatPercent(payload.labels.confidence_mass, 1)]);
    labelEntries.push(["Hard Light", friendlyClassName(payload.labels.hard_light.name)]);
    labelEntries.push(["Conf. Light", formatPercent(payload.labels.confidence_light, 1)]);

    Object.entries(payload.labels.hard_mass_variants || {}).forEach(([key, item]) => {
      labelEntries.push([`Mass ${thresholdLabel(key)}`, friendlyClassName(item.name)]);
    });
    Object.entries(payload.labels.hard_light_variants || {}).forEach(([key, item]) => {
      labelEntries.push([`Light ${thresholdLabel(key)}`, friendlyClassName(item.name)]);
    });

    ["bulbo", "disco", "barra", "brazos", "other", "incierto", "incierto_otro"].forEach((className) => {
      if (payload.labels.soft_mass[className] !== undefined) {
        labelEntries.push([`P ${friendlyClassName(className)}`, formatPercent(payload.labels.soft_mass[className], 1)]);
      }
    });
  } else {
    labelEntries.push(["Etiquetas", "No disponibles"]);
  }
  createMetaCards(spaxelLabelMeta, labelEntries);
}

function getCurrentMapPayload() {
  if (state.currentMapMode === "slice" && state.slicePayload) {
    return state.slicePayload.image;
  }
  return state.cubeMeta ? state.cubeMeta.overview_map : null;
}

function updateMapCaption() {
  let caption = state.currentMapMode === "slice" && state.slicePayload
    ? `Slice actual en ${formatNumber(state.slicePayload.wavelength, 1)} A`
    : "Mapa medio del cubo";

  if (state.overlayPayload) {
    const coverage = formatPercent(state.overlayPayload.stats.coverage, 1);
    caption += ` · overlay ${friendlyModeLabel(state.overlayPayload.mode)} / ${friendlyClassName(state.overlayPayload.class_name)} · cobertura ${coverage}`;
  }

  mapCaption.textContent = caption;
}

function renderLabelSummary(meta) {
  if (!meta || !meta.available || !meta.summary) {
    createMetaCards(labelSummary, []);
    return;
  }

  const recovered = meta.summary.global_fraction_recovered || {};
  const targets = meta.summary.global_fraction_targets || {};
  const bar = meta.summary.bar_metadata || {};
  const recoveredOther = recovered.other ?? recovered.incierto_otro;
  const entries = [
    ["Bulbo", `t ${formatPercent(targets.bulge_family)} · r ${formatPercent(recovered.bulbo)}`],
    ["Disco", `t ${formatPercent(targets.disk_family)} · r ${formatPercent(recovered.disk_family_total)}`],
    ["Barra", `${formatPercent(recovered.barra)} · ${bar.barred_target ? "barrada" : "no barrada"}`],
    ["Brazos", formatPercent(recovered.brazos)],
    ["Other", `t ${formatPercent(targets.other_family)} · r ${formatPercent(recoveredOther)}`],
    ["Bar Size", bar.bar_radius_recovered_kpc === undefined ? "-" : `${formatNumber(bar.bar_radius_recovered_kpc, 2)} kpc`],
  ];

  const hardSummary = meta.summary.hard_variant_summary || {};
  Object.entries(hardSummary).forEach(([key, families]) => {
    const mass = families.mass || {};
    const light = families.light || {};
    entries.push([
      `Mass ${thresholdLabel(key)}`,
      `disco ${mass.disco || 0} · inc ${mass.incierto || 0}`,
    ]);
    entries.push([
      `Light ${thresholdLabel(key)}`,
      `disco ${light.disco || 0} · inc ${light.incierto || 0}`,
    ]);
  });
  createMetaCards(labelSummary, entries);
}

function updateLabelControls(meta) {
  labelModeSelect.innerHTML = "";
  labelClassSelect.innerHTML = "";

  if (!meta || !meta.available) {
    const offOption = document.createElement("option");
    offOption.value = "off";
    offOption.textContent = "Sin overlay";
    labelModeSelect.appendChild(offOption);
    labelModeSelect.value = "off";

    const placeholder = document.createElement("option");
    placeholder.value = "0";
    placeholder.textContent = "Sin clases";
    labelClassSelect.appendChild(placeholder);
    labelClassSelect.disabled = true;

    overlayOpacitySlider.disabled = true;
    labelStatus.textContent = meta?.reason || "No hay etiquetas para este cubo";
    renderLabelSummary(null);
    return;
  }

  meta.modes.forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode.key;
    option.textContent = mode.label;
    labelModeSelect.appendChild(option);
  });
  labelModeSelect.value = state.overlayMode;

  meta.class_names.forEach((className, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = friendlyClassName(className);
    labelClassSelect.appendChild(option);
  });
  labelClassSelect.value = String(state.overlayClassIndex);
  labelClassSelect.disabled = state.overlayMode === "off";

  overlayOpacitySlider.disabled = false;
  labelStatus.textContent = `Etiquetas activas para ${meta.canonical_id} · ${formatPercent(meta.valid_fraction, 1)} del campo valido`;
  renderLabelSummary(meta);
}

async function refreshOverlay(redraw = true) {
  if (!state.selectedPath || !state.labelMeta?.available || state.overlayMode === "off") {
    state.overlayPayload = null;
    labelClassSelect.disabled = true;
    if (redraw) {
      drawMap(getCurrentMapPayload());
      updateMapCaption();
      renderSpectrum();
    }
    return;
  }

  labelClassSelect.disabled = false;
  state.overlayPayload = await fetchJson(
    `/api/label-map?path=${encodeURIComponent(state.selectedPath)}&mode=${encodeURIComponent(state.overlayMode)}&class_index=${state.overlayClassIndex}`,
  );

  if (redraw) {
    drawMap(getCurrentMapPayload());
    updateMapCaption();
    renderSpectrum();
  }
}

async function loadSpaxel(x, y) {
  state.selectedSpaxel = { x, y };
  drawMap(getCurrentMapPayload());
  const payload = await fetchJson(`/api/spaxel?path=${encodeURIComponent(state.selectedPath)}&x=${x}&y=${y}`);
  state.spaxelPayload = payload;
  spaxelCaption.textContent = `Spaxel seleccionado: (${x}, ${y})`;
  renderSpectrum();
}

async function loadSlice(index, redraw = true) {
  state.currentSlice = Number(index);
  const payload = await fetchJson(`/api/slice?path=${encodeURIComponent(state.selectedPath)}&index=${state.currentSlice}`);
  state.slicePayload = payload;
  sliceLabel.textContent = `#${payload.index} · ${formatNumber(payload.wavelength, 1)} A`;
  if (redraw) {
    drawMap(getCurrentMapPayload());
    updateMapCaption();
  }
  renderSpectrum();
}

async function loadCube(path) {
  state.selectedPath = path;
  state.selectedSpaxel = null;
  state.spaxelPayload = null;
  state.slicePayload = null;
  state.overlayPayload = null;

  const [payload, labels] = await Promise.all([
    fetchJson(`/api/cube?path=${encodeURIComponent(path)}`),
    fetchJson(`/api/labels?path=${encodeURIComponent(path)}`),
  ]);

  state.cubeMeta = payload;
  state.labelMeta = labels;
  state.currentSlice = payload.default_slice_index;
  if (labels.available) {
    state.overlayMode = labels.default_mode;
    state.overlayClassIndex = labels.default_class_index;
  } else {
    state.overlayMode = "off";
    state.overlayClassIndex = 0;
  }

  sliceSlider.max = payload.shape.n_wave - 1;
  sliceSlider.value = payload.default_slice_index;

  const headerCards = [
    ["Archivo", payload.path],
    ["Canonical ID", payload.canonical_id],
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

  overlayOpacityValue.textContent = `${Math.round(state.overlayOpacity * 100)}%`;
  updateLabelControls(labels);

  await loadSlice(payload.default_slice_index, false);
  await refreshOverlay(false);
  drawMap(getCurrentMapPayload());
  updateMapCaption();
  spaxelCaption.textContent = "Haz click en el mapa para cargar un espectro";
  renderSpectrum();
}

async function bootstrap() {
  const payload = await fetchJson("/api/files");
  state.files = payload.files;

  if (state.files.length === 0) {
    cubeSelect.innerHTML = "<option>No se encontraron cubos 3D</option>";
    mapCaption.textContent = "Reconstruye un cubo y vuelve a cargar esta pagina";
    labelStatus.textContent = "No hay cubos compatibles en el workspace";
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
  drawMap(getCurrentMapPayload());
  updateMapCaption();
});

sliceButton.addEventListener("click", () => {
  state.currentMapMode = "slice";
  sliceButton.classList.add("active");
  overviewButton.classList.remove("active");
  drawMap(getCurrentMapPayload());
  updateMapCaption();
});

labelModeSelect.addEventListener("change", async (event) => {
  state.overlayMode = event.target.value;
  await refreshOverlay();
});

labelClassSelect.addEventListener("change", async (event) => {
  state.overlayClassIndex = Number(event.target.value);
  await refreshOverlay();
});

overlayOpacitySlider.addEventListener("input", () => {
  state.overlayOpacity = Number(overlayOpacitySlider.value) / 100;
  overlayOpacityValue.textContent = `${overlayOpacitySlider.value}%`;
  drawMap(getCurrentMapPayload());
  updateMapCaption();
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
  labelStatus.textContent = error.message;
  spectrumSvg.innerHTML = `<text class="empty-state" x="50%" y="50%" text-anchor="middle">${error.message}</text>`;
});
