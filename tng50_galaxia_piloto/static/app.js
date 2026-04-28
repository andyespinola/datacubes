const state = {
  config: null,
  component: "stars",
  quantity: "mass",
  view: "faceon",
  radiusKpc: 35,
  cloud: {
    manifest: null,
    detailKey: "equilibrada",
    yaw: 0.8,
    pitch: -0.55,
    zoom: 1.45,
    dragging: false,
    lastX: 0,
    lastY: 0,
    showStars: true,
    showGas: true,
    renderQueued: false,
    renderer: null,
    webglSupported: true,
  },
};

const els = {
  component: document.getElementById("component-select"),
  quantity: document.getElementById("quantity-select"),
  view: document.getElementById("view-select"),
  radius: document.getElementById("radius-range"),
  radiusValue: document.getElementById("radius-value"),
  summaryGrid: document.getElementById("summary-grid"),
  statusLine: document.getElementById("status-line"),
  mapTitle: document.getElementById("map-title"),
  mapSubtitle: document.getElementById("map-subtitle"),
  heatmap: document.getElementById("heatmap"),
  colorbar: document.getElementById("colorbar"),
  legendMin: document.getElementById("legend-min"),
  legendMid: document.getElementById("legend-mid"),
  legendMax: document.getElementById("legend-max"),
  axisX: document.getElementById("axis-x-label"),
  axisY: document.getElementById("axis-y-label"),
  heroRedshift: document.getElementById("hero-redshift"),
  heroStellarMass: document.getElementById("hero-stellar-mass"),
  heroSfr: document.getElementById("hero-sfr"),
  cloudCanvas: document.getElementById("cloud-canvas"),
  cloudStatus: document.getElementById("cloud-status"),
  cloudStats: document.getElementById("cloud-stats"),
  cloudStarsToggle: document.getElementById("cloud-stars-toggle"),
  cloudGasToggle: document.getElementById("cloud-gas-toggle"),
  cloudResetButton: document.getElementById("cloud-reset-button"),
  cloudDetailSelect: document.getElementById("cloud-detail-select"),
};

const CLOUD_STYLE = {
  stars: {
    tintColor: [0.988, 0.913, 0.784],
    pointBase: 1.15,
    pointBoost: 7.2,
  },
  gas: {
    tintColor: [0.690, 0.929, 0.949],
    pointBase: 1.55,
    pointBoost: 8.0,
  },
};

const CLOUD_VERTEX_SHADER = `
attribute vec4 a_particle;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_zoom;
uniform float u_radius;
uniform float u_aspect;
uniform float u_point_base;
uniform float u_point_boost;
uniform vec3 u_tint_color;
varying vec4 v_color;

vec3 rotateParticle(vec3 position) {
  float cosYaw = cos(u_yaw);
  float sinYaw = sin(u_yaw);
  float x1 = cosYaw * position.x - sinYaw * position.y;
  float y1 = sinYaw * position.x + cosYaw * position.y;
  float cosPitch = cos(u_pitch);
  float sinPitch = sin(u_pitch);
  float y2 = cosPitch * y1 - sinPitch * position.z;
  float z2 = sinPitch * y1 + cosPitch * position.z;
  return vec3(x1, y2, z2);
}

vec3 paletteColor(float t) {
  vec3 c0 = vec3(14.0, 11.0, 18.0) / 255.0;
  vec3 c1 = vec3(33.0, 27.0, 59.0) / 255.0;
  vec3 c2 = vec3(14.0, 76.0, 118.0) / 255.0;
  vec3 c3 = vec3(202.0, 122.0, 46.0) / 255.0;
  vec3 c4 = vec3(248.0, 212.0, 141.0) / 255.0;
  vec3 c5 = vec3(255.0, 248.0, 231.0) / 255.0;

  float scaled = clamp(t, 0.0, 0.999) * 5.0;
  if (scaled < 1.0) {
    return mix(c0, c1, scaled);
  }
  if (scaled < 2.0) {
    return mix(c1, c2, scaled - 1.0);
  }
  if (scaled < 3.0) {
    return mix(c2, c3, scaled - 2.0);
  }
  if (scaled < 4.0) {
    return mix(c3, c4, scaled - 3.0);
  }
  return mix(c4, c5, scaled - 4.0);
}

void main() {
  vec3 rotated = rotateParticle(a_particle.xyz);
  float radiusNorm3d = clamp(length(a_particle.xyz) / u_radius, 0.0, 1.0);
  float centrality = 1.0 - radiusNorm3d;
  float zoomSafe = max(u_zoom, 0.35);
  float cameraDistance = u_radius * (4.0 / pow(zoomSafe, 0.35));
  float denom = max(cameraDistance - rotated.z, max(u_radius * 0.03, cameraDistance * 0.08));
  float perspective = cameraDistance / denom;
  float sceneScale = (0.72 * zoomSafe) / u_radius;

  gl_Position = vec4(
    rotated.x * sceneScale * perspective / u_aspect,
    rotated.y * sceneScale * perspective,
    0.0,
    1.0
  );

  float depthNorm = clamp((rotated.z + u_radius) / (2.0 * u_radius), 0.0, 1.0);
  float alpha = clamp(0.045 + centrality * 0.16 + a_particle.w * 0.18 + depthNorm * 0.05, 0.04, 0.48);
  float size = max(1.0, u_point_base + perspective * (1.0 + a_particle.w * u_point_boost));
  float paletteT = clamp(centrality * 0.34 + pow(a_particle.w, 1.25) * 0.66, 0.0, 1.0);
  vec3 palette = paletteColor(paletteT);
  vec3 toned = mix(palette, u_tint_color, 0.06);
  gl_PointSize = size;
  v_color = vec4(toned * (0.48 + 0.68 * a_particle.w), alpha);
}
`;

const CLOUD_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;

void main() {
  vec2 coord = gl_PointCoord * 2.0 - 1.0;
  float radius2 = dot(coord, coord);
  if (radius2 > 1.0) {
    discard;
  }

  float glow = exp(-radius2 * 3.8);
  float alpha = v_color.a * (0.24 + glow * 0.76);
  vec3 rgb = mix(v_color.rgb * 0.35, v_color.rgb, glow);
  gl_FragColor = vec4(rgb, alpha);
}
`;

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/d";
  }
  return new Intl.NumberFormat("es-AR", {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  }).format(value);
}

function formatScientific(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/d";
  }
  return value.toExponential(2);
}

function setStatus(message, isError = false) {
  els.statusLine.textContent = message;
  els.statusLine.classList.toggle("status-error", isError);
}

function setCloudStatus(message) {
  els.cloudStatus.textContent = message;
}

function colorAt(t) {
  const stops = [
    [14, 11, 18],
    [33, 27, 59],
    [14, 76, 118],
    [202, 122, 46],
    [248, 212, 141],
    [255, 248, 231],
  ];
  const scaled = Math.max(0, Math.min(0.999, t)) * (stops.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;
  const a = stops[idx];
  const b = stops[Math.min(stops.length - 1, idx + 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

function drawColorbar(vmin, vmax) {
  const ctx = els.colorbar.getContext("2d");
  const width = els.colorbar.width;
  const height = els.colorbar.height;
  const image = ctx.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    const t = 1 - y / (height - 1);
    const [r, g, b] = colorAt(t);
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;
      image.data[idx + 0] = r;
      image.data[idx + 1] = g;
      image.data[idx + 2] = b;
      image.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  els.legendMax.textContent = formatNumber(vmax, 2);
  els.legendMid.textContent = formatNumber((vmin + vmax) / 2, 2);
  els.legendMin.textContent = formatNumber(vmin, 2);
}

function drawHeatmap(mapPayload) {
  const grid = mapPayload.data;
  const height = grid.length;
  const width = grid[0].length;
  const canvas = els.heatmap;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  const denom = mapPayload.vmax - mapPayload.vmin || 1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = grid[y][x];
      const idx = (y * width + x) * 4;
      if (value === null) {
        image.data[idx + 0] = 20;
        image.data[idx + 1] = 16;
        image.data[idx + 2] = 14;
        image.data[idx + 3] = 255;
        continue;
      }
      const t = (value - mapPayload.vmin) / denom;
      const [r, g, b] = colorAt(t);
      image.data[idx + 0] = r;
      image.data[idx + 1] = g;
      image.data[idx + 2] = b;
      image.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  drawColorbar(mapPayload.vmin, mapPayload.vmax);
  els.axisX.textContent = mapPayload.axis_labels.x;
  els.axisY.textContent = mapPayload.axis_labels.y;
}

function buildOptions(select, mapping, selectedValue) {
  select.innerHTML = "";
  Object.entries(mapping).forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    if (value === selectedValue) {
      option.selected = true;
    }
    select.appendChild(option);
  });
}

function renderSummary(summary) {
  els.heroRedshift.textContent = `z = ${formatNumber(summary.redshift, 2)}`;
  els.heroStellarMass.textContent = `M* = ${formatScientific(summary.stellar_mass_msun)} Msun`;
  els.heroSfr.textContent = `SFR = ${formatNumber(summary.sfr_msun_per_yr, 2)} Msun/yr`;

  const cards = [
    ["ID canónico", summary.canonical_id],
    ["Snapshot / subhalo", `${summary.snapshot} / ${summary.subhalo_id}`],
    ["R media masa estelar", `${formatNumber(summary.stellar_halfmass_radius_kpc, 2)} kpc`],
    ["Partículas estelares", formatNumber(summary.n_stellar_particles, 0)],
    ["Celdas de gas", formatNumber(summary.n_gas_cells, 0)],
    ["Masa de gas", `${formatScientific(summary.gas_mass_msun)} Msun`],
  ];

  if (summary.morphology) {
    cards.push(["Barra cinemática", summary.morphology.barred ? "Sí" : "No"]);
    cards.push(["Bar strength", formatNumber(summary.morphology.bar_strength, 3)]);
    cards.push(["Bar size", `${formatNumber(summary.morphology.bar_size_kpc, 2)} kpc`]);
  }

  els.summaryGrid.innerHTML = "";
  cards.forEach(([label, value]) => {
    const card = document.createElement("article");
    card.className = "summary-card";
    card.innerHTML = `<span class="label">${label}</span><span class="value">${value}</span>`;
    els.summaryGrid.appendChild(card);
  });
}

function updateQuantityOptions() {
  const quantityMeta = state.config.quantity_meta[state.component];
  const options = {};
  Object.entries(quantityMeta).forEach(([key, meta]) => {
    options[key] = meta.label;
  });
  if (!options[state.quantity]) {
    state.quantity = Object.keys(options)[0];
  }
  buildOptions(els.quantity, options, state.quantity);
}

function clamp(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value));
}

function resizeCanvasToDisplaySize(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const displayWidth = Math.max(320, Math.floor(canvas.clientWidth * dpr));
  const displayHeight = Math.max(320, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    return true;
  }
  return false;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "Error compilando shader";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || "Error vinculando programa WebGL";
    gl.deleteProgram(program);
    throw new Error(message);
  }
  return program;
}

function createCloudRenderer(canvas) {
  const gl =
    canvas.getContext("webgl", { alpha: true, antialias: true, depth: false, preserveDrawingBuffer: false }) ||
    canvas.getContext("experimental-webgl");
  if (!gl) {
    throw new Error("Tu navegador o GPU no exponen WebGL");
  }

  const program = createProgram(gl, CLOUD_VERTEX_SHADER, CLOUD_FRAGMENT_SHADER);
  const renderer = {
    gl,
    program,
    attribute: {
      particle: gl.getAttribLocation(program, "a_particle"),
    },
    uniform: {
      yaw: gl.getUniformLocation(program, "u_yaw"),
      pitch: gl.getUniformLocation(program, "u_pitch"),
      zoom: gl.getUniformLocation(program, "u_zoom"),
      radius: gl.getUniformLocation(program, "u_radius"),
      aspect: gl.getUniformLocation(program, "u_aspect"),
      pointBase: gl.getUniformLocation(program, "u_point_base"),
      pointBoost: gl.getUniformLocation(program, "u_point_boost"),
      tintColor: gl.getUniformLocation(program, "u_tint_color"),
    },
    componentBuffers: {
      stars: { buffer: gl.createBuffer(), count: 0 },
      gas: { buffer: gl.createBuffer(), count: 0 },
    },
  };

  gl.useProgram(program);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);
  gl.clearColor(12 / 255, 13 / 255, 18 / 255, 1);
  return renderer;
}

function ensureCloudRenderer() {
  if (state.cloud.renderer || !state.cloud.webglSupported) {
    return state.cloud.renderer;
  }
  try {
    state.cloud.renderer = createCloudRenderer(els.cloudCanvas);
    return state.cloud.renderer;
  } catch (error) {
    console.error(error);
    state.cloud.webglSupported = false;
    setCloudStatus("WebGL no está disponible en este navegador o GPU");
    els.cloudStats.textContent = "No pude inicializar el render acelerado";
    return null;
  }
}

function uploadCloudComponent(renderer, component, floatArray) {
  const bundle = renderer.componentBuffers[component];
  const gl = renderer.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, bundle.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, floatArray, gl.STATIC_DRAW);
  bundle.count = Math.floor(floatArray.length / 4);
}

function buildCloudPresetOptions() {
  const options = {};
  Object.entries(state.config.cloud_presets).forEach(([key, preset]) => {
    options[key] = preset.label;
  });
  buildOptions(els.cloudDetailSelect, options, state.cloud.detailKey);
}

function cloudPreset() {
  return state.config.cloud_presets[state.cloud.detailKey];
}

function cloudQuantityFor(component) {
  if (component === state.component) {
    return state.quantity;
  }
  return "mass";
}

function cloudBufferUrl(component) {
  const preset = cloudPreset();
  const maxPoints = component === "stars" ? preset.max_stars : preset.max_gas;
  const params = new URLSearchParams({
    component,
    quantity: cloudQuantityFor(component),
    radius_kpc: String(state.radiusKpc),
    max_points: String(maxPoints),
  });
  return `/api/particles3d/buffer?${params.toString()}`;
}

function cloudManifestUrl() {
  const preset = cloudPreset();
  const params = new URLSearchParams({
    radius_kpc: String(state.radiusKpc),
    max_stars: String(preset.max_stars),
    max_gas: String(preset.max_gas),
    star_quantity: cloudQuantityFor("stars"),
    gas_quantity: cloudQuantityFor("gas"),
  });
  return `/api/particles3d?${params.toString()}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function fetchFloat32Buffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const buffer = await response.arrayBuffer();
  return new Float32Array(buffer);
}

function scheduleCloudRender() {
  if (state.cloud.renderQueued) {
    return;
  }
  state.cloud.renderQueued = true;
  window.requestAnimationFrame(() => {
    state.cloud.renderQueued = false;
    renderCloudScene();
  });
}

function drawCloudPlaceholder(message) {
  const canvas = els.cloudCanvas;
  resizeCanvasToDisplaySize(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(250, 241, 227, 0.92)";
  ctx.font = `${Math.max(18, canvas.width * 0.018)}px Avenir Next, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function renderCloudScene() {
  const renderer = ensureCloudRenderer();
  if (!renderer) {
    drawCloudPlaceholder("WebGL no disponible");
    return;
  }

  const canvas = els.cloudCanvas;
  const gl = renderer.gl;
  if (resizeCanvasToDisplaySize(canvas)) {
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  const manifest = state.cloud.manifest;
  if (!manifest) {
    gl.clear(gl.COLOR_BUFFER_BIT);
    return;
  }

  const drawStars = state.cloud.showStars && renderer.componentBuffers.stars.count > 0;
  const drawGas = state.cloud.showGas && renderer.componentBuffers.gas.count > 0;
  if (!drawStars && !drawGas) {
    gl.clear(gl.COLOR_BUFFER_BIT);
    return;
  }

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(renderer.program);
  gl.uniform1f(renderer.uniform.yaw, state.cloud.yaw);
  gl.uniform1f(renderer.uniform.pitch, state.cloud.pitch);
  gl.uniform1f(renderer.uniform.zoom, state.cloud.zoom);
  gl.uniform1f(renderer.uniform.radius, manifest.radius_kpc);
  gl.uniform1f(renderer.uniform.aspect, canvas.width / canvas.height);

  if (drawStars) {
    renderCloudComponent(renderer, "stars");
  }
  if (drawGas) {
    renderCloudComponent(renderer, "gas");
  }
}

function renderCloudComponent(renderer, component) {
  const gl = renderer.gl;
  const style = CLOUD_STYLE[component];
  const bundle = renderer.componentBuffers[component];
  gl.bindBuffer(gl.ARRAY_BUFFER, bundle.buffer);
  gl.enableVertexAttribArray(renderer.attribute.particle);
  gl.vertexAttribPointer(renderer.attribute.particle, 4, gl.FLOAT, false, 0, 0);
  gl.uniform1f(renderer.uniform.pointBase, style.pointBase);
  gl.uniform1f(renderer.uniform.pointBoost, style.pointBoost);
  gl.uniform3fv(renderer.uniform.tintColor, new Float32Array(style.tintColor));
  gl.drawArrays(gl.POINTS, 0, bundle.count);
}

function updateCloudStats(manifest) {
  const presetLabel = cloudPreset().label;
  const starText =
    manifest.stars.sampled_count === manifest.stars.selected_count
      ? `Estrellas: mostrando las ${formatNumber(manifest.stars.selected_count, 0)} partículas`
      : `Estrellas: mostrando ${formatNumber(manifest.stars.sampled_count, 0)} de ${formatNumber(manifest.stars.selected_count, 0)} partículas`;
  const gasText =
    manifest.gas.sampled_count === manifest.gas.selected_count
      ? `Gas: mostrando las ${formatNumber(manifest.gas.selected_count, 0)} celdas`
      : `Gas: mostrando ${formatNumber(manifest.gas.sampled_count, 0)} de ${formatNumber(manifest.gas.selected_count, 0)} celdas`;
  els.cloudStats.textContent = `${starText} | ${gasText} | detalle ${presetLabel}`;
  const halfmass = manifest.halfmass_radius_kpc
    ? ` | radio de media masa estelar: ${formatNumber(manifest.halfmass_radius_kpc, 2)} kpc`
    : "";
  const active = manifest[state.component];
  const colorText = active ? ` | color: ${active.quantity_label}` : "";
  setCloudStatus(`Esfera de ${formatNumber(manifest.radius_kpc, 1)} kpc${halfmass}${colorText}`);
}

function resetCloudView() {
  state.cloud.yaw = 0.8;
  state.cloud.pitch = -0.55;
  state.cloud.zoom = 1.45;
  scheduleCloudRender();
}

function syncCloudToggleState() {
  state.cloud.showStars = els.cloudStarsToggle.checked;
  state.cloud.showGas = els.cloudGasToggle.checked;
  scheduleCloudRender();
}

function attachCloudInteractions() {
  const canvas = els.cloudCanvas;

  canvas.addEventListener("pointerdown", (event) => {
    state.cloud.dragging = true;
    state.cloud.lastX = event.clientX;
    state.cloud.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.cloud.dragging) {
      return;
    }
    const dx = event.clientX - state.cloud.lastX;
    const dy = event.clientY - state.cloud.lastY;
    state.cloud.lastX = event.clientX;
    state.cloud.lastY = event.clientY;
    state.cloud.yaw += dx * 0.008;
    state.cloud.pitch = clamp(state.cloud.pitch + dy * 0.006, -1.45, 1.45);
    scheduleCloudRender();
  });

  const stopDragging = () => {
    state.cloud.dragging = false;
  };

  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  canvas.addEventListener("pointerleave", stopDragging);
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.0016);
      state.cloud.zoom = clamp(state.cloud.zoom * factor, 0.45, 16.0);
      scheduleCloudRender();
    },
    { passive: false }
  );

  els.cloudStarsToggle.addEventListener("change", syncCloudToggleState);
  els.cloudGasToggle.addEventListener("change", syncCloudToggleState);
  els.cloudResetButton.addEventListener("click", resetCloudView);
  els.cloudDetailSelect.addEventListener("change", async (event) => {
    state.cloud.detailKey = event.target.value;
    await refreshCloud();
  });
  window.addEventListener("resize", scheduleCloudRender);
}

async function refreshCloud() {
  const renderer = ensureCloudRenderer();
  if (!renderer) {
    return;
  }

  setCloudStatus("Cargando buffers WebGL...");
  const [manifest, starsBuffer, gasBuffer] = await Promise.all([
    fetchJson(cloudManifestUrl()),
    fetchFloat32Buffer(cloudBufferUrl("stars")),
    fetchFloat32Buffer(cloudBufferUrl("gas")),
  ]);

  state.cloud.manifest = manifest;
  uploadCloudComponent(renderer, "stars", starsBuffer);
  uploadCloudComponent(renderer, "gas", gasBuffer);
  updateCloudStats(manifest);
  scheduleCloudRender();
}

async function refreshViz(includeCloud = false) {
  setStatus(includeCloud ? "Actualizando mapa y nube 3D..." : "Actualizando mapa...");
  const mapParams = new URLSearchParams({
    component: state.component,
    quantity: state.quantity,
    view: state.view,
    radius_kpc: String(state.radiusKpc),
    bins: String(state.config.defaults.bins),
  });

  const requests = [fetchJson(`/api/map?${mapParams.toString()}`)];
  if (includeCloud) {
    requests.push(refreshCloud());
  }

  const [mapPayload] = await Promise.all(requests);

  drawHeatmap(mapPayload);
  els.mapTitle.textContent = mapPayload.label;
  els.mapSubtitle.textContent = `${state.view === "faceon" ? "Vista face-on" : "Vista edge-on"} | campo de ${formatNumber(state.radiusKpc, 1)} kpc`;
  setStatus("Listo");
}

async function initialize() {
  try {
    state.config = await fetchJson("/api/config");
    const defaults = state.config.defaults;
    state.component = defaults.component;
    state.quantity = defaults.quantity;
    state.view = defaults.view;
    state.radiusKpc = defaults.radius_kpc;
    state.cloud.detailKey = defaults.cloud_detail || "equilibrada";

    buildOptions(els.component, { stars: "Estrellas", gas: "Gas" }, state.component);
    buildOptions(els.view, state.config.views, state.view);
    buildCloudPresetOptions();
    updateQuantityOptions();

    els.radius.min = state.config.limits.radius_kpc.min;
    els.radius.max = state.config.limits.radius_kpc.max;
    els.radius.value = String(state.radiusKpc);
    els.radiusValue.textContent = `${formatNumber(state.radiusKpc, 1)} kpc`;

    renderSummary(state.config.summary);
    attachCloudInteractions();
    ensureCloudRenderer();

    els.component.addEventListener("change", async (event) => {
      state.component = event.target.value;
      updateQuantityOptions();
      await refreshViz(true);
    });
    els.quantity.addEventListener("change", async (event) => {
      state.quantity = event.target.value;
      await refreshViz(true);
    });
    els.view.addEventListener("change", async (event) => {
      state.view = event.target.value;
      await refreshViz(false);
    });
    els.radius.addEventListener("input", (event) => {
      state.radiusKpc = Number(event.target.value);
      els.radiusValue.textContent = `${formatNumber(state.radiusKpc, 1)} kpc`;
    });
    els.radius.addEventListener("change", async () => {
      await refreshViz(true);
    });

    await refreshViz(true);
  } catch (error) {
    console.error(error);
    setStatus("No pude cargar el viewer", true);
    setCloudStatus("No pude cargar la nube 3D");
    els.cloudStats.textContent = "Error inicializando el render 3D";
  }
}

initialize();
