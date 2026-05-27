const state = {
  offset: 0,
  limit: 96,
  sort: "name",
  query: "",
  total: 0,
  filteredTotal: 0,
  previewDir: "",
  exists: true,
  scannedAt: 0,
  items: [],
  selectedPath: null,
  selectedItem: null,
  requestId: 0,
  loading: false,
};

const searchInput = document.getElementById("search-input");
const sortSelect = document.getElementById("sort-select");
const pageSizeSelect = document.getElementById("page-size-select");
const prevButton = document.getElementById("prev-button");
const nextButton = document.getElementById("next-button");
const refreshButton = document.getElementById("refresh-button");
const catalogMeta = document.getElementById("catalog-meta");
const galleryCaption = document.getElementById("gallery-caption");
const imageGrid = document.getElementById("image-grid");
const selectedImage = document.getElementById("selected-image");
const selectionCaption = document.getElementById("selection-caption");
const emptyState = document.getElementById("empty-state");
const imageMeta = document.getElementById("image-meta");

function imageUrl(path) {
  return `/previews/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const digits = index === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[index]}`;
}

function formatDate(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "-";
  }
  return new Date(seconds * 1000).toLocaleString("es-AR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function createMetaCards(container, entries) {
  container.replaceChildren();
  entries.forEach(([label, value]) => {
    const card = document.createElement("div");
    const labelElement = document.createElement("label");
    const valueElement = document.createElement("strong");
    card.className = "meta-card";
    labelElement.textContent = label;
    valueElement.textContent = value;
    card.append(labelElement, valueElement);
    container.appendChild(card);
  });
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function setLoading(isLoading) {
  state.loading = isLoading;
  prevButton.disabled = isLoading || state.offset <= 0;
  nextButton.disabled = isLoading || state.offset + state.limit >= state.filteredTotal;
  refreshButton.disabled = isLoading;
}

function updateCatalogMeta() {
  const page = state.filteredTotal === 0 ? 0 : Math.floor(state.offset / state.limit) + 1;
  const pages = state.filteredTotal === 0 ? 0 : Math.ceil(state.filteredTotal / state.limit);
  createMetaCards(catalogMeta, [
    ["Total", state.total.toLocaleString("es-AR")],
    ["Encontradas", state.filteredTotal.toLocaleString("es-AR")],
    ["Pagina", `${page} / ${pages}`],
    ["Directorio", state.previewDir || "-"],
  ]);
}

function updateGalleryCaption() {
  if (!state.exists) {
    galleryCaption.textContent = "Directorio no encontrado";
    return;
  }
  if (state.filteredTotal === 0) {
    galleryCaption.textContent = "0 imagenes";
    return;
  }
  const start = state.offset + 1;
  const end = Math.min(state.offset + state.limit, state.filteredTotal);
  galleryCaption.textContent = `${start.toLocaleString("es-AR")}-${end.toLocaleString("es-AR")} de ${state.filteredTotal.toLocaleString("es-AR")}`;
}

function setActiveCard() {
  document.querySelectorAll(".image-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.path === state.selectedPath);
  });
}

function clearSelection() {
  state.selectedPath = null;
  state.selectedItem = null;
  selectedImage.removeAttribute("src");
  selectedImage.alt = "";
  selectedImage.classList.remove("visible");
  emptyState.style.display = "block";
  selectionCaption.textContent = "Sin seleccion";
  imageMeta.replaceChildren();
  setActiveCard();
}

function selectItem(item) {
  state.selectedPath = item.path;
  state.selectedItem = item;
  selectedImage.src = imageUrl(item.path);
  selectedImage.alt = item.stem;
  selectedImage.classList.add("visible");
  emptyState.style.display = "none";
  selectionCaption.textContent = item.name;
  createMetaCards(imageMeta, [
    ["Archivo", item.name],
    ["Ruta", item.path],
    ["Tamano", formatBytes(item.size)],
    ["Modificado", formatDate(item.modified)],
  ]);
  setActiveCard();
}

function renderEmpty(message, className = "empty-panel") {
  imageGrid.replaceChildren();
  const empty = document.createElement("div");
  empty.className = className;
  empty.textContent = message;
  imageGrid.appendChild(empty);
}

function renderGrid() {
  imageGrid.replaceChildren();
  if (!state.exists) {
    renderEmpty("No existe el directorio de previews configurado.", "error-panel");
    return;
  }
  if (state.items.length === 0) {
    renderEmpty("No hay previews PNG para mostrar.");
    return;
  }

  const fragment = document.createDocumentFragment();
  state.items.forEach((item) => {
    const card = document.createElement("button");
    const image = document.createElement("img");
    const name = document.createElement("span");
    const size = document.createElement("span");

    card.type = "button";
    card.className = "image-card";
    card.dataset.path = item.path;
    card.title = item.path;

    image.className = "image-thumb";
    image.src = imageUrl(item.path);
    image.alt = item.stem;
    image.loading = "lazy";
    image.decoding = "async";

    name.className = "image-name";
    name.textContent = item.name;
    size.className = "image-size";
    size.textContent = formatBytes(item.size);

    card.append(image, name, size);
    card.addEventListener("click", () => selectItem(item));
    fragment.appendChild(card);
  });
  imageGrid.appendChild(fragment);
  setActiveCard();
}

async function loadImages(options = {}) {
  const requestId = state.requestId + 1;
  state.requestId = requestId;
  state.offset = Number.isFinite(options.offset) ? Math.max(0, options.offset) : state.offset;
  setLoading(true);
  galleryCaption.textContent = "Cargando indice...";

  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
    q: state.query,
    sort: state.sort,
  });

  try {
    const payload = await fetchJson(`/api/images?${params.toString()}`);
    if (requestId !== state.requestId) {
      return;
    }

    state.previewDir = payload.preview_dir;
    state.exists = payload.exists;
    state.total = payload.total;
    state.filteredTotal = payload.filtered_total;
    state.offset = payload.offset;
    state.limit = payload.limit;
    state.scannedAt = payload.scanned_at;
    state.items = payload.items;

    updateCatalogMeta();
    updateGalleryCaption();
    renderGrid();

    const selectedOnPage = state.items.find((item) => item.path === state.selectedPath);
    if (selectedOnPage) {
      selectItem(selectedOnPage);
    } else if (state.items.length > 0) {
      const fallbackIndex = options.selectLast ? state.items.length - 1 : 0;
      selectItem(state.items[fallbackIndex]);
    } else {
      clearSelection();
    }
  } catch (error) {
    if (requestId !== state.requestId) {
      return;
    }
    renderEmpty(error.message, "error-panel");
    galleryCaption.textContent = "Error al cargar";
  } finally {
    if (requestId === state.requestId) {
      setLoading(false);
    }
  }
}

async function refreshCatalog() {
  setLoading(true);
  galleryCaption.textContent = "Actualizando indice...";
  try {
    await fetchJson("/api/refresh", { method: "POST" });
    await loadImages({ offset: 0 });
  } catch (error) {
    renderEmpty(error.message, "error-panel");
    galleryCaption.textContent = "Error al actualizar";
  } finally {
    setLoading(false);
  }
}

function selectedIndex() {
  return state.items.findIndex((item) => item.path === state.selectedPath);
}

function selectRelative(delta) {
  if (state.loading || state.items.length === 0) {
    return;
  }
  const index = selectedIndex();
  const nextIndex = index + delta;
  if (nextIndex >= 0 && nextIndex < state.items.length) {
    selectItem(state.items[nextIndex]);
    return;
  }
  if (delta > 0 && state.offset + state.limit < state.filteredTotal) {
    loadImages({ offset: state.offset + state.limit });
  } else if (delta < 0 && state.offset > 0) {
    loadImages({ offset: Math.max(0, state.offset - state.limit), selectLast: true });
  }
}

let searchTimer = null;
searchInput.addEventListener("input", () => {
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => {
    state.query = searchInput.value.trim();
    loadImages({ offset: 0 });
  }, 220);
});

sortSelect.addEventListener("change", () => {
  state.sort = sortSelect.value;
  loadImages({ offset: 0 });
});

pageSizeSelect.addEventListener("change", () => {
  state.limit = Number(pageSizeSelect.value);
  loadImages({ offset: 0 });
});

prevButton.addEventListener("click", () => {
  loadImages({ offset: Math.max(0, state.offset - state.limit) });
});

nextButton.addEventListener("click", () => {
  loadImages({ offset: state.offset + state.limit });
});

refreshButton.addEventListener("click", refreshCatalog);

document.addEventListener("keydown", (event) => {
  const tag = event.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
    return;
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    selectRelative(1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    selectRelative(-1);
  }
});

loadImages();
