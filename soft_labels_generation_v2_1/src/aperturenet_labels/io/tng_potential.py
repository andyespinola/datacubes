from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import requests

from aperturenet_labels.core.constants import TNG_SIMULATION
from aperturenet_labels.io.assets import parse_galaxy_id


PARTTYPE_STARS = 4


class TNGPotentialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SnapshotChunkRange:
    file_number: int
    local_start: int
    count: int
    global_start: int


@dataclass(frozen=True, slots=True)
class PotentialExtractionResult:
    galaxy_id: str
    snapshot: int
    subhalo_id: int
    output_path: Path
    n_particles: int
    chunks_used: tuple[int, ...]
    offset_path: Path
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class StellarPotentialCache:
    path: Path
    particle_ids: np.ndarray
    potential_raw: np.ndarray
    potential_units_raw: str


def potential_cache_path(cache_dir: str | Path, galaxy_id: str) -> Path:
    return Path(cache_dir) / f"{galaxy_id}.stellar_potential.hdf5"


def load_stellar_potential_cache(
    cache_dir: str | Path,
    galaxy_id: str,
    selected_indices: np.ndarray,
    particle_ids: np.ndarray | None = None,
) -> StellarPotentialCache | None:
    path = potential_cache_path(cache_dir, galaxy_id)
    if not path.exists():
        return None
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        ids_dataset = handle["ParticleIDs"]
        potential_dataset = handle["Potential"]
        if np.any(selected_indices < 0) or np.any(selected_indices >= ids_dataset.shape[0]):
            raise TNGPotentialError(f"Selected indices exceed potential cache length for {galaxy_id}: {path}")
        order = np.argsort(selected_indices)
        sorted_indices = selected_indices[order]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        ids = np.asarray(ids_dataset[sorted_indices], dtype=np.uint64)[inverse]
        potential = np.asarray(potential_dataset[sorted_indices], dtype=np.float64)[inverse]
        units = str(handle.attrs.get("potential_units_raw", "(km/s)^2 / a"))
    if particle_ids is not None:
        particle_ids = np.asarray(particle_ids, dtype=np.uint64)
        if particle_ids.shape != ids.shape or not np.array_equal(particle_ids, ids):
            raise TNGPotentialError(f"ParticleIDs in potential cache do not match selected cutout particles for {galaxy_id}")
    return StellarPotentialCache(path=path, particle_ids=ids, potential_raw=potential, potential_units_raw=units)


def api_headers(api_key: str) -> dict[str, str]:
    return {"API-Key": api_key}


def file_url(simulation: str, path: str) -> str:
    return f"https://www.tng-project.org/api/{simulation}/files/{path}"


def offsets_url(simulation: str, snapshot: int) -> str:
    return file_url(simulation, f"offsets.{int(snapshot)}.hdf5")


def snapshot_chunk_url(simulation: str, snapshot: int, file_number: int) -> str:
    return file_url(simulation, f"snapshot-{int(snapshot)}.{int(file_number)}.hdf5")


def download_stream_atomic(
    url: str,
    output_path: str | Path,
    api_key: str,
    force: bool = False,
    timeout: int | tuple[int, int] = (45, 120),
    retries: int = 3,
    backoff_seconds: float = 10.0,
    resume: bool = True,
    progress_label: str = "",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.exists() and not force:
        return output_path
    if force and output_path.exists():
        output_path.unlink()
    if force and part_path.exists():
        part_path.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        response: requests.Response | None = None
        resume_from = part_path.stat().st_size if resume and part_path.exists() else 0
        try:
            headers = api_headers(api_key)
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            if progress_label:
                print(f"{progress_label}: request attempt {attempt}/{retries} from {resume_from / 1024**2:.1f} MiB", flush=True)
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if response.status_code >= 500:
                raise TNGPotentialError(f"HTTP {response.status_code}: {response.text[:300]}")
            if response.status_code >= 400:
                raise TNGPotentialError(f"HTTP {response.status_code} for {url}: {response.text[:300]}")

            mode = "ab" if resume_from > 0 and response.status_code == 206 else "wb"
            if resume_from > 0 and response.status_code != 206:
                resume_from = 0
            expected_total = None
            content_range = response.headers.get("content-range", "")
            if "/" in content_range:
                try:
                    expected_total = int(content_range.rsplit("/", 1)[1])
                except ValueError:
                    expected_total = None
            elif response.headers.get("content-length"):
                try:
                    expected_total = resume_from + int(response.headers["content-length"])
                except ValueError:
                    expected_total = None

            written = resume_from
            last_report = time.monotonic()
            if progress_label:
                total_text = f" / {expected_total / 1024**2:.1f} MiB" if expected_total else ""
                print(f"{progress_label}: streaming {written / 1024**2:.1f} MiB{total_text}", flush=True)
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if progress_label and now - last_report >= 30.0:
                        total_text = f" / {expected_total / 1024**2:.1f} MiB" if expected_total else ""
                        print(f"{progress_label}: {written / 1024**2:.1f} MiB{total_text}", flush=True)
                        last_report = now
            if expected_total is not None and written != expected_total:
                raise TNGPotentialError(
                    f"incomplete download: wrote {written} bytes, expected {expected_total} bytes"
                )
            part_path.replace(output_path)
            if progress_label:
                print(f"{progress_label}: complete {written / 1024**2:.1f} MiB", flush=True)
            return output_path
        except requests.RequestException as exc:
            last_error = exc
            if progress_label:
                print(f"{progress_label}: attempt {attempt}/{retries} failed: {exc}", flush=True)
        except TNGPotentialError as exc:
            last_error = exc
            if progress_label:
                print(f"{progress_label}: attempt {attempt}/{retries} failed: {exc}", flush=True)
        finally:
            if response is not None:
                response.close()
        if attempt < retries:
            time.sleep(backoff_seconds * attempt)
    raise TNGPotentialError(f"Failed to download {url}: {last_error}")


def check_url(url: str, api_key: str, timeout: int = 120) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=api_headers(api_key), timeout=timeout, stream=True)
        first = next(response.iter_content(chunk_size=64), b"")
        result = {
            "url": url,
            "status_code": int(response.status_code),
            "content_type": response.headers.get("content-type", ""),
            "content_length": response.headers.get("content-length", ""),
            "content_disposition": response.headers.get("content-disposition", ""),
            "first_bytes_hex": first[:16].hex(),
        }
        response.close()
        return result
    except requests.RequestException as exc:
        return {"url": url, "status_code": 0, "error": str(exc)}


def read_local_stellar_particle_ids(cutout_path: str | Path) -> np.ndarray:
    with h5py.File(cutout_path, "r") as handle:
        if "PartType4/ParticleIDs" not in handle:
            raise TNGPotentialError(f"{cutout_path} does not contain PartType4/ParticleIDs")
        return np.asarray(handle["PartType4/ParticleIDs"], dtype=np.uint64)


def _dataset(handle: h5py.File, candidates: tuple[str, ...]) -> h5py.Dataset:
    for candidate in candidates:
        if candidate in handle:
            return handle[candidate]
    available: list[str] = []
    handle.visititems(lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None)
    raise TNGPotentialError(f"Missing any of datasets {candidates}; available datasets include {available[:30]}")


def read_subhalo_star_offset(offsets_path: str | Path, subhalo_id: int, part_type: int = PARTTYPE_STARS) -> int:
    with h5py.File(offsets_path, "r") as handle:
        dataset = _dataset(handle, ("Subhalo/SnapByType", "Subhalo/OffsetType"))
        if subhalo_id >= dataset.shape[0]:
            raise TNGPotentialError(f"subhalo_id={subhalo_id} outside offsets shape={dataset.shape}")
        offset = int(dataset[int(subhalo_id), int(part_type)])
        if offset < 0:
            raise TNGPotentialError(f"No PartType{part_type} offset for subhalo_id={subhalo_id}")
        return offset


def read_file_offsets_by_type(offsets_path: str | Path, part_type: int = PARTTYPE_STARS) -> np.ndarray:
    with h5py.File(offsets_path, "r") as handle:
        dataset = _dataset(handle, ("FileOffsets/SnapByType", "FileOffsets/OffsetType"))
        values = np.asarray(dataset[:, int(part_type)], dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise TNGPotentialError(f"Invalid FileOffsets/SnapByType in {offsets_path}")
    return values


def chunk_ranges_for_global_range(file_offsets: np.ndarray, global_start: int, count: int) -> list[SnapshotChunkRange]:
    if count <= 0:
        return []
    offsets = np.asarray(file_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size == 0:
        raise TNGPotentialError("file_offsets must be a non-empty 1D array")
    global_end = int(global_start) + int(count)
    cursor = int(global_start)
    ranges: list[SnapshotChunkRange] = []
    while cursor < global_end:
        file_number = int(np.searchsorted(offsets, cursor, side="right") - 1)
        if file_number < 0:
            raise TNGPotentialError(f"global_start={global_start} precedes first file offset={offsets[0]}")
        next_start = int(offsets[file_number + 1]) if file_number + 1 < offsets.size else global_end
        if next_start <= cursor:
            raise TNGPotentialError(f"Non-increasing file offsets around file {file_number}")
        take = min(global_end - cursor, next_start - cursor)
        ranges.append(
            SnapshotChunkRange(
                file_number=file_number,
                local_start=int(cursor - offsets[file_number]),
                count=int(take),
                global_start=int(cursor),
            )
        )
        cursor += take
    return ranges


def match_potential_to_local_ids(
    local_ids: np.ndarray,
    snapshot_ids: np.ndarray,
    snapshot_potential: np.ndarray,
) -> np.ndarray:
    local_ids = np.asarray(local_ids, dtype=np.uint64)
    snapshot_ids = np.asarray(snapshot_ids, dtype=np.uint64)
    snapshot_potential = np.asarray(snapshot_potential)
    if snapshot_ids.shape[0] != snapshot_potential.shape[0]:
        raise TNGPotentialError("snapshot_ids and snapshot_potential length mismatch")
    if local_ids.shape[0] != snapshot_ids.shape[0]:
        raise TNGPotentialError(f"local ids length {local_ids.shape[0]} != snapshot ids length {snapshot_ids.shape[0]}")
    if np.array_equal(local_ids, snapshot_ids):
        return np.asarray(snapshot_potential)
    order = np.argsort(snapshot_ids)
    sorted_ids = snapshot_ids[order]
    positions = np.searchsorted(sorted_ids, local_ids)
    if np.any(positions >= sorted_ids.size) or np.any(sorted_ids[positions] != local_ids):
        missing = local_ids[(positions >= sorted_ids.size) | (sorted_ids[np.minimum(positions, sorted_ids.size - 1)] != local_ids)]
        raise TNGPotentialError(f"Could not match {missing.size} local ParticleIDs to snapshot ParticleIDs")
    return np.asarray(snapshot_potential)[order][positions]


def read_snapshot_chunk_slice(chunk_path: str | Path, local_start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(chunk_path, "r") as handle:
        group_name = f"PartType{PARTTYPE_STARS}"
        if group_name not in handle:
            raise TNGPotentialError(f"{chunk_path} has no {group_name}")
        group = handle[group_name]
        if "ParticleIDs" not in group or "Potential" not in group:
            raise TNGPotentialError(f"{chunk_path} lacks {group_name}/ParticleIDs or {group_name}/Potential")
        stop = int(local_start) + int(count)
        ids = np.asarray(group["ParticleIDs"][int(local_start) : stop], dtype=np.uint64)
        potential = np.asarray(group["Potential"][int(local_start) : stop])
    return ids, potential


def default_phase2_cutout_path(data_dir: str | Path, galaxy_id: str) -> Path:
    data_dir = Path(data_dir)
    phase2 = data_dir / f"{galaxy_id}.cutout_phase2.hdf5"
    if phase2.exists():
        return phase2
    return data_dir / f"{galaxy_id}.cutout.hdf5"


def extract_stellar_potential_cache(
    galaxy_id: str,
    data_dir: str | Path,
    snapshot_cache_dir: str | Path,
    output_dir: str | Path,
    api_key: str,
    simulation: str = TNG_SIMULATION,
    force_download: bool = False,
    overwrite: bool = False,
    download_retries: int = 3,
    download_backoff_seconds: float = 10.0,
) -> PotentialExtractionResult:
    started = time.monotonic()
    snapshot, subhalo_id = parse_galaxy_id(galaxy_id)
    data_dir = Path(data_dir)
    snapshot_cache_dir = Path(snapshot_cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{galaxy_id}.stellar_potential.hdf5"
    if output_path.exists() and not overwrite:
        with h5py.File(output_path, "r") as handle:
            chunks = tuple(int(x) for x in handle.attrs.get("chunks_used", []))
            n_particles = int(handle["ParticleIDs"].shape[0])
            offset_path = Path(str(handle.attrs.get("offset_path", "")))
        return PotentialExtractionResult(galaxy_id, snapshot, subhalo_id, output_path, n_particles, chunks, offset_path, 0.0)

    local_ids = read_local_stellar_particle_ids(default_phase2_cutout_path(data_dir, galaxy_id))
    offset_path = snapshot_cache_dir / "offsets" / f"offsets.{snapshot}.hdf5"
    download_stream_atomic(
        offsets_url(simulation, snapshot),
        offset_path,
        api_key,
        force=force_download,
        retries=download_retries,
        backoff_seconds=download_backoff_seconds,
        progress_label=f"{galaxy_id} offsets.{snapshot}",
    )
    subhalo_offset = read_subhalo_star_offset(offset_path, subhalo_id)
    file_offsets = read_file_offsets_by_type(offset_path)
    ranges = chunk_ranges_for_global_range(file_offsets, subhalo_offset, int(local_ids.size))

    chunk_ids: list[np.ndarray] = []
    chunk_potentials: list[np.ndarray] = []
    for item in ranges:
        chunk_path = snapshot_cache_dir / f"snapshot_{snapshot:03d}" / f"snapshot-{snapshot}.{item.file_number}.hdf5"
        download_stream_atomic(
            snapshot_chunk_url(simulation, snapshot, item.file_number),
            chunk_path,
            api_key,
            force=force_download,
            retries=download_retries,
            backoff_seconds=download_backoff_seconds,
            progress_label=f"{galaxy_id} snapshot-{snapshot}.{item.file_number}",
        )
        ids, potential = read_snapshot_chunk_slice(chunk_path, item.local_start, item.count)
        chunk_ids.append(ids)
        chunk_potentials.append(potential)

    snapshot_ids = np.concatenate(chunk_ids) if chunk_ids else np.asarray([], dtype=np.uint64)
    snapshot_potential = np.concatenate(chunk_potentials) if chunk_potentials else np.asarray([], dtype=np.float32)
    ordered_potential = match_potential_to_local_ids(local_ids, snapshot_ids, snapshot_potential)

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()
    with h5py.File(part_path, "w") as handle:
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["simulation"] = simulation
        handle.attrs["galaxy_id"] = galaxy_id
        handle.attrs["snapshot"] = int(snapshot)
        handle.attrs["subhalo_id"] = int(subhalo_id)
        handle.attrs["source"] = "TNG full snapshot chunks"
        handle.attrs["offset_path"] = str(offset_path)
        handle.attrs["subhalo_star_global_offset"] = int(subhalo_offset)
        handle.attrs["chunks_used"] = np.asarray([item.file_number for item in ranges], dtype=np.int32)
        handle.attrs["potential_units_raw"] = "(km/s)^2 / a"
        handle.create_dataset("ParticleIDs", data=local_ids, compression="lzf")
        handle.create_dataset("Potential", data=np.asarray(ordered_potential), compression="lzf")
        handle.create_dataset("snapshot_ranges_json", data=np.bytes_(json.dumps([asdict(item) for item in ranges], sort_keys=True)))
    part_path.replace(output_path)
    return PotentialExtractionResult(
        galaxy_id=galaxy_id,
        snapshot=snapshot,
        subhalo_id=subhalo_id,
        output_path=output_path,
        n_particles=int(local_ids.size),
        chunks_used=tuple(item.file_number for item in ranges),
        offset_path=offset_path,
        elapsed_seconds=float(time.monotonic() - started),
    )
