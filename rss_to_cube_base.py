import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits


def find_hdu(hdul, preferred_names, ndim=None):
    """
    Busca una extensión por nombre; si no la encuentra, intenta inferirla por ndim.
    """
    name_to_idx = {h.name.upper(): i for i, h in enumerate(hdul)}
    for name in preferred_names:
        if name.upper() in name_to_idx:
            idx = name_to_idx[name.upper()]
            data = hdul[idx].data
            if data is not None and (ndim is None or np.ndim(data) == ndim):
                return idx, hdul[idx]

    # Fallback: buscar por dimensionalidad
    if ndim is not None:
        for i, h in enumerate(hdul):
            data = h.data
            if data is not None and np.ndim(data) == ndim:
                return i, h

    return None, None


def get_hdu(hdul, selector):
    """
    Obtiene una HDU por índice o por EXTNAME.
    """
    if selector is None:
        return None, None

    try:
        idx = int(selector)
    except (TypeError, ValueError):
        idx = None

    if idx is not None:
        if idx < 0 or idx >= len(hdul):
            raise IndexError(f"HDU fuera de rango: {selector}")
        return idx, hdul[idx]

    wanted = str(selector).strip().upper()
    for i, hdu in enumerate(hdul):
        if hdu.name.upper() == wanted:
            return i, hdu

    raise KeyError(f"No existe ninguna HDU con nombre/EXTNAME {selector!r}")


def summarize_hdul(hdul):
    print("\n=== EXTENSIONES FITS ===")
    for i, h in enumerate(hdul):
        shape = None if h.data is None else h.data.shape
        print(f"[{i:02d}] name={h.name!r:20s} shape={shape}")

    print("\n=== HEADER PRINCIPAL ===")
    for key in ("OBJECT", "PIPELINE", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"):
        if key in hdul[0].header:
            print(f"{key}: {hdul[0].header[key]}")


def looks_like_wave_axis(header, axis):
    ctype = str(header.get(f"CTYPE{axis}", "")).upper()
    cunit = str(header.get(f"CUNIT{axis}", "")).upper()
    return (
        "WAVE" in ctype
        or "LAMBDA" in ctype
        or "AWAV" in ctype
        or "WAVELENGTH" in cunit
    )


def looks_like_fiber_axis(header, axis):
    ctype = str(header.get(f"CTYPE{axis}", "")).upper()
    return "FIBER" in ctype or "FIBRE" in ctype


def infer_wave_from_header(header):
    """
    Reconstruye un eje espectral lineal a partir del header FITS.
    """
    naxis = int(header.get("NAXIS", 0))
    wave_axis = None

    for axis in range(1, naxis + 1):
        if looks_like_wave_axis(header, axis):
            wave_axis = axis
            break

    if wave_axis is None:
        for axis in range(naxis, 0, -1):
            if f"CRVAL{axis}" in header and (
                f"CDELT{axis}" in header or f"CD{axis}_{axis}" in header
            ):
                wave_axis = axis
                break

    if wave_axis is None:
        return None, None

    n_wave = int(header.get(f"NAXIS{wave_axis}", 0))
    crval = header.get(f"CRVAL{wave_axis}")
    cdelt = header.get(f"CDELT{wave_axis}", header.get(f"CD{wave_axis}_{wave_axis}"))
    crpix = float(header.get(f"CRPIX{wave_axis}", 1.0))

    if n_wave <= 0 or crval is None or cdelt is None:
        return None, wave_axis

    pix = np.arange(n_wave, dtype=np.float64) + 1.0
    wave = float(crval) + (pix - crpix) * float(cdelt)
    return wave, wave_axis


def normalize_flux_orientation(flux, wave=None, xpos=None, ypos=None, header=None):
    """
    Devuelve FLUX con forma (n_fibers, n_wave).
    """
    flux = np.asarray(flux)
    if flux.ndim != 2:
        raise ValueError("FLUX debe ser 2D")

    score_as_is = 0
    score_transpose = 0
    reasons_as_is = []
    reasons_transpose = []

    wave_len = None if wave is None else int(np.asarray(wave).size)
    if wave_len is not None:
        if flux.shape[1] == wave_len:
            score_as_is += 2
            reasons_as_is.append("shape[1] coincide con len(wave)")
        if flux.shape[0] == wave_len:
            score_transpose += 2
            reasons_transpose.append("shape[0] coincide con len(wave)")

    for label, values in (("xpos", xpos), ("ypos", ypos)):
        if values is None:
            continue
        arr = np.asarray(values)
        if arr.ndim == 1:
            if flux.shape[0] == arr.size:
                score_as_is += 1
                reasons_as_is.append(f"shape[0] coincide con {label}")
            if flux.shape[1] == arr.size:
                score_transpose += 1
                reasons_transpose.append(f"shape[1] coincide con {label}")

    if header is not None and header.get("NAXIS", 0) == 2:
        if looks_like_fiber_axis(header, 1) and looks_like_wave_axis(header, 2):
            score_transpose += 2
            reasons_transpose.append("header FITS: axis1=fiber, axis2=wavelength")
        elif looks_like_wave_axis(header, 1) and looks_like_fiber_axis(header, 2):
            score_as_is += 2
            reasons_as_is.append("header FITS: axis1=wavelength, axis2=fiber")

    if score_transpose > score_as_is:
        reason = "; ".join(reasons_transpose) or "heurística de transposición"
        return np.asarray(flux.T), f"transposed -> (n_fibers, n_wave) [{reason}]"

    if score_as_is > score_transpose:
        reason = "; ".join(reasons_as_is) or "heurística de orientación"
        return np.asarray(flux), f"kept as-is -> (n_fibers, n_wave) [{reason}]"

    if flux.shape[0] <= flux.shape[1]:
        return np.asarray(flux), "kept as-is -> (n_fibers, n_wave) [fallback: axis corto=fibras]"

    return np.asarray(flux.T), "transposed -> (n_fibers, n_wave) [fallback: axis corto=fibras]"


def collapse_positions(values, n_fibers, n_wave, label):
    """
    Reduce posiciones 1D/2D a un vector 1D por fibra.
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        if arr.size != n_fibers:
            raise ValueError(f"{label} no coincide con el número de fibras: {arr.size} != {n_fibers}")
        return arr

    if arr.ndim == 2:
        if arr.shape == (n_fibers, n_wave):
            return np.nanmean(arr, axis=1)
        if arr.shape == (n_wave, n_fibers):
            return np.nanmean(arr, axis=0)
        if arr.shape[0] == n_fibers:
            return np.nanmean(arr, axis=1)
        if arr.shape[1] == n_fibers:
            return np.nanmean(arr, axis=0)

    raise ValueError(
        f"{label} tiene forma no soportada {arr.shape}; "
        f"se esperaba 1D o 2D compatible con ({n_fibers}, {n_wave})"
    )


def infer_rss_parts(hdul, flux_selector=None):
    """
    Intenta localizar flujo, longitud de onda y posiciones x/y.
    """
    if flux_selector is not None:
        flux_idx, flux_hdu = get_hdu(hdul, flux_selector)
    else:
        flux_idx, flux_hdu = find_hdu(hdul, ["FLUX", "RSS", "SCI", "PRIMARY"], ndim=2)
        if flux_hdu is None and hdul[0].data is not None and np.ndim(hdul[0].data) == 2:
            flux_idx, flux_hdu = 0, hdul[0]

    wave_idx, wave_hdu = find_hdu(hdul, ["WAVE", "LAMBDA", "WAVELENGTH"])
    xpos_idx, xpos_hdu = find_hdu(hdul, ["XPOS", "X_IFU", "XIFU", "XS", "X"])
    ypos_idx, ypos_hdu = find_hdu(hdul, ["YPOS", "Y_IFU", "YIFU", "YS", "Y"])

    flux = None
    flux_raw_shape = None
    flux_orientation = None
    wave = None
    wave_axis = None
    wave_source = None

    if wave_hdu is not None:
        wave = np.asarray(wave_hdu.data).reshape(-1)
        wave_source = f"HDU[{wave_idx}] {wave_hdu.name}"
    elif flux_hdu is not None:
        wave, wave_axis = infer_wave_from_header(flux_hdu.header)
        if wave is not None:
            wave_source = f"header axis {wave_axis} de HDU[{flux_idx}] {flux_hdu.name}"

    xpos = None if xpos_hdu is None else np.asarray(xpos_hdu.data)
    ypos = None if ypos_hdu is None else np.asarray(ypos_hdu.data)

    if flux_hdu is not None:
        flux_raw = np.asarray(flux_hdu.data)
        flux_raw_shape = flux_raw.shape
        flux, flux_orientation = normalize_flux_orientation(
            flux_raw,
            wave=wave,
            xpos=xpos,
            ypos=ypos,
            header=flux_hdu.header,
        )

    result = {
        "flux_idx": flux_idx,
        "wave_idx": wave_idx,
        "xpos_idx": xpos_idx,
        "ypos_idx": ypos_idx,
        "flux_name": None if flux_hdu is None else flux_hdu.name,
        "wave_name": None if wave_hdu is None else wave_hdu.name,
        "flux_raw_shape": flux_raw_shape,
        "flux_orientation": flux_orientation,
        "wave_source": wave_source,
        "wave_axis": wave_axis,
        "flux_header": None if flux_hdu is None else flux_hdu.header,
        "flux": flux,
        "wave": wave,
        "xpos": xpos,
        "ypos": ypos,
    }
    return result


def print_rss_summary(parts):
    print("\n=== RESUMEN RSS ===")
    for key in ("flux_idx", "wave_idx", "xpos_idx", "ypos_idx"):
        print(f"{key}: {parts[key]}")

    if parts["flux"] is not None:
        print(f"FLUX source: HDU[{parts['flux_idx']}] {parts['flux_name']!r}")
        if parts["flux_raw_shape"] is not None:
            print(f"FLUX raw shape: {parts['flux_raw_shape']}")
        print(f"FLUX shape: {parts['flux'].shape}")
        print(f"FLUX orientation: {parts['flux_orientation']}")
    if parts["wave"] is not None:
        print(f"WAVE shape: {parts['wave'].shape}")
        print(f"λ_min={parts['wave'][0]:.2f}, λ_max={parts['wave'][-1]:.2f}")
        print(f"WAVE source: {parts['wave_source']}")

    if parts["xpos"] is not None:
        print(f"XPOS shape: {parts['xpos'].shape}")
    if parts["ypos"] is not None:
        print(f"YPOS shape: {parts['ypos'].shape}")


def reconstruct_cube_simple(flux, xpos, ypos, gridsize=0.5, sigma=1.0):
    """
    Reconstrucción SIMPLE de validación:
    - flux:  (n_fibers, n_wave)
    - xpos:  (n_fibers, n_wave) o (n_fibers, ) o similar
    - ypos:  (n_fibers, n_wave) o (n_fibers, ) o similar

    Usa una grilla regular y pesos gaussianos en el plano espacial.
    No reproduce el DRP oficial; solo sirve como aproximación controlada.
    """
    if flux.ndim != 2:
        raise ValueError("FLUX debe ser 2D: (n_fibers, n_wave)")

    n_fibers, n_wave = flux.shape

    x = collapse_positions(xpos, n_fibers, n_wave, "XPOS")
    y = collapse_positions(ypos, n_fibers, n_wave, "YPOS")

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    gx = np.arange(xmin, xmax + gridsize, gridsize)
    gy = np.arange(ymin, ymax + gridsize, gridsize)

    nx, ny = len(gx), len(gy)
    cube = np.full((n_wave, ny, nx), np.nan, dtype=np.float32)

    xx, yy = np.meshgrid(gx, gy)

    # Pesos espaciales fijos por fibra sobre toda la grilla
    weights = []
    for i in range(n_fibers):
        r2 = (xx - x[i]) ** 2 + (yy - y[i]) ** 2
        w = np.exp(-0.5 * r2 / (sigma ** 2))
        weights.append(w)
    weights = np.asarray(weights, dtype=np.float32)  # (n_fibers, ny, nx)

    flux32 = np.asarray(flux, dtype=np.float32)
    wsum = np.sum(weights, axis=0)
    good = wsum > 0
    num = np.einsum("fw,fyx->wyx", flux32, weights, optimize=True)
    np.divide(num, wsum[None, :, :], out=cube, where=good[None, :, :])

    return cube, gx, gy


def save_regular_axis(header, axis, values, ctype, cunit):
    if values is None or len(values) == 0:
        return

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    header[f"CTYPE{axis}"] = ctype
    header[f"CRPIX{axis}"] = 1.0
    header[f"CRVAL{axis}"] = float(values[0])
    header[f"CUNIT{axis}"] = cunit

    if len(values) > 1:
        step = np.diff(values)
        if np.allclose(step, step[0]):
            header[f"CDELT{axis}"] = float(step[0])


def save_cube_fits(outpath, cube, wave=None, gx=None, gy=None, source_header=None, source_name=None):
    """
    Guarda el cubo reconstruido de validación en FITS.
    Eje 0 del array numpy = lambda.
    """
    header = fits.Header()
    if source_header is not None:
        for key in ("OBJECT", "REDSHIFT", "PSF", "FOV", "UNITS", "BUNIT"):
            if key in source_header:
                header[key] = source_header[key]
        if "UNITS" in header and "BUNIT" not in header:
            header["BUNIT"] = header["UNITS"]
    if source_name:
        header["SRCEXT"] = source_name

    save_regular_axis(header, axis=1, values=gx, ctype="X_IFU", cunit="arcsec")
    save_regular_axis(header, axis=2, values=gy, ctype="Y_IFU", cunit="arcsec")
    save_regular_axis(header, axis=3, values=wave, ctype="WAVE", cunit="Angstrom")

    primary = fits.PrimaryHDU(data=cube, header=header)
    hdus = [primary]

    if wave is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(wave), name="WAVE"))
    if gx is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(gx), name="XGRID"))
    if gy is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(gy), name="YGRID"))

    fits.HDUList(hdus).writeto(outpath, overwrite=True)
    print(f"\nCubo guardado en: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Inspección RSS y reconstrucción simple de cubo")
    parser.add_argument("rss_fits", help="Ruta al archivo RSS FITS")
    parser.add_argument("--out", default="cube_simple.fits", help="Archivo FITS de salida")
    parser.add_argument(
        "--flux-hdu",
        default=None,
        help="Índice o EXTNAME de la HDU 2D a reconstruir (por ejemplo: 0, PRIMARY, 'Gas emission')",
    )
    parser.add_argument("--gridsize", type=float, default=0.5, help="Paso espacial de la grilla")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma espacial para pesos gaussianos")
    parser.add_argument("--no-reconstruct", action="store_true", help="Solo inspecciona, no reconstruye")
    args = parser.parse_args()

    path = Path(args.rss_fits)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with fits.open(path) as hdul:
        summarize_hdul(hdul)
        parts = infer_rss_parts(hdul, flux_selector=args.flux_hdu)
        print_rss_summary(parts)

        if args.no_reconstruct:
            return

        if parts["flux"] is None:
            raise RuntimeError("No se encontró una extensión 2D de FLUX/RSS/SCI")
        if parts["xpos"] is None or parts["ypos"] is None:
            raise RuntimeError("No se encontraron XPOS/YPOS; no se puede reconstruir ni aproximar el cubo")

        cube, gx, gy = reconstruct_cube_simple(
            flux=parts["flux"],
            xpos=parts["xpos"],
            ypos=parts["ypos"],
            gridsize=args.gridsize,
            sigma=args.sigma,
        )
        print(f"\nCUBE shape: {cube.shape}  -> (n_wave, ny, nx)")
        save_cube_fits(
            args.out,
            cube,
            wave=parts["wave"],
            gx=gx,
            gy=gy,
            source_header=parts["flux_header"],
            source_name=parts["flux_name"],
        )


if __name__ == "__main__":
    main()
