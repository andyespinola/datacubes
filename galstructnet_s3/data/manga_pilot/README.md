# Piloto MaNGA — 11019-12705 (mangaid 1-623253)

Espiral barrada de cara (Sersic n=1.21, b/a=0.94, z=0.0344), bundle de 127
fibras, calidad DRP perfecta (drp3qual=0). Elegida cruzando drpall con la
cobertura GZ3D: barra con 17.459 px y brazos con 46.364 px a >=3 votos (la
mejor de 40 candidatas evaluadas).

Archivos (SDSS DR17, publicos, descargados 2026-07-03):

| Archivo | Fuente | Contenido |
|---|---|---|
| manga-11019-12705-LOGCUBE.fits.gz | DRP v3_1_1 | FLUX/IVAR/MASK (4563,72,72) rejilla log10 dex 1e-4; WAVE; GIMG/RIMG/IIMG/ZIMG (imagenes banda ancha del propio cubo); GPSF/RPSF/IPSF/ZPSF (PSF reconstruida) |
| manga-11019-12705.Pipe3D.cube.fits.gz | pyPipe3D VAC v3_1_1 | SSP (21,72,72): v(13)+err(14), sigma(15)+err(16), age(5)+err(7), Z(8)+err(10), Sigma_masa(19)+err(20), Av(11)+err(12); SFH, ELINES, INDICES |
| gz3d_1-623253_127_14743455.fits.gz | GZ3D VAC v4 | mascaras crowdsourced en rejilla SDSS 525x525: ext3=brazos, ext4=barra (conteo de voluntarios 0-15); requiere reproyeccion a la rejilla MaNGA |
| drpall-v3_1_1.fits | DRP | catalogo maestro (metadatos de todas las galaxias) |

Pendiente de procesar por nosotros (ver conversacion/specs):
1. h3/h4 + errores: corrida pPXF propia sobre el LOGCUBE (o c=0).
2. snr_spec desde IVAR (ventana 5000-5500 A).
3. Empaquetado a dataset_entry HDF5 (script manga_to_entry, por escribir).
4. Reproyeccion GZ3D 525x525 (WCS SDSS) -> 72x72 (WCS MaNGA).
5. PSF: recorte centrado K x K impar del RPSF + normalizacion.
