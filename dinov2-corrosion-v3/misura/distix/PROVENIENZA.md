# Provenienza dei moduli DISTIX

- **Origine:** repo `https://github.com/MrDaSp/foto-misura` (DISTIX).
- **Commit:** `637138c8f7d8791bd0e306680e8c5146d8b34181` (2026-11-27).
- **Data della copia:** 2026-11-28.
- **Autori:** il codice è dell'utente e di un collega, quindi si riusa liberamente all'interno di Rustron.

## File copiati

| File qui | File in DISTIX |
| --- | --- |
| `camera_model.py` | `core/camera_model.py` |
| `triangulation.py` | `core/triangulation.py` |
| `bundle_adjustment.py` | `core/bundle_adjustment.py` |
| `coordinate_system.py` | `core/coordinate_system.py` |
| `uncertainty_engine.py` | `core/uncertainty_engine.py` |
| `metadata_reader.py` | `acquisition/metadata_reader.py` |
| `calibration_manager.py` | `acquisition/calibration_manager.py` |
| `rtk_validator.py` | `acquisition/rtk_validator.py` |
| `ppk_importer.py` | `acquisition/ppk_importer.py` |

`ppk_importer.py` non era nell'elenco del piano. È stato aggiunto perché `metadata_reader.py` lo importa: usa due funzioni per le date e i nomi file DJI.

## Modifiche

L'unica modifica è negli import:
- `from core.X import …` è diventato `from .X import …`;
- `from acquisition.X import …` è diventato `from .X import …`;
- gli import relativi già presenti sono rimasti com'erano.

La logica non è stata toccata. Si può verificare così: applicando la stessa sostituzione degli import ai file originali, il `diff` con questi file risulta vuoto.

## Cosa non è stato portato

Non sono stati portati:
- `gui/` e `reporting/`;
- `core/ruler_engine.py` (solo riferimento), `core/pole_geometry.py`, `core/conductor_geometry.py`, `core/building_geometry.py`, `core/distance_engine.py`;
- `config/documented_calibrations.json`.

Per quest'ultimo, `calibration_manager.load_documented_profiles()` cerca il file in `misura/config/`. Se non c'è, restituisce una lista vuota. In Rustron i profili di calibrazione stanno in `profili_drone/`.

`pyproj` è facoltativo (import già dentro `try`) e non è installato: le conversioni WGS84 → ENU locale passano per ECEF e non ne hanno bisogno.
