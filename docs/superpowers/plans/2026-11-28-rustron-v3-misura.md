# Rustron v3 – Misura in m e cm: piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** creare `dinov2-corrosion-v3`, una copia della v2 con una funzione secondaria «📐 Misura». La funzione dà aree e lunghezze in m e cm in due modi:
- da una foto sola (STIMA);
- da ≥ 2 foto con tie point, come in DISTIX (VALIDATA / NON VALIDATA).

Le misure si possono inserire nel PDF.

**Architecture:**
- Il calcolo vive in un pacchetto Python puro, `dinov2-corrosion-v3/misura/`. Contiene:
  - i moduli DISTIX copiati in `misura/distix/`;
  - la lettura del drone;
  - la stima da una foto;
  - la sessione con più foto;
  - le misure.
- Ogni modo di misurare produce un **proiettore**, cioè una funzione da pixel della foto a punti 3D su un piano. Le misure (aree di classe, anomalie, linee, poligoni) si calcolano una volta sola sopra il proiettore. L'incertezza U95 si ottiene con Monte Carlo (200 campioni, U95 = 2σ, approccio GUM Supplemento 1).
- `main.py` tiene in memoria le ultime 5 analisi, indicizzate da `result_id`, ed espone endpoint `/api/misura/*`. La UI aggiunge il pannello Misura in «Anteprima ingrandita» e una finestra per la sessione con più foto.

**Tech Stack:** Python 3.11, FastAPI, numpy, **scipy** (nuova dipendenza, serve al bundle adjustment DISTIX), Pillow, reportlab, unittest. Nel frontend, JS vanilla dentro `static/index.html` (IIFE).

**Spec:** `docs/superpowers/specs/2026-11-27-rustron-v3-misura-design.md`. Bozza grafica: `.superpowers/brainstorm/misura-v3-bozza-1.html`.

## Global Constraints

- La poc (`dinov2-corrosion-poc`, porta 8765) e la v2 (`dinov2-corrosion-v2`, porta 8766) **non si modificano**. Tutto il lavoro avviene in `dinov2-corrosion-v3`, porta **8767**.
- Non si riavvia mai il server dell'utente. I test con server reale usano la porta **8799**.
- Commit e push solo quando l'utente li chiede. Il passo «Commit» di ogni task significa «prepara il commit e chiedi»: se l'utente non ha autorizzato, lascia le modifiche non committate e prosegui.
- Con «Misura» spento la v3 si comporta come la v2: nessun calcolo di misura, PDF identico.
- Coordinate e date arrivano solo dai metadati della foto.
- Il tool resta generalista: nessun nome di classe YOLO o di drone scritto nel codice delle misure. I droni stanno nei file JSON di `profili_drone/`.
- La qualità della maschera non cambia: le misure leggono la maschera classi così com'è.
- Etichette, sempre presenti su ogni numero:
  - `STIMA` per la foto singola e per ogni superficie reale;
  - `VALIDATA` / `NON VALIDATA` per la sessione con più foto.
- Ogni misura riporta valore ± U95 (k = 2).
- Testi della UI e del PDF in italiano. I messaggi d'errore sono frasi semplici che dicono cosa fare.
- Numeri fissati dallo spec:
  - soglia zoom 5 %;
  - suolo ±3 m (σ 1,5 m);
  - σ clic 1,5 px;
  - convergenza ≥ 15°;
  - almeno 4 tie point, ognuno in almeno 2 foto;
  - un LRF che vale 0 o è fuori portata conta come assente.
- Fattore di sviluppo predefinito **3,0 ± 1,0 (U95)** *(scelta del piano)*. Motivazione da riportare nel codice: un angolare a L di ala a sviluppa circa 4a di acciaio per metro e ne mostra in foto circa 1,2–1,4a, quindi ≈ 3.
- Origine del codice DISTIX: repo `https://github.com/MrDaSp/foto-misura`, commit `637138c8f7d8791bd0e306680e8c5146d8b34181`.
- I test sono `unittest` nella cartella `tests/` della v3, come nella v2. Si lanciano con `.venv\Scripts\python.exe -m pytest` (pytest va installato nel task 1).

## Review Focus

1. **Foto ridotta o ruotata.** La v2 accetta `meta_source`, cioè l'originale per i metadati mentre si analizza una copia ridotta, e applica `exif_transpose`.
   - Gli intrinseci vanno scalati sulla larghezza dell'immagine **analizzata**, non su quella nativa.
   - Se l'Orientation EXIF è diversa da 1, la foto è «non misurabile».
   - Test nel task 3.
2. **Metadati assenti o tolti** (foto non DJI, gimbal mancante, nessun GPS). Il risultato deve essere «non misurabile» con il motivo, senza eccezioni né numeri inventati. Test nei task 3 e 4.
3. **LRF a 0,000 m o fuori dal sostegno** (come M3T Camera_4). Va ignorato e si usa la geometria. Test nel task 4.
4. **Base del sostegno non stimabile.** Succede quando il riquadro tocca il bordo basso, il raggio va sopra l'orizzonte o non c'è un riquadro. Il risultato è «nessuna stima» con un suggerimento, e mai un numero enorme o negativo. Test nel task 4.
5. **Sessione sporca:**
   - tie point segnato in una sola foto;
   - stessa foto caricata due volte;
   - foto extra di un altro sostegno o in RTK FLOAT (`RtkFlag` 16);
   - foto extra con zoom.

   Ogni caso dà un messaggio chiaro e l'esito corretto (incompleta o NON VALIDATA). Test nel task 6.

---

### Task 1: Cartella v3 avviabile (copia della v2)

**Files:**
- Create: `dinov2-corrosion-v3/` (copia di `dinov2-corrosion-v2/`)
- Create: `AVVIA RUSTRON V3.bat` (nella radice del repo)
- Modify: `dinov2-corrosion-v3/AVVIA_V2.bat` → rinominato `AVVIA_V3.bat`, con porta 8767
- Modify: `dinov2-corrosion-v3/.venv/pyvenv.cfg` (`home`)
- Modify: `dinov2-corrosion-v3/requirements.txt`, `static/index.html` (titolo)
- Modify: `.gitignore` (radice)
- Create: `dinov2-corrosion-v3/LEGGIMI_V3.md`
- Test: `dinov2-corrosion-v3/tests/test_v3_avvio.py`

**Interfaces:**
- Produces: la cartella v3 con `.venv\Scripts\python.exe`, che ha scipy e pytest; `ROOT` di `main.py` punta a v3.

- [ ] **Step 1: Copia.** Usa `robocopy "dinov2-corrosion-v2" "dinov2-corrosion-v3" /E /XD __pycache__ /XF server.log server-error.log`. Copia anche `.venv`, `.python`, `model_cache` (~2,7 GB), così la v3 è autonoma.
  - Copia `impostazioni/` ma non `model_cache/fine_masks` (`/XD fine_masks`): è solo una cache e si rigenera.
  - Atteso: exit code di robocopy < 8.
- [ ] **Step 2: Rendi il venv indipendente dalla v2.** In `dinov2-corrosion-v3/.venv/pyvenv.cfg` sostituisci in `home` il percorso `dinov2-corrosion-v2\.python\…` con `dinov2-corrosion-v3\.python\…`.
  - Verifica: `dinov2-corrosion-v3\.venv\Scripts\python.exe -c "import sys;print(sys.base_prefix)"` deve stampare un percorso sotto `dinov2-corrosion-v3`.
- [ ] **Step 3: Dipendenze.** Aggiungi a `requirements.txt` le righe `scipy>=1.11,<2` e `pytest>=8,<9`, poi installa con `uv pip install --python dinov2-corrosion-v3\.venv\Scripts\python.exe scipy pytest`. Se `uv` non è nel PATH, usa `python -m pip`.
  - Verifica: `import scipy` funziona.
- [ ] **Step 4: Avvio sulla porta 8767.**
  - In `AVVIA_V3.bat` sostituisci ogni `8766` con `8767` e i testi «v2» con «v3».
  - Crea `AVVIA RUSTRON V3.bat` nella radice, come `AVVIA RUSTRON.bat` ma con `call "%~dp0dinov2-corrosion-v3\AVVIA_V3.bat"`.
  - Nel `<title>` e nell'intestazione di `static/index.html` aggiungi « v3».
  - In `.gitignore` duplica le due righe `dinov2-corrosion-v2/...` con `dinov2-corrosion-v3/...` e aggiungi `dinov2-corrosion-v3/misure/`.
- [ ] **Step 5: Test che fallisce.** Crea `tests/test_v3_avvio.py`:

```python
class AvvioV3Test(unittest.TestCase):
    def test_root_e_porta(self):
        import main
        self.assertEqual(main.ROOT.name, "dinov2-corrosion-v3")
        bat = (main.ROOT / "AVVIA_V3.bat").read_text(encoding="utf-8", errors="ignore")
        self.assertIn("8767", bat); self.assertNotIn("8766", bat)
    def test_status(self):
        from fastapi.testclient import TestClient; import main
        self.assertEqual(TestClient(main.app).get("/api/status").status_code, 200)
```

- [ ] **Step 6: Esegui tutti i test della v3.** Comando: `cd dinov2-corrosion-v3 && .venv\Scripts\python.exe -m pytest tests -q`. Atteso: tutto verde, compresi i test ereditati dalla v2.
- [ ] **Step 7: Prova reale.** Avvia `.venv\Scripts\python.exe -m uvicorn main:app --port 8799` in background, chiama `GET http://127.0.0.1:8799/api/status` e controlla 200, poi ferma **solo quel** processo. La v2 sulla 8766 non va toccata.
- [ ] **Step 8: LEGGIMI_V3.md.** Scrivi in breve: cos'è la v3, come avviarla, porta 8767, poc e v2 invariate. Il resto si aggiunge nel task 10.
- [ ] **Step 9: Commit.** Messaggio: `Rustron v3: copy of v2 on port 8767`. Non includere `.venv`, `.python` e `model_cache`, già ignorati o da ignorare: controlla con `git status` che non compaiano.

---

### Task 2: Moduli DISTIX copiati in `misura/distix/`

**Files:**
- Create: `dinov2-corrosion-v3/misura/__init__.py`
- Create: `dinov2-corrosion-v3/misura/distix/__init__.py`
- Create, copiati da DISTIX senza modifiche di logica:
  - `misura/distix/camera_model.py`
  - `misura/distix/triangulation.py`
  - `misura/distix/bundle_adjustment.py`
  - `misura/distix/coordinate_system.py`
  - `misura/distix/uncertainty_engine.py`
  - `misura/distix/metadata_reader.py`
  - `misura/distix/calibration_manager.py`
  - `misura/distix/rtk_validator.py`
- Create: `misura/distix/PROVENIENZA.md`
- Test: `tests/test_distix_import.py`

**Interfaces:**
- Produces, importabili come `from misura.distix.<modulo> import …`:
  - `CameraModel(name, pos, front, up, image_width=, image_height=, fx_pix=, fy_pix=, cx_pix=, cy_pix=, calibration=, rtk_fix=, rtk_flag=, rtk_std=, rtk_std_hgt=, gnss_quality=, metadata=)`, con `.pixel_to_ray(u,v)->(C,d)`, `.project_point(X)`, `.rotation_matrix`, `.has_intrinsics`, `.clone()`;
  - `triangulate_from_cameras(observations, min_convergence_deg=15.0, …) -> TriangulationResult` (`.point_3d`, `.covariance_3d`, `.is_reliable`, `.all_in_front`, `.max_convergence_angle_deg`, `.max_reproj_px`, `.notes`);
  - `compute_epipolar_curve_points(cam_src, uv_src, cam_dst) -> list[(u,v)]`;
  - `TiePointObservation(point_id, camera_name, u_pix, v_pix)`;
  - `run_constrained_bundle_adjustment_with_diagnostics(cameras: dict, observations: list) -> BundleAdjustmentResult` (ruota le camere sul posto);
  - `dji_euler_to_vectors(pitch, yaw, roll) -> (front, up)` in ENU;
  - `parse_dji_xmp(path)`, `parse_image_exif(path)`;
  - `CoordinateManager.set_local_enu_origin(lon, lat, alt)` / `.wgs84_to_local(lon, lat, alt)`;
  - `evaluate_dji_xmp_gnss(xmp, exif_gps) -> GNSSQuality` (`.is_validated`, `.max_horiz_std_m`, `.std_hgt_m`);
  - `parse_dewarp_data(raw, w, h)`.

- [ ] **Step 1: Test che fallisce** (`tests/test_distix_import.py`). Costruisci due `CameraModel` sintetiche:
  - 4000×3000, fx = fy = 2800, cx = 2000, cy = 1500;
  - posizioni (0,0,0) e (10,0,0), che guardano verso +Y (`front=[0,1,0]`, `up=[0,0,1]`).

  Proietta il punto (5, 30, 2) con `project_point` in entrambe e triangola con `triangulate_from_cameras`. Asserzioni: `np.allclose(res.point_3d, [5,30,2], atol=1e-6)` e `res.all_in_front`.
- [ ] **Step 2: Esegui.** Comando: `.venv\Scripts\python.exe -m pytest tests/test_distix_import.py -q`. Atteso: FAIL con ModuleNotFoundError `misura`.
- [ ] **Step 3: Copia i file** dal clone DISTIX (o dal repo GitHub al commit indicato). Trasforma gli import `from core.x`, `from acquisition.x` e `from .x` in import relativi dentro `misura.distix`. Non portare `gui/`, `reporting/`, `ruler_engine`, `pole_geometry`, `conductor_geometry`, `building_geometry`, `distance_engine`. Se un modulo copiato importa uno di questi, togli solo quell'import e annotalo in PROVENIENZA.md. `pyproj` è facoltativo (import già in try) e non va installato.
- [ ] **Step 4: Scrivi PROVENIENZA.md** con: URL del repo, commit, data, elenco dei file e di ogni modifica fatta (solo import). Specifica che il codice è dell'utente e di un collega.
- [ ] **Step 5: Esegui.** Il test deve passare.
- [ ] **Step 6: Commit.** Messaggio: `misura: vendor DISTIX geometry modules`.

---

### Task 3: Riconoscimento drone e profili

**Files:**
- Create: `dinov2-corrosion-v3/misura/drone.py`
- Create: `dinov2-corrosion-v3/profili_drone/dji_m3t.json`, `dji_h30t.json`, `LEGGIMI.md`
- Test: `tests/test_drone.py`

**Interfaces:**
- Consumes: `parse_dji_xmp`, `parse_image_exif`, `evaluate_dji_xmp_gnss`, `parse_dewarp_data` (task 2).
- Produces:
  - `@dataclass FotoMeta`: `make, model, image_source, width, height, orientation, focal_mm, focal35_mm, digital_zoom, lat, lon, abs_alt, rel_alt, gimbal_pitch, gimbal_yaw, gimbal_roll, rtk_flag, rtk_std_h, rtk_std_v, lrf_status, lrf_m, dewarp, xmp: dict`. I campi mancanti valgono `None`.
  - `leggi_foto(meta_bytes: bytes, analysed_size: tuple[int,int]) -> FotoMeta`. `width` e `height` sono quelli **analizzati**.
  - `@dataclass Profilo`: `id, nome, make, model, focal_mm, sensor_w_mm, native_w, native_h, has_rtk, has_lrf, lrf_range_m: (min,max), sigma_gimbal_deg, sigma_focal_rel, calibrazione: dict|None, generico: bool`.
  - `scegli_profilo(meta: FotoMeta, cartella: Path = ROOT/"profili_drone") -> Profilo`. Si sceglie per make + model (+ `image_source`, se presente nel profilo). Senza corrispondenza si restituisce un profilo generico: `generico=True`, `sigma_focal_rel=0.05`, `sigma_gimbal_deg=2.0`.
  - `intrinseci(meta, profilo) -> dict(fx, fy, cx, cy, fonte)`. Fonti in ordine:
    1. `"calibrazione"` (scalata a `width/native_w`);
    2. `"dewarp"`;
    3. `"35mm"`, con `fx = focal35_mm * hypot(w,h) / 43.267`;
    4. `"sensore"`, con `fx = focal_mm / sensor_w_mm * w`.
  - `misurabile(meta, profilo) -> tuple[bool, str]`.

- [ ] **Step 1: Test che falliscono** (`tests/test_drone.py`). Le foto reali stanno in `C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione`; se la cartella manca, salta con `skipUnless`.
  - `test_m3t_camera4`: `make=="DJI"`, `model=="M3T"`, `focal_mm==4.4`, `rtk_flag==50`, `rel_alt≈35.174`, `gimbal_pitch≈-46.1`, `lrf_m==0.0`. Il profilo ha id `dji_m3t`, non è generico; `misurabile` dà `True`.
  - `test_h30t_camera6`: `model=="H30T"`, `image_source=="WideCamera"`, `lrf_m≈30.234`, profilo `dji_h30t`.
  - `test_intrinseci_35mm_e_ridimensionata`: Camera_4 letta con `analysed_size=(8000,6000)` dà `fx≈24*10000/43.267` (±0,5 %). Con `analysed_size=(4000,3000)` dà `fx` dimezzato. In entrambi i casi `fonte=="35mm"`.
  - `test_zoom_non_misurabile`: `FotoMeta` di Camera_4 con `focal_mm=4.4*1.2`, oppure con `digital_zoom=2.0` → `(False, msg)` e `"zoom" in msg.lower()`.
  - `test_termica_e_orientamento`: `image_source="InfraredCamera"` → False con «termica»; `orientation=6` → False con «ruotata».
  - `test_metadati_mancanti`: un JPEG generato con PIL, senza EXIF → `scegli_profilo(...).generico` è True e `misurabile` → `(False, msg)` con «metadati» nel messaggio. Nessuna eccezione.
  - `test_drone_sconosciuto`: un `FotoMeta` completo con `make="Acme"`, `model="X1"` → profilo generico e `misurabile` True.
- [ ] **Step 2: Esegui.** Atteso: FAIL perché il modulo non esiste.
- [ ] **Step 3: Profili JSON.**
  - `dji_m3t.json`: `make "DJI"`, `model "M3T"`, `image_source "WideCamera"`, `focal_mm 4.4`, `sensor_w_mm 6.4`, `native_w 8000`, `native_h 6000`, `has_rtk true`, `has_lrf true`, `lrf_range_m [3,1200]`, `sigma_gimbal_deg 1.0`, `sigma_focal_rel 0.02`, `calibrazione null`.
  - `dji_h30t.json`: `model "H30T"`, `focal_mm 6.72`, `native_w 8064`, `native_h 6048`, `sensor_w_mm 9.6`, gli altri campi come M3T.
  - `LEGGIMI.md` spiega in italiano come aggiungere un drone copiando un file.
- [ ] **Step 4: Implementa `misura/drone.py`.** I metadati vengono dai byte di `meta_source`, se c'è, altrimenti dalla foto caricata. Scrivi i byte in un file temporaneo per le funzioni DISTIX, che lavorano su percorsi.

  Regole di `misurabile`, in quest'ordine e con il messaggio indicato:

  | Condizione | Messaggio |
  | --- | --- |
  | gimbal, `rel_alt` o lat/lon mancanti | «metadati di posizione/gimbal mancanti» |
  | `orientation` non 1 | «foto ruotata: usare l'originale del drone» |
  | `image_source` contiene «Infrared» o «Thermal» | «foto termica» |
  | `digital_zoom` > 1,01 oppure `focal_mm` fuori ±5 % dal profilo (se il profilo non è generico) | «zoom attivo: si misura solo con la camera grandangolare senza zoom» |

- [ ] **Step 5: Esegui.** Atteso: PASS.
- [ ] **Step 6: Commit.** Messaggio: `misura: drone recognition and profiles`.

---

### Task 4: Stima da una foto (proiettore su piano verticale)

**Files:**
- Create: `dinov2-corrosion-v3/misura/geometria.py`
- Create: `dinov2-corrosion-v3/misura/stima_foto.py`
- Test: `tests/test_stima_foto.py`

**Interfaces:**
- Consumes: `FotoMeta`, `Profilo`, `intrinseci`, `misurabile` (task 3); `CameraModel`, `dji_euler_to_vectors` (task 2).
- Produces:
  - `geometria.raggi(cam: CameraModel, u: np.ndarray, v: np.ndarray) -> np.ndarray[...,3]`: direzioni unitarie vettorizzate, pinhole senza distorsione.
  - `geometria.interseca_piano(C, dirs, p0, n) -> np.ndarray[...,3]`: punti 3D; NaN dove il raggio è parallelo al piano o il punto sta dietro la camera.
  - `@dataclass Piano(p0: np.ndarray, n: np.ndarray)`.
  - `@dataclass Proiettore`:
    - `cam: CameraModel`, `piano: Piano`, `etichetta: str` ("STIMA" / "VALIDATA" / "NON VALIDATA"), `campioni: list[tuple[CameraModel, Piano]]` (200 varianti Monte Carlo), `info: dict` (riga del pannello);
    - `punti(u, v) -> np.ndarray` (proiezione con cam e piano nominali);
    - `punti_campione(k, u, v)`.
  - `stima_foto.costruisci(meta, profilo, supporto_box: list[float]|None, base_px: tuple|None, seed=0) -> Proiettore | NessunaStima`.
  - `@dataclass NessunaStima(motivo: str, suggerimento: str)`.
  - `info` contiene: `drone`, `rtk` (es. «RTK FIX · ±1 cm»), `distanza_m`, `fonte_distanza` ("laser" / "geometria"), `scala_mm_px_centro`, `profilo_generico`.

**Algoritmo** (la firma da sola non lo determina):

```text
camera: pos (0,0,0) in ENU locale; front, up = dji_euler_to_vectors(pitch, yaw, roll); intrinseci dal task 3.
base B:
  1. LRF valido = lrf_m non None, min ≤ lrf_m ≤ max del profilo, centro immagine dentro supporto_box
     → P = C + lrf_m · raggio(centro); B = proiezione verticale di P alla quota del piede del box
       (intersezione del raggio del piede con il piano verticale per P ortogonale alla vista orizzontale).
  2. altrimenti piede = base_px oppure (centro-x del box, y1 del box); se y1 ≥ 0.99·h, o niente box
     e niente base_px → NessunaStima("base del sostegno non visibile", "apri la sessione multi-foto o clicca la base").
     Il raggio del piede deve scendere (d_z < -0.02), altrimenti NessunaStima.
     B = intersezione con il piano z = -rel_alt.
piano: n = componente orizzontale di (C - B) normalizzata, p0 = B (piano verticale rivolto alla camera).
Monte Carlo (200, seed): quota suolo ~ N(0, 1.5 m) (non usata se LRF); pitch/yaw/roll ~ N(0, sigma_gimbal_deg);
  fx, fy ~ ×N(1, sigma_focal_rel); normale ruotata di N(0, 10°) intorno alla verticale e inclinata di N(0, 5°).
etichetta = "STIMA".
```

- [ ] **Step 1: Test che falliscono** (`tests/test_stima_foto.py`).
  - `test_raggi_coincidono_con_distix`: per 20 pixel casuali, `geometria.raggi` coincide con la direzione di `cam.pixel_to_ray` a meno di 1e-9, su una camera senza distorsione.
  - `test_frontale_area_pixel`:
    - camera sintetica fx = 2000, front orizzontale, piano verticale a 10 m;
    - l'area di un quadrato di 100×100 px al centro è 0,25 m² ± 0,5 %.
  - `test_camera4_scala_varia`:
    - Camera_4 reale con `supporto_box` = il riquadro della ROI automatica dell'utente (bordi in pixel 8000×6000 da `object_detection` su Camera_4; se non è disponibile, usa `[3000,300,5000,5700]`);
    - il risultato è un `Proiettore` con `fonte_distanza=="geometria"`, perché l'LRF vale 0;
    - la scala in mm/px nel pixel più basso del box è > 2× quella nel pixel più alto;
    - la `distanza_m` orizzontale sta fra 5 e 40.
  - `test_h30t_usa_laser`: Camera_6 con un box che contiene il centro immagine → `fonte_distanza=="laser"`.
  - `test_base_non_visibile`: box con `y1 = h` → `NessunaStima` con «base» nel motivo.
  - `test_raggio_sopra_orizzonte`: camera con pitch +5° e piede sopra il centro → `NessunaStima`. Nessun NaN e nessun numero negativo.
  - `test_metadati_mancanti`: se `misurabile` è falso → `NessunaStima` con lo stesso motivo.
- [ ] **Step 2: Esegui.** Atteso: FAIL.
- [ ] **Step 3: Implementa** `geometria.py` e `stima_foto.py` seguendo l'algoritmo sopra.
- [ ] **Step 4: Esegui.** Atteso: PASS.
- [ ] **Step 5: Commit.** Messaggio: `misura: single-photo estimate on vertical plane`.

---

### Task 5: Calcolo delle misure sopra un proiettore

**Files:**
- Create: `dinov2-corrosion-v3/misura/misure.py`
- Test: `tests/test_misure.py`

**Interfaces:**
- Consumes: `Proiettore` (task 4).
- Produces:
  - `@dataclass Misura`:
    - campi: `id: str`, `tipo: str` ("classe" | "anomalia" | "linea" | "area"), `nome: str`, `valore: float`, `u95: float`, `unita: str` ("m²" | "m"), `etichetta: str`, `extra: dict`;
    - le anomalie hanno `extra={"lunghezza_m","larghezza_m","u95_lunghezza_m","u95_larghezza_m"}`;
    - le classi hanno `extra={"reale_m2","u95_reale_m2","fattore"}`;
    - metodo `to_dict()`.
  - `misure_automatiche(proiettore, class_mask: np.ndarray, roi_box: list[float], clusters: list[dict]|None, fattore: float = 3.0, u95_fattore: float = 1.0) -> list[Misura]`. Il pixel (i, j) della griglia della maschera corrisponde al pixel originale `u = x0 + (i+0.5)·(x1-x0)/w`, `v = y0 + (j+0.5)·(y1-y0)/h`. Le classi sono 2 «Ruggine media», 3 «Ruggine intensa», 4 «Zone grigie». I `bbox_px` delle anomalie sono in coordinate di griglia.
  - `misura_linea(proiettore, punti_px: list[tuple], nome) -> Misura` (polilinea; lunghezza 3D sul piano).
  - `misura_area(proiettore, punti_px: list[tuple], nome) -> Misura` (poligono chiuso; area dei punti 3D sul piano).
  - Note di implementazione:
    - l'area di una cella si calcola dai 4 angoli 3D come `0.5·|d1 × d2|`, vettorizzato sugli angoli `(w+1)×(h+1)`;
    - l'U95 è `2·std` sui 200 campioni del proiettore;
    - l'incertezza dei clic (σ 1,5 px) si aggiunge a linee e aree perturbando i vertici dentro lo stesso Monte Carlo;
    - la superficie reale è `valore·fattore` e la sua U95 combina in quadratura l'incertezza relativa dell'area e `u95_fattore/fattore`;
    - l'etichetta della superficie reale è sempre "STIMA".

- [ ] **Step 1: Test che falliscono** (`tests/test_misure.py`). Usa il proiettore frontale sintetico del task 4: fx = 2000, piano a 10 m, quindi 5 mm/px.
  - `test_area_classe`: maschera 100×100 con 400 pixel di classe 3; `roi_box` mappa 1 cella = 1 px → «Ruggine intensa» vale `400·0.005²` m² ± 1 %; `extra["reale_m2"] == 3×valore`; `etichetta=="STIMA"`.
  - `test_linea`: linea orizzontale di 200 px → 1,00 m ± 1 %; `u95 > 0`.
  - `test_poligono`: quadrato di 200×200 px → 1,00 m² ± 1 %.
  - `test_anomalia`: un cluster con `bbox_px [10,20,30,60]` → lunghezza 0,20 m e larghezza 0,10 m (lato lungo = lunghezza).
  - `test_nessun_pixel`: maschera senza classe 2 → «Ruggine media» = 0 con `u95 = 0`, senza divisioni per zero.
- [ ] **Step 2: Esegui.** Atteso: FAIL.
- [ ] **Step 3: Implementa** `misura/misure.py`.
- [ ] **Step 4: Esegui.** Atteso: PASS.
- [ ] **Step 5: Commit.** Messaggio: `misura: areas, anomalies, lines, polygons with U95`.

---

### Task 6: Sessione multi-foto (tie point, BA, piano, validazione, salvataggio)

**Files:**
- Create: `dinov2-corrosion-v3/misura/sessione.py`
- Test: `tests/test_sessione.py`

**Interfaces:**
- Consumes: task 2 (`CameraModel`, `CoordinateManager`, `TiePointObservation`, `run_constrained_bundle_adjustment_with_diagnostics`, `triangulate_from_cameras`, `compute_epipolar_curve_points`, `evaluate_dji_xmp_gnss`), task 3 (`leggi_foto`, `scegli_profilo`, `intrinseci`, `misurabile`), task 4 (`Piano`, `Proiettore`).
- Produces:
  - `CARTELLA = ROOT/"misure"`, che contiene `<sha_principale>.json` e `foto/<sha>.jpg`.
  - `@dataclass Sessione`:
    - campi: `sha_principale: str`, `foto: list[dict(sha, nome)]`, la prima è la principale; `tie: dict[str, dict[str, (u,v)]]`, cioè punto → sha foto → pixel;
    - `salva()`, `@classmethod carica(sha) -> Sessione|None`;
    - `aggiungi_foto(bytes, nome) -> dict(sha, nome, misurabile, motivo, drone, rtk)`.
  - `camere(sessione) -> dict[sha, CameraModel]`:
    - ENU locale con origine nella foto principale, posizione da lat/lon e **AbsoluteAltitude**;
    - intrinseci dal task 3;
    - `rtk_fix = rtk_flag == 50`;
    - `gnss_quality` da `evaluate_dji_xmp_gnss`;
    - `calibration` = la calibrazione del profilo, o `None`.
  - `calcola(sessione) -> EsitoSessione`.
  - `@dataclass EsitoSessione`: `completa: bool`, `etichetta: str` ("VALIDATA" / "NON VALIDATA" / "INCOMPLETA"), `controlli: list[dict(id, testo, ok: bool)]`, `tie_info: dict[pid, dict(n_foto, reproj_px)]`, `proiettore: Proiettore|None`, `ba: dict`, `messaggi: list[str]`.
  - `suggerisci(sessione, pid, sha_dst) -> list[(u,v)]` (curva epipolare dalla prima osservazione di `pid`).

**Algoritmo di `calcola`:**

```text
1. Foto non misurabili (task 3) escluse, con messaggio. Foto con sha duplicato rifiutate già in aggiungi_foto.
2. tie point validi = osservati in ≥2 foto misurabili; se < 4 → INCOMPLETA con messaggio che nomina i punti
   mancanti (es. "T4 manca in una seconda foto").
3. BA: run_constrained_bundle_adjustment_with_diagnostics(camere, osservazioni) sul posto.
4. triangola ogni tie point (triangulate_from_cameras, default DISTIX).
5. piano = minimi quadrati (SVD) sui punti 3D; normale orientata verso la camera principale.
6. controlli (testi esatti):
   "RTK FIX su tutte le foto"            → tutte gnss_quality.is_validated
   "Convergenza ≥ 15°"                    → min sui tie point di max_convergence_angle_deg ≥ 15
   "Riproiezione entro soglia DISTIX"     → tutti TriangulationResult.is_reliable e all_in_front
   "Almeno 4 tie point in ≥ 2 foto"       → sempre vero qui (altrimenti INCOMPLETA)
   "Camera calibrata"                     → tutte le camere con calibration non None
   tutti ok → VALIDATA, altrimenti NON VALIDATA; messaggi in parole semplici per ogni controllo fallito
   (convergenza: "le foto sono troppo allineate: scattane una più di lato").
7. Proiettore: cam = camera principale dopo BA, piano dal punto 5, etichetta = esito;
   200 campioni Monte Carlo: tie point ~ N(punto, covariance_3d) → piano rifittato; assetto camera principale
   ~ N(0, post_ba_attitude_std_deg).
```

- [ ] **Step 1: Test che falliscono** (`tests/test_sessione.py`). Scena sintetica costruita con `CameraModel` reali: 3 camere a 15 m da un piano verticale, yaw −20°, 0°, +25°, RTK FIX, con calibrazione; 6 tie point sul piano, proiettati con `project_point` più rumore N(0, 0,5 px), seed fisso. Una funzione di test fornisce queste camere al posto di `camere()` (iniettabile come parametro `camere_fn` di `calcola`).
  - `test_validata`: `etichetta=="VALIDATA"`; la normale del piano dista < 2° da quella vera; `misura_area` (task 5) di un quadrato noto di 1 m² proiettato nella principale dà 1 ± 0,05 m².
  - `test_senza_calibrazione`: stesse camere con `calibration=None` → "NON VALIDATA" con il controllo «Camera calibrata» a `ok=False`; il proiettore c'è comunque.
  - `test_rtk_float`: una camera con `rtk_flag=16` → "NON VALIDATA" e controllo RTK a False.
  - `test_pochi_punti`: 3 tie point, oppure T4 in una sola foto → "INCOMPLETA" e un messaggio che contiene "T4".
  - `test_foto_allineate`: yaw 0°, 3°, 6° → controllo di convergenza a False e messaggio «più di lato».
  - `test_salva_carica`: `salva()` e poi `carica(sha)` restituiscono gli stessi `tie` e `foto`, scritti in una cartella temporanea (parametro `cartella`).
  - `test_foto_duplicata`: chiamare due volte `aggiungi_foto` con gli stessi byte lascia una sola foto e restituisce `motivo` con «già presente».
- [ ] **Step 2: Esegui.** Atteso: FAIL.
- [ ] **Step 3: Implementa** `misura/sessione.py`.
- [ ] **Step 4: Esegui.** Atteso: PASS. Se `test_validata` fallisce per colpa del BA su dati sintetici perfetti, verifica prima con il BA disattivato per isolare il problema. Non allentare le soglie DISTIX.
- [ ] **Step 5: Commit.** Messaggio: `misura: multi-photo tie-point session with DISTIX BA and validation`.

---

### Task 7: Endpoint API e cache delle analisi

**Files:**
- Modify: `dinov2-corrosion-v3/main.py` (fine di `classify`, `analyze`, nuovi endpoint dopo `/api/presets`)
- Create: `dinov2-corrosion-v3/misura/api.py` (un `APIRouter` con prefisso `/api/misura`, incluso in `main.app`)
- Test: `tests/test_api_misura.py`

**Interfaces:**
- Consumes: task 3–6; `report_pdf.build_pdf(..., misure=None)` (task 8).
- Produces:
  - `main.ANALISI: OrderedDict[result_id, dict]`, con le ultime 5 analisi. Ogni voce contiene: `crop`, `blend` (PIL), `pred` (np.uint8), `report`, `clusters`, `map_info`, `photo_name`, `meta_bytes`, `analysed_size`, `sha`. `classify` ce le mette senza cambiare la risposta.
  - `POST /api/misura/stima` (JSON `{result_id, supporto_box?, base_px?, linee:[{nome,punti}], aree:[{nome,punti}], fattore?}`). Risponde:
    - `{info, etichetta, misure:[Misura.to_dict()]}`;
    - oppure `{nessuna_stima:{motivo,suggerimento}}`;
    - 404 se `result_id` non è più in memoria, con il messaggio «Analisi non più in memoria: ripeti l'analisi».
  - `POST /api/misura/sessione/apri` (`{result_id}`): crea o carica la sessione della foto principale e salva la foto in `misure/foto/`. Risponde `{sessione}`.
  - `POST /api/misura/sessione/foto` (multipart `sha_principale`, `file`) → l'esito di `aggiungi_foto`.
  - `GET /api/misura/sessione/foto/{sha}` → il JPEG, per la UI.
  - `POST /api/misura/sessione/punto` (`{sha_principale, pid, sha_foto, u, v}`; con `u=null` cancella il punto).
  - `POST /api/misura/sessione/suggerisci` (`{sha_principale, pid, sha_foto}`) → `{curva:[[u,v],…]}`.
  - `POST /api/misura/sessione/calcola` (`{sha_principale, result_id, linee, aree, fattore?}`) → `{etichetta, controlli, tie_info, messaggi, misure}`.
  - `POST /api/misura/pdf` (`{result_id, misure:[Misura dict selezionate], nota_fattore}`) → `{report_pdf_base64}`, rigenerato con `build_pdf(..., misure=...)`.
  - Tutti i calcoli pesanti girano in `asyncio.to_thread`.

- [ ] **Step 1: Test che falliscono** (`tests/test_api_misura.py`, con `TestClient`). Dove serve un'analisi vera, inserisci a mano una voce sintetica in `main.ANALISI`: crop e blend PIL 100×100, `pred` con classi, `meta_bytes` di Camera_4 (skip se manca la cartella foto), `analysed_size` (8000,6000) e `roi_original_pixels`.
  - `test_stima_ok`: 200; `etichetta=="STIMA"`; tra le misure c'è «Ruggine intensa».
  - `test_stima_result_scaduto`: un `result_id` inesistente → 404 con «ripeti l'analisi».
  - `test_nessuna_stima`: `supporto_box` con `y1 = 6000` → `nessuna_stima.motivo` contiene «base».
  - `test_pdf_con_misure`: PDF base64 valido (inizia con `%PDF`).
  - `test_cache_max_5`: dopo 6 inserimenti tramite la funzione `ricorda_analisi(...)` di `main`, la più vecchia non c'è più.
  - `test_analyze_invariato`: le chiavi della risposta di `classify` sono le stesse della v2 (confronta con la lista letterale delle 7 chiavi di `main.py:848-851`).
- [ ] **Step 2: Esegui.** Atteso: FAIL.
- [ ] **Step 3: Implementa** `ricorda_analisi(result_id, **voce)` in `main.py`, chiamata alla fine di `classify` con i dati sopra.
  - `analyze` passa a `classify` anche `meta_bytes` (i byte di `meta_source` se c'è, altrimenti quelli di `file`) e `analysed_size=img.size`. Le firme pubbliche dell'endpoint restano invariate.
  - Poi implementa `misura/api.py` e includilo in `main.app`.
- [ ] **Step 4: Esegui.** `pytest tests -q`: tutti i test passano, compresi quelli della v2.
- [ ] **Step 5: Commit.** Messaggio: `misura: API endpoints and in-memory analysis cache`.

---

### Task 8: Pagina «Misure» nel PDF

**Files:**
- Modify: `dinov2-corrosion-v3/report_pdf.py` (`build_pdf`, riga ~239, più una nuova funzione `_pagina_misure`)
- Test: `tests/test_pdf_misure.py`

**Interfaces:**
- Produces: `build_pdf(photo_name, roi_image, classified_image, report, operator_notes="", clusters=None, shot_map=None, misure=None)`. `misure` è una lista di `Misura.to_dict()`.

- [ ] **Step 1: Test che falliscono.**
  - `test_senza_misure_identico`: con `misure=None` il numero di pagine e il testo estratto (usa `pypdf` se c'è nel venv, altrimenti conta `/Type /Page` nei byte) sono uguali alla chiamata senza il parametro.
  - `test_con_misure`: con 2 misure (una classe STIMA e una linea NON VALIDATA) c'è una pagina in più. Il testo contiene «Misure», «STIMA», «NON VALIDATA», «±», «Superficie reale stimata» e la riga esplicativa.
- [ ] **Step 2: Esegui.** Atteso: FAIL.
- [ ] **Step 3: Implementa `_pagina_misure(canvas, misure)`** con lo stile delle pagine esistenti.
  - Tabella con colonne: Misura | Valore ± U95 | Superficie reale stimata (solo per le classi) | Esito.
  - Le aree sono in m² con 2 decimali. Le lunghezze in m con 2 decimali, oppure in cm se < 1 m.
  - Riga esplicativa fissa: «Area vista = ruggine visibile proiettata sulla faccia del sostegno. Superficie reale stimata = area vista × fattore di sviluppo dei profilati (parti nascoste supposte come quelle visibili): sempre STIMA.»
  - Sotto la tabella: drone, fonte della distanza e, se ci sono, i controlli della sessione.
- [ ] **Step 4: Esegui.** Atteso: PASS.
- [ ] **Step 5: Commit.** Messaggio: `report: optional Misure page`.

---

### Task 9: UI A – pulsante Misura e pannello in «Anteprima ingrandita»

**Files:**
- Modify: `dinov2-corrosion-v3/static/index.html` (header del viewer, riga ~209–218; CSS; script IIFE)
- Test: `tests/check_ui_misura.py` (script CDP headless Edge sulla porta 8799, sul modello dei test CDP già usati per la v2: legge il DOM, non le globali)

**Interfaces:**
- Consumes: gli endpoint del task 7; nel client, `last.report.result_id`, il box della ROI automatica già scelto (`autoRoiInfo` / `focusBox`) in pixel originali, e le coordinate del viewer.
- Produces, come id del DOM:
  - `#vMeasure`: pulsante «📐 Misura», visibile solo se `last` esiste;
  - `#measurePanel`, con le sezioni `#mpPhoto`, `#mpAuto`, `#mpManual`, `#mpPrecise`;
  - `#mpLine`, `#mpArea`: strumenti;
  - `#mpList`: righe con checkbox `.mp-pdf`;
  - `#mpOpenSession`.

Comportamento:
- **Accensione.** Il pulsante «📐 Misura» è spento di default e non viene salvato nei preset. Quando lo accendi, il pannello occupa la colonna destra del viewer (come nella bozza A) e si chiama `/api/misura/stima`. Finché non arriva la risposta, la barra `setBar` mostra «Calcolo misure…».
- **Riquadri del pannello.**
  - Foto e drone: drone, RTK, distanza e fonte, scala al centro ed etichetta colorata (STIMA gialla, VALIDATA verde, NON VALIDATA rossa).
  - Misure automatiche: per ogni classe l'area vista ± U95 e, sotto, la «superficie reale stimata». Per le anomalie le dimensioni, che vengono anche disegnate in `#vSvg` accanto ai riquadri A1, A2…
  - Misure a mano: Linea (clic sui vertici, doppio clic per finire) e Area (clic sui vertici, chiusura con doppio clic). Le linee sono blu, con l'etichetta del valore. Il tasto Canc elimina la misura selezionata. Ogni modifica richiama `/api/misura/stima`, o `/sessione/calcola` se la sessione esiste e ha un esito.
  - Fattore di sviluppo: un campo numerico in `#mpAuto`, valore iniziale 3,0, salvato nei preset come chiave `fattore_sviluppo`. In `main.py` aggiungi `"fattore_sviluppo"` a `PRESET_KEYS`.
- **Nessuna stima.** Il pannello mostra motivo e suggerimento. Se manca il riquadro, attiva «clicca la base del sostegno»: il clic successivo sulla foto imposta `base_px`.
- **PDF.**
  - Ogni misura ha una casella «PDF», spenta di default.
  - Se almeno una è spuntata, il pulsante PDF della colonna risultati chiama `/api/misura/pdf` e scarica il PDF rigenerato.
  - Se non ce ne sono, scarica il PDF originale, come nella v2.

- [ ] **Step 1: Test che fallisce.** Scrivi `tests/check_ui_misura.py`, che:
  1. avvia il server v3 sulla 8799;
  2. carica la pagina in Edge headless e la foto Camera_4;
  3. lancia un'analisi veloce (risoluzione minima) e apre il viewer;
  4. controlla che `#vMeasure` esista e sia spento, e che `#measurePanel` sia nascosto;
  5. clicca `#vMeasure` e aspetta che `#mpPhoto` contenga «M3T» e «STIMA»;
  6. disegna una linea con due clic e un doppio clic, e controlla che `#mpList` contenga « m ±»;
  7. spunta `.mp-pdf`, clicca PDF e controlla nel DOM o nella rete che sia partito `/api/misura/pdf`;
  8. ferma solo il proprio server.

  Esegui lo script. Atteso: FAIL al punto 4.
- [ ] **Step 2: Implementa** la UI nel file `index.html`, nello stile esistente (pulsanti `.grp`, `.rsec`) e seguendo la bozza A.
- [ ] **Step 3: Esegui `check_ui_misura.py`.** Atteso: tutti i controlli OK. Poi `pytest tests -q`: tutto verde.
- [ ] **Step 4: Controllo a vista.** Fai uno screenshot headless del viewer con il pannello aperto e confrontalo con la bozza A. Lo screenshot va mostrato all'utente prima di chiudere il task: l'utente vuole vedere gli esempi prima di cambi grandi della UI.
- [ ] **Step 5: Commit.** Messaggio: `ui: Misura panel in enlarged view`.

---

### Task 10: UI B – finestra della sessione multi-foto, documentazione, verifica finale

**Files:**
- Modify: `dinov2-corrosion-v3/static/index.html`
- Modify: `dinov2-corrosion-v3/LEGGIMI_V3.md`
- Test: `tests/check_ui_sessione.py` (CDP, porta 8799)

**Interfaces:**
- Consumes: gli endpoint `/api/misura/sessione/*` (task 7).
- Produces, come id del DOM:
  - `#sessionDlg`;
  - `#sdStrip`: miniature `.sd-thumb`, con la ★ per la principale; ogni miniatura mostra lo yaw e un badge «non misurabile» con il motivo;
  - `#sdAdd`: input file, multipli;
  - `#sdLeft` (sempre la principale), `#sdRight` (la miniatura selezionata): con zoom e trascinamento come nel viewer;
  - `#sdRequired`, `#sdTies`, `#sdChecks`, `#sdCalc`.

Comportamento:
- La finestra si apre da `#mpOpenSession` (`/sessione/apri`).
- Come in DISTIX, sono tre gesti:
  1. «＋ Foto»;
  2. clic sui punti;
  3. «Calcola».
- **Punti richiesti** (testo fisso, generalista per tralicci):
  1. «T1, T2: piedi dei montanti alla stessa quota»;
  2. «T3, T4: nodo superiore della stessa faccia»;
  3. «(facoltativi) altri nodi ben riconoscibili».
- **Come si segnano i punti.**
  - Il punto attivo si sceglie in `#sdTies`. «＋ punto» crea T5, T6…
  - Un clic sulla foto lo posiziona; Alt+clic lo cancella da quella foto.
  - Quando un punto c'è nella principale e manca nella foto di destra, si disegna la curva di `/sessione/suggerisci`, tratteggiata in blu, con il testo «dovrebbe essere qui».
- **Elenco dei tie point** (`#sdTies`): per ogni punto, «Tn · k/N foto» e l'errore in px dopo Calcola.
- **Controlli** (`#sdChecks`): una spunta o un avviso per ogni controllo, l'esito colorato e i messaggi in parole semplici.
- **Dopo Calcola** il pannello Misura (task 9) passa al proiettore della sessione. Le etichette diventano VALIDATA o NON VALIDATA. La superficie reale resta STIMA.
- **Salvataggio.** La sessione si salva sul server a ogni modifica. Riaprendo la stessa foto principale si ricarica.

- [ ] **Step 1: Test che fallisce.** Scrivi `tests/check_ui_sessione.py`, che:
  1. usa la stessa preparazione del task 9;
  2. apre la sessione e controlla che `#sdStrip` abbia 1 miniatura con ★;
  3. aggiunge Camera_3 (stesso drone, altro sostegno: va bene per testare la UI) e controlla che le miniature siano 2;
  4. clicca T1 nella principale e controlla che a destra compaia la curva suggerita (un `path` dentro `#sdRight svg`);
  5. clicca Calcola con meno di 4 punti e controlla che `#sdChecks` mostri «INCOMPLETA» e un messaggio con «T»;
  6. chiude, riapre e controlla che T1 sia ancora presente.

  Atteso: FAIL.
- [ ] **Step 2: Implementa** la finestra seguendo la bozza B.
- [ ] **Step 3: Esegui `check_ui_sessione.py` e `check_ui_misura.py`, poi `pytest tests -q`.** Atteso: tutto OK.
- [ ] **Step 4: LEGGIMI_V3.md.** Aggiungi in italiano:
  - come si usa Misura;
  - cosa vogliono dire STIMA, VALIDATA e NON VALIDATA;
  - area vista e superficie reale, con il fattore di sviluppo;
  - come aggiungere un drone (`profili_drone/`);
  - perché le M3T restano NON VALIDATA senza un profilo di calibrazione;
  - limiti: niente zoom, niente termiche, superficie reale sempre stimata.
- [ ] **Step 5: Verifica finale sulle foto reali.** Sulla 8799 analizza Camera_4 e Camera_6 con Misura accesa. Annota per entrambe:
  - drone riconosciuto;
  - fonte della distanza (Camera_4 geometria, Camera_6 laser);
  - scala in alto e in basso;
  - m² di ruggine intensa ± U95.

  Controlla che nel PDF compaia la pagina Misure solo con le caselle spuntate. Mostra all'utente i numeri e gli screenshot. Confermali o smentiscili con i dati, senza abbellirli.
- [ ] **Step 6: Commit.** Messaggio: `ui: multi-photo tie-point session; docs`. Poi chiedi all'utente se vuole il push.
