# DINOv2 Corrosion Intelligence — Eagle Projects (PoC)

**Real Meta DINOv2-small Vision Transformer**, CPU-only, Windows VM, local browser UI. Research prototype for visible corrosion in high-voltage steel transmission towers, not a validated structural inspection system.

## What it does
1. Upload JPG/PNG/WEBP image. For files above 18 MB or 40 MP, the browser creates a local JPEG copy of at most 32 MP and 18 MB before upload; the original file remains unchanged. The interface shows both sizes. JSON `source_dimensions_px` and the source SHA-256 refer to the analyzed copy.
2. Wait for automatic local BiRefNet background removal, then select a region of interest.
3. Click representative samples of **healthy metal** and visible **medium or intense corrosion**. Background comes from the automatic mask. Do not invent examples of a corrosion class absent from the photo; add that class later from another photo.
   Use **+**, **−** and **Adatta** above the photo to zoom while placing points;
   scroll the enlarged view to reach other parts. Zoom changes only the display,
   not the image coordinates or the DINOv2 inference resolution.
4. Choose **Dettaglio · riquadri sovrapposti**, give the profile a name and press **Salva profilo e analizza**. Real DINOv2 features are extracted on CPU and compared with the individual annotated examples.
5. View an overlaid corrosion map and measurements. Export the overlay PNG,
   class-ID mask PNG, JSON report and CSV; `class_id_labels` maps mask IDs to names. ID 4 marks metal whose rust-versus-healthy margin is too small and needs review.
6. Give the calibration a profile name. On another photo, select that profile and press **Analizza con profilo** to reuse it.
7. To improve a named profile with examples from another support, load the photo,
   mark representative classes, and press **Aggiungi esempi al profilo**. DINOv2
   extracts patch embeddings locally and retains distinct examples in new schema-2 profiles.
   The source photos are not retained. The local numeric profile is retained.

## Rimozione sfondo integrata e utility separata facoltativa

Dal Desktop, aprire con doppio clic **`Rimuovi sfondo locale.bat`** nella cartella
`DINOv2 CORROSION INTELLIGENCE` oppure trascinarvi sopra le fotografie.
Selezionare una o più fotografie: il programma
usa BiRefNet General tramite rembg e ONNX Runtime **solo sulla CPU locale**,
senza inviare le immagini a un sito. Apre la cartella
`model_cache/sfondo_rimosso` con, per ogni foto, un PNG trasparente e una
maschera in scala di grigi. I file mantengono le dimensioni della foto originale.
L'ambiente sperimentale e i pesi sono in `model_cache` e non vanno pubblicati.

La dimensione del PNG esportato non coincide con la risoluzione interna del
modello: BiRefNet General lavora a circa 1024 pixel e la maschera viene riportata
alle dimensioni originali. Per dettagli sottili sono utili ritagli ravvicinati.
Su un ritaglio locale di 3800×2500 pixel, il modello ha impiegato circa 31 s
sulla CPU e ha conservato gran parte delle aste e dei vuoti. Un confronto con
BiRefNet Dynamic a 2048×1344 pixel ha richiesto circa 115 s e ha incluso più
cavi/isolatori. Nessuno dei due risultati è una maschera validata del **solo
metallo**: cavi, isolatori e altre parti possono rimanere in primo piano.

Nell'app basta caricare la foto: BiRefNet genera automaticamente la maschera
nello stesso tool, usando un ambiente `rembg[cpu]` isolato in
`model_cache/seg_env`. La foto viene inviata solo all'API locale e rimane in
memoria temporanea durante l'elaborazione. La maschera non viene salvata come
file; overlay, maschera classi, JSON e CSV usano gli stessi conteggi vincolati
al primo piano. Il report registra hash, soglia e numero di pixel ammessi.
BiRefNet può mantenere cavi e isolatori in primo piano: restringere la ROI al
sostegno. I conteggi sono sul raster esportato (lato massimo 1100 px), non una
misura geometrica sul file originale da 4K. L'installatore Windows prepara
anche l'ambiente CPU isolato per la rimozione sfondo.

Per escludere un isolatore rimasto nella maschera, premere **Escludi isolatore**,
impostare la larghezza in pixel della foto originale e trascinare dal primo
all'ultimo disco. La capsula viola viene sottratta ai pixel classificati prima
di produrre overlay, maschera classi e percentuali. Ogni esclusione, con
coordinate e pixel eliminati, appare nel report JSON/CSV. Questo passaggio è
geometrico e manuale per foto; controllare che la capsula non copra aste.
Il riconoscimento automatico degli isolatori per i batch richiede dati annotati
e una valutazione separata.

### Aste sottili non rilevate

Su una ROI alta e stretta, DINOv2 riduce il lato lungo a 392 o 518 pixel.
Per esempio, su una ROI di 728×2697 px a 518 px, una patch rappresenta circa
73 px della foto originale: una diagonale sottile può essere mescolata con
metallo sano e sfondo. Lo zoom del browser non cambia questa risoluzione.

Per verificare una zona specifica senza perdere il profilo salvato, selezionare
una ROI stretta attorno all'asta, scegliere 518 px, marcare metallo sano e
corrosione effettivamente visibile, quindi premere **Prova dettaglio ROI**.
Il risultato usa DINOv2 reale e un profilo temporaneo, indicato in JSON/CSV
come `profile_persisted: false`; non modifica `local_profile.json`.
La percentuale risultante riguarda solo la ROI di dettaglio e non sostituisce
la stima della ROI ampia. L'esito sulla stessa foto serve a diagnosticare la
risoluzione e la calibrazione, non prova l'accuratezza su altre fotografie.
La modalità Dettaglio esegue una scansione a riquadri sovrapposti, con
fusione delle caratteristiche sui bordi. Su immagini grandi adatta la scala
per restare entro 64 riquadri CPU; il report indica i pixel sorgente per patch.
È più lenta e conserva più dettaglio;
non ricostruisce dettagli assenti nella foto. Il profilo deve essere calibrato
con questa modalità: quelli vecchi restano in modalità rapida.

### Aggiornamento 8 novembre 2026

- Nuovi profili schema 2: campioni individuali, hash della foto, ROI, patch,
  risoluzione e modalità. Confronto con la media delle tre maggiori similarità
  per classe (o dei campioni disponibili se meno di tre). Nessun fine-tuning
  dei pesi DINOv2. Nessuna fotografia salvata nel profilo.
- Duplicati della stessa foto/ROI/configurazione/patch ignorati, classi in
  conflitto rifiutate. Massimo 2048 campioni per profilo, errore esplicito oltre
  il limite. Le scritture concorrenti non sovrascrivono aggiornamenti intervenuti.
- Analizza e Aggiungi esempi usano la configurazione salvata nel profilo.
  Il selettore Dettaglio/Rapido e la risoluzione si applicano ai nuovi profili
  e alle prove temporanee. I vecchi profili e le loro medie restano invariati.
- Mappa delle ambiguità separata: bianco indica pixel classificati metallo con
  margine coseno inferiore a 0,025. La quota è calcolata sul metallo classificato.
  I candidati ruggine con margine inferiore a 0,025 rispetto al sano diventano
  grigi (classe 4, da verificare): restano nel denominatore del metallo visibile
  ma non nel numeratore della ruggine. La soglia è sperimentale, non una probabilità.
- Le classi senza esempi appaiono come **Non calibrata**, anziché suggerire una
  assenza di corrosione accertata. Nessuna etichetta di riferimento viene inventata.
- Su ROI reale 866×2807: griglia contenuto da circa 11×37 a 41×133 patch,
  passo effettivo da circa 76 a 21 pixel originali. 10 riquadri, 14,64 s di
  forward CPU (maschera BiRefNet esclusa). Il vecchio profilo resta inadeguato;
  maggiore dettaglio non dimostra maggiore accuratezza.

Collaudo e ricerca dei progetti esistenti: [MIGLIORAMENTI_2026-11-08.md](MIGLIORAMENTI_2026-11-08.md).

### Profili nominati e campioni da più sostegni

I profili nominati sono salvati localmente in `local_profiles/`, uno per
ambiente o gruppo di sostegni. Selezionare il profilo prima dell'analisi.
Per aggiungere esempi da un altro sostegno, selezionare marker coerenti con le
classi esistenti e usare **Aggiungi esempi al profilo**. Il profilo conserva
embedding numerici e conteggi, senza fotografie. I nuovi profili conservano
anche hash della foto e coordinate ROI/patch per tracciare gli esempi.
I campioni di più sostegni possono aumentare la varietà visiva, ma non
garantiscono accuratezza: mantenere separati ambienti con illuminazione,
verniciatura o condizioni diverse se peggiorano la separazione delle classi.
Il vecchio `local_profile.json` resta selezionabile come profilo legacy; per
aggiungere campioni va creato un nuovo profilo nominato.

## What the values mean
- **Corrosion %** = predicted corrosion pixels / predicted visible metal pixels **within the image or selected ROI**. This is *NOT* the percentage of the entire 3D tower surface.
- **Visual intensity (1–3)** = corrosion-pixel-weighted average of the operator-defined visual classes (only present classes). This is *NOT* depth, steel loss, mechanical integrity, or a validated technical standard.
- **Low similarity margin** is only a heuristic of uncertain separation, **NOT** a calibrated probability.
- Coarse patch embedding resolution means small rust defects may be missed; image colors/light conditions strongly influence model behavior.
- Calibrating and testing on the *same image* is not an independent accuracy evaluation. Validate on different towers, lighting and sites before any real claims.

## Windows CPU installation — VM Brasile
- Python **3.11 64 bit**, Internet for first install/download, minimum approx. 4 GB **available** RAM (8 GB preferable), several GB free disk.
- From a PowerShell prompt in this folder, or Explorer:
  1. Run `installa_windows.bat` (isolated `.venv`, PyTorch CPU).
  2. Run `precarica_modello_windows.bat` while Internet is available.
  3. Run `avvia_server.ps1` in PowerShell for a detached local server, or
     `avvia_windows.bat` for an interactive session.
  4. Open **http://127.0.0.1:8766** **in the VM browser**.
- Per il normale utilizzo basta il doppio clic su **Avvia DINOv2 Corrosion
  Intelligence.bat**, nella cartella superiore: avvia il server e apre il browser.
- No Docker and no GPU. Do not expose the unprotected app to the public Internet.
- For Linux: `bash installa_linux.sh` then `bash avvia_linux.sh`.

## Reproducibility / tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Unit tests mock embeddings and **do not** establish real DINOv2 inference or corrosion accuracy. Validate actual model loading with the preloader and a real representative photo after installation.

On 2026-11-07, the HTTP workflow was exercised with two different locally
generated synthetic images and real DINOv2-small CPU inference. Calibration,
profile reuse, PNG decoding, class-mask counts and CSV percentages passed.
This validates software flow only; accuracy requires independent real photos
with expert annotations.

## Security / privacy
- Web service binds **127.0.0.1 only** by default; no password/HTTPS. Use only VM browser/RDP. Any network exposure requires VPN, authentication, firewall and TLS before production.
- Photos processed locally; the normal app doesn't retain source photos. Legacy
  profiles save numeric prototypes; schema-2 profiles save numeric exemplars and
  source hash/ROI/patch provenance. Diagnostic scripts can explicitly save local
  overlays under `model_cache`; those artifacts must not be published.
- Never commit `local_profile.json`, photos, `.venv`, caches, model weights or customer data to GitHub.
- First download uses public Hugging Face; ensure corporate permissions for data and downloads.
- Model: Meta DINOv2-small, check model card/license before distribution: https://huggingface.co/facebook/dinov2-small

## Continuation for Codex
Maintain browser-only UX, CPU execution and isolation from VM Brasil tools. Real-weight HTTP smoke tests have passed with synthetic calibration examples. Next steps: sample dataset with expert masks on independent towers; confusion matrix, mIoU/Dice, percentage MAE and CPU latency; multi-view geometry for whole-tower coverage estimates; standards-based severity categories separately validated by qualified experts.
