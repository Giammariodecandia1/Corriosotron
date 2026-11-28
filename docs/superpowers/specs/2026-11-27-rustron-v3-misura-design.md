# Rustron v3: misurare aree e lunghezze in metri e centimetri

Data: 2026-11-27. Stato: bozza da approvare.
Bozza grafica di riferimento: `.superpowers/brainstorm/misura-v3-bozza-1.html` (approvata dall'utente come direzione).

## 1. Perché

Oggi Rustron dice *quanta* ruggine c'è in percentuale sul metallo visibile, ma non *quanto è grande*. Per pianificare gli interventi serve sapere, ad esempio, «la ruggine intensa copre 0,8 m²», «l'anomalia A3 è lunga 45 cm», «questo tratto di montante è lungo 2,3 m».

DISTIX (repo `https://github.com/MrDaSp/foto-misura`, codice dell'utente e di un collega, quindi riusabile liberamente) sa già misurare in 3D da foto DJI con tie point, incertezza GUM e regola «validata / non validata». La v3 porta quelle capacità dentro Rustron, senza la sua interfaccia PyQt.

## 2. Come si usa (deciso dall'utente)

- La misurazione è una **funzione secondaria**. Si attiva solo **dopo l'analisi**, con un pulsante «📐 Misura» nella schermata **Anteprima ingrandita**. Se non lo accendi, la v3 si comporta esattamente come la v2.
- Due livelli di misura, entrambi richiesti:
  1. **Stima da una foto sola** (automatica). Usa i metadati della foto e il riquadro del sostegno trovato da YOLO. Esce sempre con una fascia di incertezza ed etichetta **STIMA**.
  2. **Sessione a più foto (≥ 2) con tie point**, come in DISTIX. Lo stesso punto fisico viene cliccato in almeno 2 foto e Rustron lo ricostruisce in 3D. Esito **VALIDATA** o **NON VALIDATA** secondo i controlli di DISTIX.
- Cosa si misura (tutte e due le cose):
  - **automatico**: area in m² di ogni classe di ruggine (media, intensa, grigie) e dimensioni (lunghezza × larghezza in cm, area) di ogni anomalia numerata A1, A2…;
  - **a mano**: l'operatore traccia una **linea** (lunghezza) o un **poligono** (area) sulla foto.
- **Report PDF**: ogni misura ha una casella «PDF». Le misure spuntate finiscono in una pagina «Misure» opzionale del report, con incertezza ed etichetta (STIMA / VALIDATA / NON VALIDATA). Senza misure spuntate il PDF è identico a quello di oggi.
- **Adattamento al drone**: Rustron legge i metadati della foto, capisce che drone e camera l'hanno scattata e si comporta di conseguenza (sensore, focale, RTK, telemetro laser, calibrazione).
- **Niente zoom**: si assume che le foto siano fatte con la camera grandangolare a focale fissa.
- **Realtà d'uso**: quasi sempre ci sarà solo la foto principale (quella analizzata). Le foto in più si scattano apposta quando serve una misura precisa. Quindi la stima da una foto è il caso normale, la sessione a più foto è il caso raro ma «di qualità».

## 3. Obiettivi, non obiettivi, cose da evitare

**Obiettivi**
- Misure in m e cm con incertezza dichiarata, da una foto o da più foto.
- Riconoscimento automatico del drone dai metadati, con profili estendibili.
- Riuso delle parti di calcolo di DISTIX (metadati DJI, gimbal → direzioni, pixel → raggio, triangolazione, incertezza, validazione).
- Misure opzionali nel PDF.

**Non obiettivi**
- Zoom, camere tele, foto termiche (R-JPEG): non si misurano.
- Fotogrammetria automatica completa (abbinamento automatico dei punti, nuvola di punti). I tie point li mette l'operatore. Rustron può solo *suggerire* dove cercare il punto nelle altre foto.
- Valutazioni strutturali (perdita di spessore, sicurezza).
- Interfaccia PyQt di DISTIX.

**Cose da evitare (sarebbe un fallimento anche se funziona)**
- Presentare una stima come misura certa: ogni numero deve portare incertezza ed etichetta.
- Toccare la poc (porta 8765) o la v2 (porta 8766).
- Rallentare l'analisi normale: nessun calcolo di misura se «Misura» è spento.
- Inventare coordinate o date: arrivano solo dai metadati della foto.
- Specializzare il tool su un solo drone o un solo tipo di sostegno: deve restare generalista.

## 4. Vincoli

- La v3 vive in una **cartella nuova** `dinov2-corrosion-v3` accanto a v2 e poc, partendo da una copia della v2, con un suo avvio (`AVVIA RUSTRON V3.bat`) e la **porta 8767**. *(Scelta mia: segue lo schema già usato per poc → v2.)*
- Interfaccia solo web, nello stesso `static/index.html` della v2 (script IIFE, FastAPI).
- Non riavviare mai il server dell'utente: i test girano su una porta separata (es. 8799).
- Commit e push solo su richiesta.
- La qualità della maschera di ruggine non cambia: la misura usa la maschera così com'è.

## 5. Come funziona

### 5.1 Riconoscimento del drone

- Si leggono EXIF (Make, Model, FocalLength, FocalLengthIn35mmFormat, dimensioni immagine, GPS) e XMP DJI (`drone-dji:` GimbalYaw/Pitch/RollDegree, FlightYaw…, AbsoluteAltitude, RelativeAltitude, RtkFlag, RtkStdLon/Lat/Hgt, LRFTargetDistance e stato LRF, DewarpData/DewarpFlag, CalibratedFocalLength, ecc.). Si riusa `acquisition/metadata_reader.py` di DISTIX.
- **Profili drone**: un file JSON per modello in `dinov2-corrosion-v3/profili_drone/`, scelto da Make + Model (+ nome camera XMP se c'è). Contiene: larghezza sensore in mm, risoluzione nativa, focale nominale della camera wide, presenza di RTK e LRF, incertezze tipiche (gimbal, focale), e se presente un **profilo di calibrazione** (fx, fy, cx, cy, distorsione). Profili iniziali per i droni effettivamente trovati nelle foto dell'utente: **DJI M3T** (Camera_4: RTK FIX; LRF «Normal» ma 0,000 m, quindi da ignorare) e **DJI H30T su M400** (Camera_6: `ImageSource="WideCamera"`, LRF valido a 30,2 m). Altri si aggiungono copiando un file. Un LRF che vale 0 o è fuori portata va trattato come assente.
- **Drone sconosciuto**: si usano solo i dati EXIF standard (focale + equivalente 35 mm). Il risultato è marcato «profilo generico» con incertezza più larga.
- **Controllo zoom**: se la focale della foto differisce di più del 5 % da quella nominale del profilo, o la foto è termica, la foto è marcata «non misurabile» e non si usa per le misure. *(Soglia 5 %: scelta mia.)*
- Focale in pixel: dalla calibrazione se c'è, altrimenti da DewarpData/CalibratedFocalLength se c'è, altrimenti dall'equivalente 35 mm. Nota: DISTIX (`calibration_manager.resolve_dji_calibration`) oggi **non** ricava fx dall'equivalente 35 mm. Nella v3 questo ripiego va aggiunto, così le foto M3T restano misurabili.

### 5.2 Stima da una foto

Il punto chiave è la **distanza tra camera e sostegno**. In ordine di preferenza:
1. **Telemetro laser (LRF)** valido nei metadati e puntato sul sostegno (il centro dell'immagine cade dentro il riquadro del sostegno) → distanza misurata.
2. **Geometria**: il raggio dal centro camera al punto più basso del riquadro del sostegno (base del traliccio) viene intersecato con il piano del suolo alla quota di decollo (RelativeAltitude). Si ottiene la distanza orizzontale. È stato verificato su Camera_4: 74,5° sotto l'orizzonte, quota 35,17 m → ~9,75 m, sostegno ~31 m, ~4 mm/px.
3. Se la base del sostegno non è visibile (riquadro che tocca il bordo basso della foto) e non c'è LRF → **nessuna stima**, con un messaggio chiaro che suggerisce la sessione a più foto.

**Quale riquadro è il sostegno**: quello scelto come ROI automatica della v2 (riquadro in modalità «roi», sostegno centrale più sicuro). Se non c'è un riquadro «roi», l'operatore clicca a mano la base del sostegno nel pannello Misura. Nessun nome di classe è scritto nel codice: resta generalista.

**Scala pixel per pixel, non una scala unica**: il sostegno è modellato come un **piano verticale** che passa per la base trovata sopra ed è perpendicolare alla direzione orizzontale di vista. Il raggio di **ogni pixel** viene intersecato con quel piano. Le misure si fanno sulle coordinate 3D ottenute (area del pixel = area del suo quadrilatero sul piano). Questo è necessario: con il drone in alto la scala cambia di 3–4 volte tra la cima e la base (Camera_4: ~4 mm/px in alto, ~13 mm/px alla base). Una scala unica sbaglierebbe la ruggine alla base di circa 10 volte. *(Correzione emersa dalla revisione.)* Il pannello mostra la scala al centro del riquadro solo come informazione. Aree di ruggine = somma delle aree dei pixel della classe. Anomalie, linee e poligoni a mano si calcolano con lo stesso piano.

**Cosa significa un'area in m²**: vedi §5.5.

**Incertezza** (stile GUM come DISTIX, U95 con k = 2), che combina:
- quota del suolo diversa da quella di decollo (default ±3 m, che su Camera_4 vale circa 8 %);
- angoli del gimbal;
- focale (più larga senza calibrazione);
- errore di clic (σ 1,5 px come DISTIX);
- inclinazione reale della faccia rispetto al piano verticale assunto. Con il traliccio rastremato e le facce ruotate, questo diventa il termine più grande.

I valori di default stanno nel profilo drone.

### 5.3 Sessione a più foto con tie point

- **Deve essere semplice da gestire come in DISTIX** (richiesta esplicita dell'utente). L'operatore fa solo tre cose: aggiunge le foto, clicca i punti richiesti, preme «Calcola». Bundle adjustment, triangolazione, incertezza e controlli girano dietro le quinte, senza parametri da regolare. Se qualcosa non va, Rustron dice in parole semplici cosa fare (es. «T4 manca in una seconda foto», «le foto sono troppo allineate: scattane una più di lato»).
- Si apre dal pannello Misura («＋ Apri sessione multi-foto»). La foto analizzata è la **principale (★)**. Si caricano 1 o più foto in più dello stesso sostegno.
- **Punti richiesti**, adattati al traliccio: T1 e T2 sono i piedi dei montanti alla stessa quota, T3 e T4 il nodo superiore della stessa faccia, più punti facoltativi su altri nodi ben riconoscibili. Servono almeno 4 tie point, ognuno visto in almeno 2 foto. L'utente ha confermato che i punti vanno bene così.
- Dopo i primi punti, Rustron **suggerisce** dove cercare il punto nelle altre foto (linea epipolare o proiezione stimata), ma è l'operatore a cliccare.
- **Calcolo**: triangolazione di ogni tie point (`core/triangulation.py`, `core/camera_model.py` di DISTIX, con i controlli di cheiralità e convergenza). Dai tie point si ricava il **piano della faccia** del sostegno. I pixel della foto principale (maschere di ruggine, anomalie, linee e poligoni a mano) vengono proiettati su quel piano e misurati lì. Le stime della foto singola vengono così sostituite da misure 3D.
- **Controlli e validazione** (come DISTIX):
  - RTK FIX su tutte le foto usate;
  - convergenza ≥ 15°;
  - errore di riproiezione sotto soglia (stessa soglia di DISTIX);
  - profilo di calibrazione della camera presente.

  Se passano tutti: **VALIDATA**. Altrimenti **NON VALIDATA**, con il motivo scritto. Le misure escono comunque, con U95. Con le M3T senza calibrazione l'esito sarà NON VALIDATA finché non si aggiunge un profilo di calibrazione.
- Per ogni tie point si mostrano in quante foto è stato segnato e l'errore in px.
- La sessione si salva insieme all'analisi, così si può riaprire e correggere. *(Scelta mia.)*

### 5.4 Codice DISTIX da riusare

I moduli si copiano in un pacchetto `dinov2-corrosion-v3/misura/`, con una nota di provenienza:

| Modulo DISTIX | Cosa fa |
| --- | --- |
| `acquisition/metadata_reader.py` | Lettura dei metadati DJI, `dji_euler_to_vectors` |
| `core/camera_model.py` | `pixel_to_ray`, modello pinhole con distorsione Brown-Conrady |
| `core/triangulation.py` | Triangolazione da più camere |
| `core/uncertainty_engine.py` | Incertezza GUM |
| `core/coordinate_system.py` | Sistemi di coordinate |
| `acquisition/calibration_manager.py` | Gestione della calibrazione, con l'aggiunta del ripiego sull'equivalente 35 mm |

Si portano anche:
- **`core/bundle_adjustment.py`** (`run_constrained_bundle_adjustment_with_diagnostics`), che in DISTIX affina le pose prima della misura. Senza di esso, con solo RTK e gimbal grezzi l'errore di riproiezione difficilmente rientra nella soglia, e VALIDATA non uscirebbe quasi mai;
- la **regola VALIDATA / NON VALIDATA**, che in DISTIX sta in parte in `gui/` e `reporting/`. Va estratta in una funzione pura dentro `misura/`.

Il riferimento per «come DISTIX» è il flusso di misura reale di `gui/main_window.py`: metadati → tie point → bundle adjustment vincolato → triangolazione → incertezza → validazione. L'utente ha chiesto di mantenere «le ottime caratteristiche di DISTIX».

`core/ruler_engine.py` va usato come riferimento per la misura A–B con U95. Non si portano l'interfaccia PyQt né `pole_geometry.py` (specifico dei pali BT). Il clone in sola lettura usato per l'analisi si trova nello scratchpad della sessione; il builder parte dal repo GitHub.

### 5.5 Area vista e superficie reale

Su un traliccio a grigliato la foto mostra insieme la faccia davanti, quella dietro e le diagonali, che stanno a profondità diverse. Ogni asta è un profilato (di solito un angolare a L) con facce nascoste. Per questo il report dà **due numeri** per ogni classe di ruggine (decisione dell'utente):

1. **Area vista**: l'area dei pixel della classe proiettata sul piano della faccia (§5.2) o sul piano ricostruito dai tie point (§5.3). Porta l'etichetta della misura da cui viene: STIMA, VALIDATA o NON VALIDATA.
2. **Superficie reale stimata** = area vista × **fattore di sviluppo** del profilato, cioè il rapporto tra l'acciaio totale dell'asta e la parte che se ne vede. Si suppone che le parti nascoste siano arrugginite come quelle visibili. Ha sempre etichetta **STIMA**, anche nella sessione a più foto, e un'incertezza più larga che include quella del fattore.

- Il fattore parte da un **valore medio** per angolari a L, modificabile dall'utente. Sta nelle **impostazioni salvate** (preset), così si cambia per tipo di traliccio senza toccare il codice. *(Il valore iniziale preciso e la sua incertezza li sceglie il builder con una giustificazione geometrica breve nel codice: ordine di grandezza 2–4.)*
- Il PDF spiega in una riga cosa significano i due numeri.
- Una superficie reale misurata davvero, con un modello 3D completo da molte foto tutt'intorno, è fuori da questo progetto.

## 6. Interfaccia (vedi bozza)

- **Anteprima ingrandita**: nuovo pulsante «📐 Misura». Se è acceso, a destra si apre un pannello con quattro riquadri:
  - **Foto e drone**: drone, RTK, distanza, scala ed etichetta;
  - **Misure automatiche**: le misure sulla ruggine;
  - **Misure a mano**: strumenti Linea e Area, con l'elenco delle misure e la casella PDF;
  - **Misura precisa**: il pulsante della sessione a più foto.
- Le misure si disegnano sopra la mappa ingrandita: linee blu con etichetta e anomalie con le dimensioni.
- **Sessione a più foto**:
  - in alto, la striscia di miniature (★ principale, yaw di ogni foto, «＋ Foto»);
  - al centro, due foto affiancate per cliccare i punti;
  - a destra, i punti richiesti, l'elenco dei tie point e i controlli con l'esito.
- Tutto in italiano, come il resto di Rustron.

## 7. Lasciato al builder

- La struttura del codice dentro `misura/` e gli endpoint FastAPI.
- Il formato di salvataggio della sessione.
- Come disegnare le misure sul canvas e nel PDF (stile coerente con l'esistente).
- Come mostrare il suggerimento epipolare.
- I test. Ce ne devono essere almeno due:
  - un test numerico della stima da una foto su Camera_4 (scala ~4 mm/px);
  - un test della triangolazione con dati sintetici di cui si conosce il risultato.
