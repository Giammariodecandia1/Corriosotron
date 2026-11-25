# DINOv2 Corrosion Intelligence · v2

La v2 è una copia separata del primo strumento (`dinov2-corrosion-poc`, che resta invariato) con in più il **riconoscimento automatico degli oggetti**. Gira su **http://127.0.0.1:8766** (la prima versione resta sulla 8765), quindi le due possono essere aperte insieme.

Avvio: doppio clic su **`AVVIA_V2.bat`**. Il file chiude un eventuale server v2 già aperto, avvia quello nuovo e apre il browser da solo. Per spegnere il server basta chiudere la sua finestra nera.

## Cosa aggiunge

Dopo il caricamento della foto e la maschera di sfondo, lo strumento esegue in automatico i modelli YOLO attivi. Per i modelli che danno solo i rettangoli, SAM ricava il contorno reale di ogni oggetto. Per ogni classe di ogni modello si sceglie cosa farne:

| Scelta | Effetto sull'analisi |
|---|---|
| **Escludi dall'analisi** | i pixel dell'oggetto vengono tolti dal conteggio della ruggine. Oggi si usa per gli isolatori, al posto dell'esclusione manuale |
| **Analizza solo questo** | l'analisi guarda solo dentro gli oggetti di questa classe (per esempio, in futuro, il sostegno) |
| **Ignora** | l'oggetto viene mostrato ma non cambia nulla |

Sulla foto, le zone azzurre sono escluse, le verdi sono quelle analizzate in "solo questo" e le grigie sono ignorate. Nella mappa di classificazione gli oggetti esclusi compaiono in azzurro, e quanto sta fuori da un "solo questo" viene scurito. L'esclusione manuale resta disponibile per correggere a mano quello che il modello non trova.

Il report riporta i dati del riconoscimento:
- **JSON:** la sezione `object_recognition`, con modelli, SAM, soglia, ogni oggetto con la sua modalità e i pixel esclusi;
- **CSV:** una riga per ogni oggetto;
- **PDF:** la riga "Ambito", per esempio "metallo visibile · esclusi 12 Isolatori".

## Correggere a mano gli oggetti riconosciuti

Prima di avviare l'analisi puoi fare **tasto destro su una zona riconosciuta** e scegliere:
- **Togli questa zona (non è …):** la zona sparisce e torna nell'analisi. Resta un rettangolo rosso tratteggiato: con un altro tasto destro lo puoi **ripristinare**.
- **Escludi / Analizza solo questo / Ignora · solo questo oggetto:** cambia la scelta solo per quell'oggetto. Il suo rettangolo diventa più spesso.
- **Torna alla scelta della classe:** annulla la scelta fatta su quell'oggetto.

Le correzioni restano registrate nel report:
- **JSON:** `operator_override` per ogni oggetto e `removed_by_operator` per il totale;
- **CSV:** la colonna `modifica_operatore`.

Un nuovo rilevamento azzera le correzioni, per esempio quando cambi la confidenza o i modelli attivi.

## Scegliere i modelli

- **Menu "Modello YOLO":** elenca i `.pt` di tutte le cartelle configurate, raggruppati per cartella. Scegli un modello e premi **Aggiungi all'analisi**.
- **Più modelli insieme:** puoi aggiungerne quanti vuoi. Il rilevamento li esegue tutti sulla stessa foto e ognuno ha le sue scelte per classe. Con **Rimuovi** togli un modello dall'analisi.
- **Memoria:** il browser ricorda i modelli attivi e le scelte fatte.
- **Cartelle:** si gestiscono in "Cartelle dei modelli" sotto il riquadro.
  - **Sfoglia cartella…** apre la finestra di Windows. Se non la vedi, può essere finita dietro il browser.
  - In alternativa puoi incollare il percorso.
  - **Togli** rimuove la cartella solo dall'elenco: i file restano dove sono.
  - L'elenco delle cartelle è salvato in `modelli_oggetti/cartelle.json`.

## Aggiungere un modello

1. Copia il file `.pt` di YOLO (rilevamento o segmentazione, Ultralytics) nella cartella `modelli_oggetti/`, oppure aggiungi la cartella in cui si trova già.
2. Facoltativo: affiancagli un file `.json` con lo stesso nome. Serve a decidere se il modello è attivo di default, quale scelta proporre per ogni classe e la risoluzione di inferenza. Per esempio:
   ```json
   {"description": "Isolatori V1 (YOLO11s, YoloTron)", "enabled": true,
    "default_modes": {"Isolatori": "exclude"}, "imgsz": 960}
   ```
   Senza questo file il modello compare spento, con tutte le classi su "Ignora", e usa la risoluzione con cui è stato addestrato.
3. Ricarica la pagina. Non serve riavviare il server.

I modelli di segmentazione (`-seg`) usano le proprie maschere, senza SAM.

## SAM

Per i contorni la v2 usa `sam2.1_b.pt`, salvato in `model_cache/sam/` e scaricato una sola volta. Il modello si cambia con la variabile d'ambiente `V2_SAM_WEIGHTS`, per esempio `mobile_sam.pt`.

Prova del 24/11/2026 sulle foto DJI di esempio, con 16 isolatori:

| Modello SAM | Tempo su CPU | Risultato |
|---|---|---|
| SAM 2.1-B | circa 7–27 s | contorni aderenti agli isolatori |
| MobileSAM | circa 3 volte più veloce | sborda sul traliccio dietro gli isolatori, quindi toglierebbe acciaio sano dall'analisi |

## Modello attuale

`Isolatori V1s.pt` viene da YoloTron: YOLO11s, 1 classe "Isolatori", imgsz 960.
