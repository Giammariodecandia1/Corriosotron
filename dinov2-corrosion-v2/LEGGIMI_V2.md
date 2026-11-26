# Rustron / Corrosion Intelligence

Rustron è il nome ufficiale della v2. Si avvia con **`AVVIA RUSTRON.bat`**, che si trova nella cartella principale.

## Maschera del metallo: anticipata, solo dove serve, su tutti i core

- **Parte in anticipo.** Appena la ROI e gli oggetti sono definiti, BiRefNet comincia a calcolare la maschera del metallo. Intanto puoi controllare gli oggetti e scegliere il profilo. Sotto la foto compare l'avanzamento, per esempio "riquadro 5/15"; quando la maschera è pronta, **Analizza** richiede circa 1–2 minuti, nelle prove anche solo 8 secondi.
- **Salta i riquadri inutili.** I riquadri che stanno tutti fuori dalle zone "Analizza solo qui" o tutti dentro zone escluse non vengono calcolati, perché il loro risultato verrebbe scartato. La calibrazione e "Aggiungi esempi" calcolano invece la maschera completa, perché i marker possono stare ovunque.
- **Nessun lavoro perso.** I riquadri finiti restano salvati per 3 giorni. Se il server si chiude, o cambi la scelta degli oggetti, il calcolo riparte da lì, purché la ROI sia identica.
- **Tutti i core.** BiRefNet usa tutti i core: YOLO imponeva un solo core, il limite è stato tolto.
- **Niente risultati senza maschera.** Se la maschera non è disponibile, l'analisi si ferma con un messaggio invece di dare un numero sbagliato.
- **Errori registrati.** Gli errori imprevisti finiscono in `errori_server.log`.

## Mappa del punto di scatto (opzionale)

Nel passo "Report" c'è la casella **Mappa del punto di scatto**. Se non è spuntata, il report non cambia.

Se è spuntata, il PDF aggiunge un riquadro **"Posizione dello scatto"** con:
- un estratto di OpenStreetMap e il segnaposto nel punto esatto;
- le coordinate;
- il link cliccabile **"Apri il punto in Google Maps"**.

**Dove compare:** se hai chiesto anche le anomalie numerate, il riquadro va nello spazio libero della pagina 2, sotto la tabella. Altrimenti viene aggiunta una pagina "Posizione dello scatto".

**Coordinate:** vengono lette solo dai metadati GPS della foto. Se mancano, il report lo segnala e la mappa non viene inserita.

**Internet:** per scaricare la mappa serve la connessione. Al server di OpenStreetMap vanno solo le coordinate, la foto no. I riquadri della mappa restano in `model_cache\osm_tiles`, quindi la stessa zona si scarica una volta sola. Senza connessione il report contiene comunque coordinate e link.

## Zone escluse nel report

Nella mappa di classificazione e nel PDF le zone escluse hanno due colori, spiegati in una legenda sotto la mappa:
- **blu:** oggetti esclusi da un modello YOLO;
- **azzurro chiaro:** zone escluse a mano, disegnate dall'operatore.

La riga "Ambito" del PDF riporta anche il numero di zone escluse a mano.

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

## ROI automatica (senza disegnarla)

Se nessuna ROI è disegnata a mano e almeno una classe è su **Analizza solo questo**, per esempio *Sostegni*, la ROI viene presa in automatico. È il riquadro del **sostegno principale** più gli oggetti "solo questo" che stanno per lo più dentro di lui, con un margine del 3%. Sulla foto compare tratteggiata.

- **Come sceglie il sostegno principale:** prende quello con il riquadro più grande, pesato per la confidenza, cioè di solito quello in primo piano. I tralicci sullo sfondo restano fuori dalla ROI.
- **Per cambiarlo:** tasto destro su un altro oggetto "solo questo", poi **Usa come ROI automatica**. Se togli il sostegno scelto, la ROI passa al successivo.
- **Le classi su Escludi valgono anche dentro il sostegno.** Per esempio *Etichetta su Sostegno* e *Isolatori* vengono escluse anche se stanno dentro il riquadro del sostegno.
- **La ROI disegnata a mano ha sempre la precedenza.** **Intera foto** disattiva la ROI automatica per quella foto. La casella **ROI automatica sugli oggetti «Analizza solo qui»**, nel pannello oggetti, la spegne del tutto.

Con la ROI automatica il rilevamento gira su tutta la foto, perché la ROI nasce proprio dai suoi risultati. La maschera di dettaglio BiRefNet invece si calcola solo nella ROI, quindi l'analisi resta veloce.

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
