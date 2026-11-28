# Audit iniziale — 6 novembre 2026

## Provenienza e perimetro

Letti tutti i 12 file di `dinov2-corrosion-poc` nel repository privato
`Giammariodecandia1/Giammario`, branch `main`, commit
`784ce4c901927050ea33bfdf2096427ab227af06`.
Copia locale recuperata tramite connettore GitHub; non è un clone Git.
Nessuna modifica remota effettuata. Nessuna istruzione AGENTS.md applicabile
nel percorso remoto esaminato. Altre cartelle del repository non modificate.

## Ambiente rilevato

- Host: XBRASILE.
- Sistema: Microsoft Windows 11 Enterprise multi-session, build 26200.
- CPU virtuale QEMU: 8 core e 8 processori logici esposti.
- RAM: 31,93 GiB totali; 22,41 GiB disponibili al controllo.
- Disco C: circa 826,13 GiB disponibili.
- Git: 2.54.0.windows.1.
- Python non disponibile nel PATH; `py -0p` non rileva installazioni.
  Nessun interprete trovato nei percorsi standard controllati; non è stata
  eseguita una ricerca esaustiva di tutti i dischi.
- HTTPS HEAD dalla VM: risposta 200 da Hugging Face (config modello), PyPI,
  indice PyTorch CPU e Python.org. Download dei grandi pesi ancora da provare.
- Git HTTPS: autenticazione fallita. Connettore GitHub: lettura consentita,
  repository privato e permesso push dichiarato; nessuna scrittura testata.
- Inventario GPU non eseguito: il progetto forza esplicitamente la CPU.

## Audit statico

### Architettura e modello

Backend FastAPI con frontend HTML/CSS/JS locale. `AutoModel` carica realmente
`facebook/dinov2-small` con safetensors, CPU, eval e inference_mode.
Non ci sono risultati AI simulati nel percorso applicativo.
Il modello è un estrattore di caratteristiche generico: la segmentazione
dipende dai prototipi etichettati dall'operatore, non da un addestramento
specifico sulla corrosione.

La configurazione ufficiale indica patch 14 e hidden size 384. L'esclusione
del token CLS e il reshape sono coerenti: 392 -> 28x28 patch;
518 -> 37x37 patch. Ridimensionamento con padding, coordinate dei punti e
rimozione del padding sono geometricamente coerenti alla lettura.
Restano da verificare su CPU forme, valori finiti, norme e coordinate.

### Maschere e misure

La formula è `100 * (rust1 + rust2 + rust3) / (healthy + rust1 + rust2 + rust3)`.
Lo sfondo è escluso dal denominatore. Nessun metallo predetto produce null.
Le percentuali riguardano pixel della ROI, dopo ricampionamento fino a 1100 px,
non area fisica o superficie totale del sostegno. L'interpolazione dei punteggi
non crea dettaglio reale oltre la risoluzione delle patch.

L'intensità è la media pesata degli indici 1, 2, 3 sui pixel corrosi. Formula
coerente con una scala visiva operatore, senza validità metrologica dimostrata.
È permesso calibrare una sola classe di ruggine: in tal caso non esiste una
vera discriminazione su tre livelli e questo va esplicitato nei risultati.
La bassa differenza fra similarità è un indicatore euristico, non probabilità.

### Problemi da correggere prima della demo

1. Frontend: cambiando foto o ROI i risultati precedenti rimangono visibili
   ed esportabili. Durante l'inferenza alcuni controlli restano modificabili.
   Occorre invalidare risultati obsoleti e associare report/foto/ROI.
2. Output: PNG disponibile solo come overlay, non come maschera numerica delle
   classi con legenda e conteggi riproducibili.
3. Calibrazione: annotazioni di classi diverse sulla stessa patch sono accettate;
   duplicati pesano più volte. Serve rilevamento dei conflitti e conteggio patch.
4. Profilo: controllo tramite assert disattivabile con Python -O; vettori nulli
   accettati, nessuna versione di schema o revisione esatta del checkpoint.
5. Modello: revisione non fissata. Il preloader stampa soltanto la forma,
   senza assert su device, forma attesa, valori finiti o norme non nulle.
6. Dipendenze: intervalli ampi e torch non vincolato negli installer;
   requirements.txt da solo non installa torch. Compatibilità esecutiva e
   riproducibilità non ancora accertate.
7. Prestazioni: inferenza sincrona dentro endpoint async blocca l'event loop;
   possibile indisponibilità dello stato durante il calcolo. Il tempo esclude
   caricamento iniziale e parte del preprocessing. Nessun benchmark effettuato.
8. Test: due soli test con vettori costruiti, nessun test HTTP, persistenza,
   riuso su seconda foto o controllo di percentuali con valori attesi esatti.

## Sequenza proposta

1. Predisporre runtime Python dedicato al progetto senza cambiare PATH globale,
   associazioni, servizi o installazioni esistenti; creare `.venv`.
2. Correggere per piccoli interventi installer, gestione modello e smoke test;
   selezionare e registrare le versioni effettivamente compatibili.
3. Correggere calibrazione, validazione profilo, export e stato dell'interfaccia;
   rendere il backend modulare mantenendo la UI dark e l'ascolto su loopback.
4. Scaricare checkpoint reale nella cache locale, fissare revisione e verificare
   embedding CPU a 392 e 518 con forma, finitezza e norme; misurare tempi/RAM.
5. Eseguire test numerici, API, export e riuso del profilo. Poi prova manuale
   con due foto reali autorizzate, elaborate solo sulla VM.
6. Valutare accuratezza separatamente su foto indipendenti con maschere esperte;
   registrare metriche e limiti senza dichiarare validazione strutturale.
7. Sincronizzare solo codice e documentazione del progetto tramite GitHub,
   escludendo foto, profili, cache, runtime e pesi.

## Stato delle evidenze

Audit statico completato. Benchmark e inferenza CPU reale eseguiti. Nessuna foto
reale presente nel sottoalbero del progetto. Accuratezza della corrosione NON
validata.

## Collaudo funzionale 7 novembre 2026

Identificati due processi `main.py`: il listener apparteneva all'interprete
portatile `.python`; riavviato Uvicorn dalla `.venv` e verificato startup senza
errori. Workflow HTTP con due PNG sintetici differenti e DINOv2 reale: calibra
200, analizza con profilo 200, modello caricato su CPU e profilo presente.
Campi API verificati: report, overlay PNG, class-mask PNG, ID-etichetta e CSV.
PNG decodificati a 700x700; conteggi dei cinque ID coincidono con il CSV. Le
percentuali classi ruggine sono coerenti col JSON e sommano al totale corrosione
entro arrotondamento. Nessuna accuratezza su immagini reali dedotta.

UI controllata nel browser: pagina caricata e modello indicato come caricato.
Risultato invalidato su cambio foto, ROI, risoluzione, punti e classe operatore.
Test manuale nel browser di selezione punti/ROI e click download non completato.

Riferimenti ufficiali:
- https://huggingface.co/facebook/dinov2-small/blob/main/config.json
- https://huggingface.co/docs/transformers/model_doc/dinov2

## Aggiornamento 8 novembre 2026 — dettaglio e calibrazione multipla

Implementata estrazione DINOv2 CPU a riquadri sovrapposti, banca di campioni
schema 2, provenienza delle patch e deduplicazione tra invii. Profili precedenti
conservati. Corretto il calcolo UI dell'ambiguità (denominatore: metallo),
segnalate classi non calibrate e bloccati i selettori durante l'elaborazione.
14 test software passati e workflow HTTP con pesi reali su due immagini
sintetiche. Provati nel browser caricamento, BiRefNet, ROI, marker, analisi
temporanea, mappa ambiguità e invalidazione risultati.

La foto reale fornita è stata letta dalla condivisione autorizzata e analizzata
localmente; overlay diagnostici sono ora presenti nella cache esclusa da Git.
Questa evidenza aggiorna la precedente nota sull'assenza di immagini reali.
La qualità del vecchio profilo resta insufficiente: maggiore risoluzione non
risolve da sola la classificazione. Nessuna validazione dell'accuratezza.
Dettagli, tempi, limiti e fonti: `MIGLIORAMENTI_2026-11-08.md`.
