# Valutazione per rilievi completi e analisi batch

Data: 7 novembre 2026. Revisione del codice locale, delle quattro immagini
fornite in chat e dei metadati originali letti dalla condivisione autorizzata.
Nessuna fotografia inviata a servizi esterni. Nessuna nuova inferenza sulle
fotografie aziendali in questa revisione; non sono state create etichette di
corrosione presunte. Il motore applicativo non e' stato modificato.

## Conclusione

La PoC dimostra l'esecuzione CPU di DINOv2 e il flusso interattivo. Il sistema
attuale non e' ancora un analizzatore automatico affidabile di interi rilievi.
Il percorso raccomandato mantiene DINOv2 come estrattore reale, aggiunge
localizzazione del sostegno, analisi locale ad alta risoluzione, segmentazione
supervisionata e una coda persistente. Prima di realizzare il batch occorre
misurare la capacita' di riconoscimento su esempi reali annotati.

## Fotografie e dettaglio

Tutte e quattro: 4000x3000, 7,11-7,55 MiB, compatibili con i limiti correnti
18 MiB / 40 MP. Osservazione sulle anteprime: profilati sottili, numerose
occlusioni e sfondi urbani/vegetazione; tetti e muri introducono colori e
texture potenzialmente confondibili. Le ultime tre viste sembrano appartenere
allo stesso sostegno/scena: verificare l'identita' asset prima dello split dati.
Non e' possibile assegnare con affidabilita' tre livelli di corrosione da queste
anteprime. Occorrono dettagli a risoluzione nativa e giudizio esperto.

Con l'attuale letterbox 4000x3000 -> 518x389 dentro un quadrato 518x518, una
patch da 14 pixel copre circa 108x108 pixel originali. Solo circa 28 righe
della griglia 37x37 intersecano la foto. A 392, circa 143 pixel originali per
patch. Un profilato largo 20 pixel originali diventa circa 2,6 pixel a 518:
e' un esempio geometrico, non una misura fatta sui profilati allegati.
Il ricampionamento dell'overlay fino a 1100 pixel non ricrea il dettaglio perso.
Cambiare liberamente patch_size non e' una regolazione compatibile del
checkpoint preaddestrato. Ritagli e scala d'ingresso sono le prime variabili
da sperimentare, mantenendo le caratteristiche del modello.

## Lacune confermate nel codice

| Priorita' | Evidenza | Intervento necessario |
|---|---|---|
| Alta | main.py train accetta etichette diverse nella medesima patch; riprodotto con vettori sintetici, senza caricare un modello | Rifiutare conflitti e deduplicare punti per patch; mostrare la patch all'operatore |
| Alta | classify usa argmax su tutte le posizioni, senza classe non valutabile | Separare qualita', metallo e corrosione; introdurre astensione e soglie validate |
| Alta | Un singolo prototipo medio per classe, un solo local_profile.json sovrascritto | Dataset multi-immagine e profili versionati; congelare profilo/modello per ogni job |
| Alta | Embedding e classificazione sincroni negli endpoint async | Worker dedicato e coda persistente, ripresa dopo errore e limiti di memoria |
| Alta | UI lascia file, ROI e risoluzione modificabili durante la richiesta; show accetta sempre la risposta | Identificare richiesta/foto/configurazione e scartare risposte obsolete; bloccare i controlli necessari |
| Alta | ID della maschera dipendono dalle classi presenti; class_id_labels e' fuori dal report esportato JSON | ID stabili e legenda/metadati inclusi in ogni pacchetto export |
| Media | Validazione profilo con assert, vettori nulli ammessi, nessun versionamento dello schema | Controlli espliciti, norme non nulle, compatibilita' di modello e preprocessing |
| Media | Modello senza revisione fissata e dipendenze ad intervalli ampi | Registrare revisione pesi, versioni software, parametri, identificazione input e tempi completi |
| Alta per batch | Nessuna coda, stato job, storage risultati, aggregazione asset o regola alert | Implementare dopo la validazione della pipeline visiva |

La formula corrosione/metallo classificato e' algebricamente corretta. Non
garantisce che il denominatore sia metallo reale: su elementi sottili anche
pochi falsi pixel di sfondo possono cambiare molto il risultato. Con nessun
metallo, distinguere non valutabile da assenza di corrosione. La media 1-3
non e' una misura di danno strutturale e puo' nascondere piccole zone intense.

Il test con immagini diverse ma dello stesso sostegno verifica trasferimento
locale; non dimostra generalizzazione a sostegni, siti e condizioni nuovi.
I due test unitari persistenti restano insufficienti. Le prove HTTP precedenti
non sono ancora una suite di regressione end-to-end ripetibile.

## Architettura raccomandata

1. Ingestione di una cartella di rilievo, con input originali in sola lettura.
   Manifest con asset/missione, hash, dimensioni, qualita', duplicati e stato.
   Ignorare file ancora in copia e deduplicare le riesecuzioni.
2. Localizzare il sostegno sull'immagine completa; una bounding box seleziona
   i ritagli ma NON e' la superficie metallica usata nel denominatore.
3. Analizzare ritagli sovrapposti in coordinate originali. Scegliere scala e
   sovrapposizione confrontando immagini annotate; evitare migliaia di ritagli
   su tetti e vegetazione. Valutare anche gli errori di esclusione del detector.
4. DINOv2 congelato + classificatore/decoder supervisionato da confrontare con
   gli attuali prototipi. Separare metallo, sfondo/accessori e non valutabile;
   poi sana/corrosione ed eventuali livelli visivi sostenuti dalle annotazioni.
   Un modello ausiliario per localizzare il sostegno e' una proposta, non e'
   presente oggi; i suoi costi CPU vanno inclusi nel benchmark.
5. Ricomporre le predizioni sulla foto originale, fondendo punteggi nelle
   sovrapposizioni e contando ciascun pixel una sola volta. Conservare mappa,
   legenda, ROI/ritagli, immagine sorgente identificata e parametri.
6. Worker CPU persistente (inizialmente uno, modello caricato una sola volta),
   coda SQLite locale e output per job. Gestire pause, riavvii, errori, annullamento
   e backpressure; memoria limitata ai file in lavorazione. Non caricare tutte
   le fotografie in RAM e non aumentare i worker prima di misurare contesa CPU.
7. Report per foto e riepilogo per sostegno; alert interni alla dashboard a
   completamento di ogni immagine. Invii esterni richiedono canali autorizzati.

Gli alert iniziali devono segnalare candidati da revisionare: sospetta corrosione,
piccola zona visivamente intensa oppure foto non valutabile. Definire soglie e
costo dei falsi allarmi con Eagle Projects; non usare il margine coseno come
probabilita' calibrata. Evitare la media semplice fra foto dello stesso sostegno:
le superfici possono essere viste piu' volte. Non ricavare una percentuale 3D
globale senza geometria e copertura esplicite. Il report per asset deve mostrare
evidenze, viste e parti non osservate, senza moltiplicare lo stesso alert.

## Prestazioni: stime, non benchmark batch

Il benchmark esistente misura due forward su immagine uniforme per taglia.
Contrariamente alla frase in BENCHMARK.md, il timer parte DOPO il processor:
esclude preprocessing, I/O, export, coda e caricamento modello. Il picco RSS e'
campionato dopo il caricamento, non e' il picco dell'intero job. Non e' una
misurazione p95 su rilievi.

Usando 1,6 s per forward 518 come semplice ipotesi seriale:

| Passaggi 518 per foto | Solo modello per foto | Solo modello su 300 foto |
|---:|---:|---:|
| 1 | 1,6 s | 8 minuti |
| 10 | 16 s | 80 minuti |
| 20 | 32 s | 160 minuti |
| 70 | 112 s | 9 ore 20 minuti |

Settanta finestre corrispondono circa a coprire 4000x3000 con tile 518 e passo
414 (circa 20% overlap): 10 colonne e 7 righe. Dieci/venti sono scenari
ipotetici di selezione delle zone, non conteggi misurati sulle foto fornite.
Latenze vere includeranno tutte le fasi e dipenderanno dal carico della VM.
Obiettivo realistico da concordare: primi esiti progressivi e una finestra
temporale del lotto, non completamento istantaneo di centinaia di foto.

## Percorso con verifiche prima di procedere

A. **Correttezza minima**: conflitti patch, risposte obsolete, legenda export,
   profili e tracciabilita'. Test di regressione salvati; nessun nuovo modulo batch.
B. **Esperimento su dati reali**: piccolo lotto rappresentativo, ad esempio
   30-50 foto per impostare il protocollo (non dimensione sufficiente garantita
   per validazione). Includere corrosione confermata, sano, casi dubbi, ombra,
   distanza, superfici/accessori diversi e piu' sostegni. Annotazioni di metallo,
   corrosione e non valutabile, con criteri visivi condivisi.
C. **Confronto a parita' di dati**: immagine ridotta, ROI manuali, tile automatici.
   Confrontare prototipi e piccolo decoder addestrato su feature DINOv2. Dividere
   training/calibrazione/test per sostegno e preferibilmente missione/sito;
   tenere intatto un test finale. Se manca dettaglio anche nel nativo, richiedere
   viste piu' ravvicinate: ritagliare o interpolare non inventa informazione.
D. **Scelta pipeline**: misurare recall difetti, falsi alert per 100 foto,
   precisione/recall o Dice/IoU metallo e corrosione, errore della percentuale,
   copertura valutabile, prestazioni per classe e dimensione dei difetti. Misurare
   disaccordo esperti sui tre livelli prima di automatizzarli. Stabilire soglie
   d'accettazione prima di guardare il test finale.
E. **Batch pilota**: manifest, coda, worker, ripresa, report e alert progressivi
   su un rilievo completo. Misurare throughput reale, p50/p95, picco RAM,
   errori/recuperi e duplicazione alert. Poi collaudo su 100-300 foto.

Il prossimo risultato utile e' un confronto originale/maschera/manuale su un
piccolo lotto annotato, con tempi e falsi allarmi. La validazione applicativa
precede qualsiasi promessa operativa su centinaia di fotografie.

## Fonti tecniche

- https://huggingface.co/facebook/dinov2-small/blob/main/config.json
- https://github.com/facebookresearch/dinov2
