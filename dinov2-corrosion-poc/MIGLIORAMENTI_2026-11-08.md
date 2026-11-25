# Miglioramenti e ricerca GitHub — 8 novembre 2026

## Diagnosi confermata

La precedente calibrazione conserva solo una media per classe; gli esempi
individuali non sono recuperabili dai profili esistenti. La ROI 866×2807 a
518 pixel contiene circa 11×37 patch utili. Una patch mescola aste, vernice,
sfondo e altri componenti. La maschera di primo piano ritaglia il risultato
ma non elimina lo sfondo dalle caratteristiche già estratte.

## Interventi implementati

- `feature_sampling.py`: scansione allineata alle patch, sovrapposizione ~25%,
  fusione pesata dei vettori, normalizzazione L2. Lato nominale sorgente 784 px,
  input 392 o 518; nessun ingrandimento del contenuto piccolo. Elaborazione di
  un riquadro alla volta, massimo 64 riquadri; oltre il limite si richiede una ROI
  più piccola, senza ridurre la qualità di nascosto.
- Classificatore nuovo: media delle fino a tre similarità coseno più alte per
  ciascuna classe. Conserva modalità visive distinte e non assegna più peso a una
  classe semplicemente perché contiene più esempi. Non è una replica del metodo
  HIL coral e non ne eredita risultati o garanzie.
- Profilo schema 2 con banca di esempi, provenienza numerica e configurazione
  di estrazione. Fusione idempotente, rifiuto dei conflitti e aggiornamento
  atomico con verifica contro modifiche concorrenti. Nessuna migrazione fittizia
  delle vecchie medie; schema 1 resta utilizzabile con metodo precedente.
- Report JSON e risposta API con modalità, dimensioni griglia, passo originale,
  conteggio riquadri, metodo classificatore, numero foto e pixel ambigui.
  La maschera classi mantiene ID 0–4; l'ambiguità è un PNG separato 0/255.
- La UI mostra ambiguità sul metallo, classi non calibrate, configurazione dei
  profili e avvisi. Corretto il pulsante aggiunta esempi dopo un marker e il
  blocco dei selettori durante richieste. Invalidazione dei risultati mantenuta.
- Inferenza in thread per lasciare libero il ciclo HTTP durante i forward.
  Avvio controlla anche la data dei moduli Python, non solo `main.py`.

## Prove eseguite

1. **14 test software passati**, con vettori sintetici: coordinate, copertura e
   fusione dei riquadri, rappresentazione multimodale, nomi profili, aggiunta da
   una seconda foto, duplicati/confitti, conteggi, percentuali e PNG ambiguità.
2. **HTTP con DINOv2 reale su CPU**, due immagini sintetiche 1000×1200:
   calibrazione, riuso prima dell'aggiunta, aggiunta esempi seconda immagine e
   rifiuto reinvio. 4 riquadri, griglia 48×57×384, circa 6 s di forward per
   richiesta. PNG decodificati, conteggi e percentuali JSON/CSV confrontati.
   Profilo di test eliminato; profili utente non modificati.
3. **Foto reale fornita e ROI del suo report**, offline. Verificata identità
   tramite SHA256, CPU, embedding 384 e assenza di NaN/Inf. Stessa maschera e
   stesse 11 esclusioni. Risultato originale riprodotto esattamente: 47,44%.
4. **Browser**: caricamento di immagine sintetica, BiRefNet automatico, ROI
   trascinata, marker delle tre classi, analisi temporanea reale, controlli
   bloccati durante analisi, vista ambiguità, classi mancanti indicate,
   invalidazione al cambio risoluzione e Ricomincia. Nessun errore di analisi.

| Estrazione sulla ROI reale | Griglia vettori | Pixel originali/patch | Forward CPU | Corrosione* | Ambigui/metallo* |
|---|---:|---:|---:|---:|---:|
| Precedente | 37×37 con padding | ~76 | 1,53 s | 47,44% | 22,1% |
| Riquadri sovrapposti | 41×133 | ~21 | 14,64 s | 41,95% | 31,1% |

\* Prova di sensibilità con le **stesse vecchie medie**, apprese a un'altra
scala. Non è una conversione supportata del profilo e NON dimostra un guadagno
di accuratezza: le false assegnazioni persistono. La pipeline normale usa nei
nuovi profili la stessa estrazione in calibrazione e inferenza. Non sono state
create etichette sulla foto reale né salvati nuovi profili aziendali.

Artefatti locali esclusi da Git: `model_cache/verification/real-http.json`,
`real-photo-comparison.json`, overlay comparativi e `browser-test.png`.
Script riproducibili: `tests/check_real_workflow.py` e
`tests/compare_real_sampling.py` (argomenti foto/report espliciti; nessun percorso
aziendale incorporato nel codice). Il secondo non modifica profili e non deve
essere interpretato come validazione di corrosione.

## Ricerca di soluzioni esistenti

Consultati repository originali e ricerca del connettore GitHub. Nessun codice
esterno copiato, nessuna nuova dipendenza o peso installato in questa fase.

| Progetto | Evidenza disponibile | Utilità e limite per noi |
|---|---|---|
| [beric7/corrosion_cs_classification](https://github.com/beric7/corrosion_cs_classification) | DeepLabV3+, dataset annotato, link a pesi addestrati, codice MIT | Candidato prioritario per confronto specialistico su ritagli. Classi Good/Fair/Poor/Severe su ponti: non equivalgono automaticamente ai tre livelli visivi AT. Pesi non scaricati/testati; il resolver DOI non era leggibile dallo strumento web. |
| [StuvX/SpotRust](https://github.com/StuvX/SpotRust) | HRNet e varianti bayesiane, link a pesi, metodi di incertezza | Corrosione specifica; costi CPU da misurare, inferenze multiple nella variante Monte Carlo. Dataset originale non distribuito per restrizioni industriali. |
| [sgraine/HIL-coral-segmentation](https://github.com/sgraine/HIL-coral-segmentation) | DINOv2 denoised e KNN da annotazioni sparse, codice GPL-3.0 | Conferma una strada tecnica per propagare marker. Dominio coralli, stack originale con faiss-gpu: non integrazione diretta sulla VM CPU. |
| [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | Segmentazione primo piano ad alta risoluzione, zoo di pesi e ONNX | Già utilizzato. Non distingue per definizione metallo sano, corrosione, cavi e isolatori. |

[DINOv2 ufficiale](https://github.com/facebookresearch/dinov2) fornisce
caratteristiche generali e strumenti per compiti successivi; non un modello
già validato per corrosione dei sostegni. Un rilevatore Grounding DINO/SAM può
aiutare a localizzare componenti, ma non è lo stesso modello DINOv2 e non
risolve da solo i vuoti reticolari e l'intensità della corrosione.

## Decisione e lavoro ancora necessario

La PoC resta DINOv2 reale. Usare nuovi profili Dettaglio con esempi distribuiti
su vernice, ombra, illuminazione e corrosione identificabile. Non assegnare tre
livelli se non sono presenti e riconoscibili. Almeno tre campioni distinti per
classe sono consigliati, senza presentarli come soglia di validazione.

Per decidere se il nuovo classificatore è migliore servono annotazioni esperte
su foto **non usate nella calibrazione**, separate per sostegno/rilievo (non
semplicemente scatti vicini dello stesso sostegno). Confrontare precision/recall
corrosione, Dice, falsi positivi su aste sane e MAE percentuale; selezionare soglie
su un insieme diverso dal test finale. Le aree non leggibili vanno annotate
come non valutabili, non come sane.

Solo dopo questo confronto ha senso scegliere tra DINOv2 con classificatore
supervisionato, eventuale fine-tuning e modello specialistico esterno. Quest'ultimo
sarebbe un confronto esplicito, non una sostituzione nascosta.

Restano aperti: segmentazione del solo metallo, esclusione automatica dei
componenti, calibrazione dell'incertezza, accuratezza reale, coda batch con
limiti RAM/ripresa, aggregazione per sostegno senza doppio conteggio fra viste.
I tempi misurati NON includono BiRefNet; centinaia di foto non sono istantanee
su questa CPU. Nessuna soglia di alert strutturale è stata introdotta.
