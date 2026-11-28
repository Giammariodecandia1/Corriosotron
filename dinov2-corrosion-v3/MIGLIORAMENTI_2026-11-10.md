# Falsi positivi e ruggine non rilevata — 10 novembre 2026

## Cause misurate (foto DJI_20260219 e DJI_20260223, profilo «Profilo 3», 117 patch marcate)

1. **Maschera del metallo grossolana.** BiRefNet riduce l'intera foto 4000×3000 a
   circa 1024 px: il traliccio diventa una macchia che comprende terreno, vegetazione
   e veicoli visibili attraverso il reticolo e perde i profilati sottili. Lo stesso
   modello su riquadri della ROI a risoluzione nativa separa i singoli profilati.
2. **Le patch DINOv2 non vedono la ruggine dei profilati.** Una patch copre ~21×21 px
   originali, un profilato è largo 10–25 px: nelle patch marcate come ruggine il
   metallo occupa il 40–50 % della patch. Leave-one-out sulle patch marcate:

   | Criterio | AUC ruggine/sano | Ruggine trovata | Sano dato per ruggine |
   |---|---:|---:|---:|
   | DINOv2, media top-3 coseno (prima) | 0,87 | 49 % | 5 % |
   | Colore Lab del metallo, maschera nativa | 0,99 | 89 % | 1 % |

   Calibrando su una foto e provando sull'altra: colore 100 % / 62 % di ruggine
   trovata (falsi 3 % / 0 %), DINOv2 50 % / 7 %. Sommare DINOv2 al colore, o usarlo
   per decidere i casi dubbi, non ha migliorato nulla su questi dati.
3. **Astensione asimmetrica**: la regola «ruggine − sano < 0,025 → zona grigia»
   scartava circa il 70 % dei pixel inizialmente assegnati a ruggine.

## Cosa è cambiato

- `segmenta_riquadri.py` + `main.make_fine_mask`: maschera del metallo sulla ROI a
  riquadri nativi, in cache in `model_cache/fine_masks` (2–3 minuti la prima volta
  per foto/ROI, poi immediata). Se fallisce si torna alla maschera grossolana con avviso.
- `color_evidence.py`: ogni marker salva il colore Lab del metallo sotto il click
  (`color_lab` nei campioni). La decisione ruggine/sano è per pixel, sulla sola
  cromaticità (a*, b*) rispetto ai 3 esempi più vicini di ogni classe; media/intensa
  usa anche la luminosità. Colori lontani da ogni esempio o quasi neri → zona grigia.
  La luminosità è esclusa dalla decisione ruggine/sano perché, con qualsiasi peso,
  le facce in ombra di un traliccio bianco sano finivano in «ruggine intensa».
- DINOv2 resta calcolato e salvato; nel report compare come contesto
  (`dino_context`), non decide più. Profili senza colori usano ancora il metodo
  precedente, con avviso.
- Mappa classi e overlay fino a 2200 px (prima 1100). Bordo di 1 px della maschera
  escluso: toglie i pixel misti e gran parte dei cavi.
- `aggiorna_profilo_colore.py CARTELLA_FOTO`: recupera i colori dei profili
  esistenti dalle foto originali (copia `.json.bak`). Eseguito su «Profilo 3» e «Profilo1».

## Limiti, da non leggere come validazione

- I numeri sopra vengono dalle stesse due foto usate per calibrare e da etichette a
  livello di patch: indicano che il criterio è quello giusto, non l'accuratezza su
  altri sostegni. Servono foto annotate non usate in calibrazione.
- Zone grigie alte (23–27 % del metallo nelle due foto): quasi tutte facce in ombra
  mai marcate. Marcare metallo sano in ombra le riduce.
- Il colore non distingue ruggine da oggetti color ruggine dentro la maschera
  (isolatori con cappe ossidate, vernici beige/arancio): usare le esclusioni.
- Bilanciamento del bianco diverso tra rilievi sposta i colori: aggiungere esempi
  da ogni nuovo rilievo con «Aggiungi esempi al profilo».
- 9 test preesistenti in `tests/test_smoke.py` e `tests/test_sampling.py` fallivano
  già prima (usano le vecchie classi `background`/`rust1-3`); non toccati.
  Nuovi test: `tests/test_color.py` (7, passano).
