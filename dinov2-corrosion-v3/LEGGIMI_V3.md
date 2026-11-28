# Rustron v3

La v3 è Rustron v2 più una funzione secondaria di **misura** in metri e centimetri: aree di ruggine, anomalie, linee e aree tracciate a mano. Si può stimare da una foto sola oppure misurare da 2 o più foto con tie point, come in DISTIX.

- **Avvio:** doppio clic su `AVVIA RUSTRON V3.bat` nella cartella principale, oppure su `AVVIA_V3.bat` in questa cartella.
- **Indirizzo:** http://127.0.0.1:8767
- La **v2** (porta 8766, `AVVIA RUSTRON.bat`) e la **poc** (porta 8765) restano invariate e si possono usare in parallelo.
- La v3 ha un suo ambiente Python (`.venv`, `.python`, `model_cache`), copiato dalla v2 e indipendente da essa. In più ha `scipy`, che serve alla misura da più foto.
- Per tutto ciò che non riguarda la misura vale `LEGGIMI_V2.md`.

## Come si misura

1. Fai l'analisi come nella v2. Se un modello YOLO ha il sostegno in modalità «Usa come ROI», il riquadro del sostegno serve anche alla misura.
2. Apri la mappa (**⤢ Ingrandisci**) e premi **📐 Misura**. Si apre il pannello a destra. Se non lo accendi, Rustron v3 si comporta esattamente come la v2.
3. Il pannello mostra:
   - **Foto e drone:** drone riconosciuto, RTK, distanza dal sostegno (telemetro laser oppure gimbal + quota), scala al centro, esito.
   - **Misure automatiche:** area vista di ruggine media, ruggine intensa e zone grigie in m², superficie reale stimata e dimensioni delle anomalie A1, A2… (se hai attivato «Clusterizza anomalie»).
   - **Misure a mano:** con **Linea** o **Area** fai clic sui vertici e doppio clic per finire. **Esc** annulla, **Canc** elimina la misura selezionata.
   - **Misura precisa:** apre la sessione con più foto.
4. Spunta **PDF** accanto alle misure da mettere nel report e premi **PDF** nella colonna dei risultati. Il report avrà una pagina «Misure» in più. Se non spunti niente, il PDF resta quello di sempre.

Se la stima non è possibile (base del sostegno fuori dalla foto, nessun riquadro del sostegno), il pannello lo dice. Puoi cliccare tu la base del sostegno oppure aprire la sessione con più foto.

## Sessione con più foto (come DISTIX)

1. **📐 Misura → ＋ Apri sessione multi-foto.** La foto analizzata è la principale (★).
2. **＋ Foto:** aggiungi una o più foto dello **stesso sostegno**, scattate da posizioni diverse: almeno 15° di differenza di direzione. Va bene anche girandoci intorno.
3. Clicca i **punti richiesti** nella foto principale e nelle altre:
   - T1 e T2: piedi dei montanti alla stessa quota;
   - T3 e T4: nodo superiore della stessa faccia;
   - con «＋ punto» puoi aggiungere altri nodi ben riconoscibili.

   Dopo il primo clic, nell'altra foto compare una linea tratteggiata: il punto deve trovarsi lì sopra. Alt+clic cancella un punto da una foto.
4. **Calcola.** Rustron affina gli assetti delle foto (bundle adjustment DISTIX), ricostruisce i punti in 3D e da lì la faccia del sostegno. Le misure del pannello vengono ricalcolate su quella faccia.
5. La sessione si salva da sola, in `misure/`. Riaprendo la stessa foto principale la ritrovi.

## STIMA, VALIDATA, NON VALIDATA

Ogni numero porta valore ± U95, cioè l'incertezza estesa con k = 2 (circa il 95 % di probabilità), e un'etichetta:

| Etichetta | Significato |
| --- | --- |
| **STIMA** | Misura da una foto sola. Il sostegno è trattato come un piano verticale per la sua base. La distanza viene dal telemetro laser se valido, altrimenti dagli angoli del gimbal e dalla quota sul punto di decollo. L'incertezza tiene conto di gimbal, focale, dislivello del terreno (±3 m) e inclinazione della faccia. |
| **VALIDATA** | Sessione con più foto che supera tutti i controlli DISTIX: RTK FIX su tutte le foto, convergenza ≥ 15°, riproiezione entro soglia, almeno 4 tie point visti in ≥ 2 foto, camera calibrata. |
| **NON VALIDATA** | Sessione con più foto in cui almeno un controllo non passa. Le misure ci sono comunque, con la loro incertezza, ma non sono validate. Il motivo è scritto nel pannello e nel PDF. |

**Perché le M3T e le H30T escono NON VALIDATA:** senza un profilo di calibrazione certificato della camera, la focale viene ricavata dall'equivalente 35 mm scritto nella foto. Per ottenere VALIDATA va inserita la calibrazione nel profilo del drone (vedi sotto).

## Area vista e superficie reale

- **Area vista:** ruggine visibile in foto, proiettata sulla faccia del sostegno. È quello che si misura.
- **Superficie reale stimata:** area vista × **fattore di sviluppo** dei profilati, cioè l'acciaio totale diviso la parte visibile in foto. Per un angolare a L vale circa 3: un'ala di larghezza a sviluppa circa 4a di acciaio per metro e se ne vede circa 1,2–1,4a. Le parti nascoste si suppongono arrugginite come quelle visibili. È **sempre una STIMA**.
- Il fattore si cambia nel pannello (predefinito 3,0 ± 1,0) e si salva con le **Impostazioni**.

## Droni

Rustron riconosce il drone dai metadati della foto (marca e modello) e sceglie il profilo in `profili_drone/`. Ci sono già DJI M3T e DJI H30T. Per aggiungerne uno vedi `profili_drone/LEGGIMI.md`. Un drone senza profilo funziona lo stesso, con un'incertezza più larga.

Il telemetro laser si usa solo se è valido e puntato sul sostegno. Sulle M3T vale spesso 0,000 m e viene ignorato.

## Limiti

- **Niente zoom:** una foto con zoom digitale, o con focale diversa più del 5 % da quella del profilo, è «non misurabile».
- **Niente foto termiche**, e niente foto ruotate: va usato l'originale del drone.
- Senza metadati di posizione e gimbal (foto non DJI o metadati tolti) la foto non si misura.
- La superficie reale è sempre stimata. Per misurarla davvero servirebbe un modello 3D completo da molte foto, che non fa parte di questa versione.

## Test

- **Calcoli e server:**

  ```
  .venv\Scripts\python.exe -m pytest tests -q
  ```

  9 test vecchi della v2 (`test_sampling`, `test_smoke`) falliscono anche nella v2: non riguardano la misura.
- **Interfaccia** (avvia da solo un server di prova sulla porta 8799 ed Edge senza finestra):

  ```
  .venv\Scripts\python.exe tests\check_ui_misura.py --sessione
  ```
