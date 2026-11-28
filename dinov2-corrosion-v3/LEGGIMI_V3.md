# Rustron v3

La v3 è Rustron v2 più una funzione secondaria di **misura** in metri e centimetri: aree di ruggine, anomalie, linee e aree tracciate a mano. Le misure si ricavano da **2 o più foto dello stesso sostegno** con tie point, come in DISTIX.

- **Avvio:** doppio clic su `AVVIA RUSTRON V3.bat` nella cartella principale, oppure su `AVVIA_V3.bat` in questa cartella.
- **Indirizzo:** http://127.0.0.1:8767
- La **v2** (porta 8766, `AVVIA RUSTRON.bat`) e la **poc** (porta 8765) restano invariate e si possono usare in parallelo.
- La v3 ha un suo ambiente Python (`.venv`, `.python`, `model_cache`), copiato dalla v2 e indipendente da essa. In più ha `scipy`, che serve alla misura da più foto.
- Per tutto ciò che non riguarda la misura vale `LEGGIMI_V2.md`.

## Perché servono almeno 2 foto

Una prima versione stimava le misure da una foto sola, supponendo la faccia del sostegno di fronte alla camera. Sul traliccio 23341A1-074 (Codrongianos) la faccia era ruotata di 45–50° rispetto alla visuale e la stima sbagliava del 20–30 %, oltre l'incertezza dichiarata. Con due foto da direzioni diverse la faccia si ricostruisce davvero: sullo stesso traliccio la distanza ricavata coincide con il telemetro laser (46,3 m contro 46,3 m e 64,0 m contro 65,3 m). La stima da una foto sola è stata tolta.

## Come si misura

1. Fai l'analisi come nella v2.
2. Apri la mappa (**⤢ Ingrandisci**) e premi **📐 Misura**. Si apre il pannello a destra. Se non lo accendi, Rustron v3 si comporta esattamente come la v2.
3. **＋ Apri sessione multi-foto.** La foto analizzata è la principale (★).
4. **＋ Foto:** aggiungi una o più foto dello **stesso sostegno**, scattate da direzioni diverse (almeno 15°; vanno bene anche lati opposti). Rustron rifiuta, dicendo il motivo, le foto senza metadati, quelle senza quota assoluta, quelle scattate a più di 600 m dalla principale (altro sostegno) e i doppioni.
5. Clicca gli **stessi punti fisici** nella foto principale e nelle altre (T1, T2, … almeno 4). Vanno bene punti nitidi e riconoscibili da ogni lato: la punta del cimino, le punte delle mensole, i piedi dei montanti, i nodi principali. Dopo il primo clic, nell'altra foto compare una linea tratteggiata: il punto deve trovarsi lì sopra. Alt+clic cancella un punto da una foto; con «＋ punto» ne aggiungi altri.
6. **Calcola.** Rustron affina gli assetti delle foto (bundle adjustment DISTIX), ricostruisce i punti in 3D e da lì la faccia del sostegno. I clic incoerenti vengono esclusi e segnalati.
7. Il pannello mostra l'esito e le misure:
   - **Misure automatiche:** area vista di ruggine media, ruggine intensa e zone grigie, superficie reale stimata e dimensioni delle anomalie A1, A2… (se hai attivato «Clusterizza anomalie»).
   - **Misure a mano:** con **Linea** o **Area** fai clic sui vertici e doppio clic per finire. **Esc** annulla, **Canc** elimina la misura selezionata. Si possono disegnare anche prima del calcolo: il valore compare dopo «Calcola».
8. Spunta **PDF** accanto alle misure da mettere nel report e premi **PDF** nella colonna dei risultati. Il report avrà una pagina «Misure» in più. Se non spunti niente, il PDF resta quello di sempre.

La sessione si salva da sola, in `misure/`. Riaprendo la stessa foto principale la ritrovi.

**Consiglio:** la faccia è ricostruita dai tie point; più una misura è lontana da loro, più l'incertezza cresce. Per misurare la parte bassa del sostegno aggiungi tie point anche in basso (piedi dei montanti, nodi del tronco).

## VALIDATA e NON VALIDATA

Ogni numero porta valore ± U95, cioè l'intervallo che contiene il valore con circa il 95 % di probabilità, e un'etichetta:

| Etichetta | Significato |
| --- | --- |
| **VALIDATA** | La sessione supera tutti i controlli DISTIX: RTK FIX su tutte le foto, convergenza ≥ 15°, riproiezione entro soglia, almeno 4 tie point visti in ≥ 2 foto, camera calibrata. |
| **NON VALIDATA** | Almeno un controllo non passa. Le misure ci sono comunque, con la loro incertezza, ma non sono validate. Il motivo è scritto nel pannello e nel PDF. |
| **INCOMPLETA** | Mancano punti o foto: nessuna misura, il pannello dice cosa manca. |

**Perché le foto dei droni senza RTK o senza calibrazione escono NON VALIDATA:** senza RTK la posizione del drone è incerta di qualche metro; senza un profilo di calibrazione certificato la focale viene ricavata dall'equivalente 35 mm scritto nella foto. Per ottenere VALIDATA servono RTK FIX e la calibrazione nel profilo del drone (vedi sotto).

## Area vista e superficie reale

- **Area vista:** ruggine visibile in foto, proiettata sulla faccia del sostegno. È quello che si misura.
- **Superficie reale stimata:** area vista × **fattore di sviluppo** dei profilati, cioè l'acciaio totale diviso la parte visibile in foto. Per un angolare a L vale circa 3: un'ala di larghezza a sviluppa circa 4a di acciaio per metro e se ne vede circa 1,2–1,4a. Le parti nascoste si suppongono arrugginite come quelle visibili. È **sempre una STIMA**.
- Il fattore si cambia nel pannello (predefinito 3,0 ± 1,0) e si salva con le **Impostazioni**.
- Se una parte della zona non cade sulla faccia del sostegno (sopra l'orizzonte o troppo lontana), la misura è segnata «parziale» con la percentuale non misurata.

## Droni

Rustron riconosce il drone dai metadati della foto (marca e modello) e sceglie il profilo in `profili_drone/`. Ci sono già DJI M3T e DJI H30T. Per aggiungerne uno vedi `profili_drone/LEGGIMI.md`. Un drone senza profilo (per esempio la H20T) funziona lo stesso con i dati standard della foto, marcato «profilo generico».

## Limiti

- **Niente zoom:** una foto con zoom digitale, o con focale diversa più del 5 % da quella del profilo, è «non misurabile».
- **Niente foto termiche**, e niente foto ruotate o ritagliate: va usato l'originale del drone.
- Senza metadati di posizione, quota assoluta e gimbal (foto non DJI o metadati tolti) la foto non si misura.
- La superficie reale è sempre stimata. Per misurarla davvero servirebbe un modello 3D completo da molte foto, che non fa parte di questa versione.

## Test

- **Calcoli e server:**

  ```
  .venv\Scripts\python.exe -m pytest tests -q
  ```

  9 test vecchi della v2 (`test_sampling`, `test_smoke`) falliscono anche nella v2: non riguardano la misura. Il test sul traliccio reale 23341A1-074 parte solo se la cartella di rete `\\192.168.199.10\Elaborazioni\…\Codrongianos\20231130` è raggiungibile.
- **Interfaccia** (avvia da solo un server di prova sulla porta 8799 ed Edge senza finestra; usa le foto A e B del traliccio 23341A1-074 dalla cartella di rete):

  ```
  .venv\Scripts\python.exe tests\check_ui_misura.py
  ```
