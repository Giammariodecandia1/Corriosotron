# Profili drone

Ogni file `.json` di questa cartella descrive una camera di drone. Rustron sceglie il profilo confrontando **Make** e **Model** scritti nella foto (EXIF) con i campi `make` e `model` del file. Le maiuscole non contano.

## Aggiungere un drone

1. Copia un file esistente, ad esempio `dji_m3t.json`, e rinominalo, ad esempio `dji_m4t.json`.
2. Cambia questi campi:
   - `id`: uguale al nome del file;
   - `nome`: il nome che compare nel pannello Misura;
   - `make` e `model`: come compaiono nella foto (in Windows: Proprietà → Dettagli → Produttore / Modello fotocamera);
   - `focal_mm`: la focale **vera** della camera grandangolare, cioè il valore «Lunghezza focale» della foto, non l'equivalente 35 mm;
   - `sensor_w_mm`, `native_w`, `native_h`: larghezza del sensore in mm e risoluzione nativa in pixel;
   - `has_rtk`, `has_lrf`, `lrf_range_m`: se c'è l'RTK, se c'è il telemetro laser e la sua portata utile in metri;
   - `sigma_gimbal_deg`, `sigma_focal_rel`: incertezze tipiche (1 σ) degli angoli del gimbal e della focale. Se non le conosci, lascia 1,0° e 0,02.
3. Riavvia Rustron v3.

## Calibrazione

Se hai una calibrazione certificata della camera, inseriscila in `calibrazione`:

```json
{ "image_width": 8000, "image_height": 6000, "fx": 0, "fy": 0, "cx": 0, "cy": 0,
  "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0 }
```

Servono **tutti** i valori, compresi i coefficienti di distorsione (scrivi 0 se il certificato li dà nulli). Rustron li usa per raddrizzare i raggi. Una calibrazione incompleta viene usata per la focale, ma la camera non conta come calibrata.

Con la calibrazione la sessione a più foto può risultare **VALIDATA**; senza resta **NON VALIDATA**. Non inserire valori stimati.

## Drone senza profilo

Se una foto non corrisponde a nessun profilo, Rustron usa solo i dati standard della foto (focale ed equivalente 35 mm). Il risultato esce marcato «profilo generico», con un'incertezza più larga.
