# DeMeTrA quick-start: correzioni generali richieste

## Obiettivo

Rendere il quick start riproducibile da un clone pubblico pulito, fino alla produzione del CSV e del video di tracking, senza dipendere da file o configurazioni presenti solo sulla macchina di sviluppo.

## Correzioni bloccanti

1. **Riparare la distribuzione Git LFS.** Il clone pubblico fallisce perché gli oggetti LFS di `all_manos_CL.csv` e/o `all_manos_CL_pixel.csv` non sono disponibili sul server. Ripristinare gli oggetti oppure rimuovere dal repository i riferimenti LFS non necessari. Verificare con un clone anonimo in una directory vuota.

2. **Versionare `ffmpeg_utils.py`.** `scripts/predict_firstpass_and_track_from_folder.py` importa `resolve_ffmpeg_executable` da un modulo assente dal repository. Aggiungere il modulo e un test che importi realmente l'entrypoint.

3. **Correggere le dipendenze runtime.** `IPython` è importato dal percorso di inferenza ma non è dichiarato. Preferibilmente eliminare l'import notebook dal percorso runtime o renderlo opzionale; in alternativa aggiungere una versione compatibile ai requirements.

4. **Separare installazione CPU e CUDA.** Il file principale non deve imporre CUDA 11.3 e Triton a tutti gli utenti. Fornire un insieme comune di dipendenze e istruzioni esplicite per installare PyTorch CPU oppure la variante CUDA appropriata. `pip install -r requirements.txt` deve funzionare nel percorso documentato.

5. **Eliminare i path locali hard-coded.** I default `/media/isacDisk2/...` e `/home/isac/...` devono diventare path relativi al repository/output o valori derivati da `sys.executable`. Tutti i path ricevuti dalla CLI devono essere risolti prima di avviare subprocess.

6. **Propagare i parametri di risorse.** `download_and_track_range.py` deve esporre e inoltrare almeno batch size e worker di first-pass e tracking. I default devono essere conservativi; in modalità CPU usare batch 1 e pochi/zero worker, lasciando all'utente la possibilità di aumentarli.

7. **Pubblicare checkpoint di sola inferenza.** Il checkpoint tracking da 3,65 GB contiene modello, copia dello state dict e stato optimizer. Pubblicare un artefatto `model-only` portabile, senza optimizer, RNG state, oggetti `Path` o altri oggetti Python non necessari. Aggiornare URL, dimensione e SHA-256 nel README.

## Qualità e documentazione

- Documentare chiaramente: credenziali EUMETSAT obbligatorie per marzo 2026, spazio disco, download aggiuntivo del backbone X3D-M, requisiti RAM e tempi CPU indicativi.
- Aggiungere un comando di smoke test con un piccolo dataset incluso e senza credenziali esterne.
- Correggere il test manifest: le immagini demo sono 384×384, mentre il test costruisce il dataset aspettandosi 512×512.
- Aggiungere CI su clone pulito che esegua almeno: installazione CPU, import degli entrypoint, `--help`, test automatici, caricamento dei checkpoint e una piccola inferenza.
- Fare fallire subito il quick start con messaggi chiari se mancano credenziali, checkpoint, FFmpeg o spazio/memoria, prima di scaricare dataset voluminosi.

## Criteri di accettazione

Il lavoro è concluso quando, partendo da una directory vuota:

1. il clone anonimo termina senza errori;
2. l'installazione CPU documentata termina senza modifiche manuali ai requirements;
3. entrambi gli entrypoint si importano e `--help` funziona;
4. lo smoke test incluso produce il CSV finale;
5. il comando del README arriva al controllo/download EUMETSAT senza errori locali;
6. con credenziali valide e i checkpoint pubblici produce la traccia richiesta.
