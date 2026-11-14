# GEAR-BCI-Motor-Imagery

# Confronto di 7 modelli adattivi per BCI motor imagery (RWOS-ELM)

Codice MATLAB della tesi di laurea di Samuele Pinello, "Development of an unsupervised, self-adaptive BCI
system for motor intention recognition", Università di Padova, 2025/2026.

Lo script confronta 7 varianti di classificatore adattivo (RWOS-ELM) per una BCI a due classi (Motor Imagery vs Rest), valutate in modalità pseudo-online su sessioni registrate in momenti diversi.

## Contenuto del repository

| File | Descrizione |
|---|---|
| `Confronto_7_modelli.m` | Script unico: training, simulazione online, statistiche, export e grafici. Le funzioni di supporto sono in fondo allo stesso file. |


## Dati attesi

Lo script si aspetta file EEGLAB `.set` organizzati così:

```
<base_path>/
└── sub-XX/
    └── ses-N/
        └── eeg/
            └── sub-XX_<nome_sessione>_run-R_eeg.set
```

- `XX`: numero del soggetto a due cifre (soggetti usati: 1, 2, 4, 5, 7, 9, 10, 12, 13, 14, 15, 25)
- Sessioni: `ses-1` (`pre`), `ses-4` (`post`), `ses-5` (`follow`)
- Run: 1–4 per ogni sessione
- Eventi usati: trigger `3` (o `Target_MI`) = Motor Imagery, `9` (o `Target_Rest`) = Rest



Il seed è fisso (`rng(42)`), quindi i risultati sono riproducibili.

## Pipeline

1. **Training (Ses-1):** primi 70% dei trial in ordine cronologico per il training, restanti 30% per il test offline.
2. **Feature:** per ogni trial, matrice di covarianza calcolata sull'**intero trial**, cioè una finestra fissa di 5 s dall'onset del cue (1251 campioni a 250 Hz). La covarianza è calcolata separatamente per 4 bande (8–12, 12–16, 16–22, 22–30 Hz) sui 5 canali selezionati (indici 10, 17, 18, 19, 40) e riunita in una matrice a blocchi 20×20. Le feature sono la proiezione nel tangent space riemanniano rispetto alla media di Karcher del training.
3. **Classificatore:** ELM con 400 neuroni nascosti (`C = 0.1`), aggiornato in modo ricorsivo con forgetting (`lambda = 0.98`, peso del campione `w = 0.6`).
4. **Test online (Ses-4 e Ses-5):** simulazione trial dopo trial. I primi 20 trial di ogni sessione sono usati per la calibrazione e i restanti sono valutati. Ogni trial produce una singola predizione. Il flusso è causale ma sincrono (una decisione per trial a fine finestra), non un controllo asincrono.

## Modelli confrontati

| Modello | Descrizione |
|---|---|
| M1 | Static RWOS: nessun aggiornamento |
| M2 | RWOS con pseudo-label: aggiorna sempre con la propria predizione |
| M3 | RWOS-ED: aggiorna solo se la predizione concorda con il gate euclideo |
| M4 | RWOS-Riemann senza rebias: gate riemanniano |
| M5 | RWOS-Riemann base: gate riemanniano con rebias statico sui 20 trial di calibrazione |
| M6 | RWOS-LW-Rebias: covarianza con shrinkage di Ledoit-Wolf e rebias statico |
| M7 | GEAR: rebias adattivo con media riemanniana aggiornata online |

## Metriche e statistica

- Per soggetto e sessione: accuracy, balanced accuracy, sensibilità, specificità, F1, AUROC, percentuale di trial predetti come Rest, numero di aggiornamenti e loro correttezza (precisione del gate).
- Soglia di chance per soggetto (test binomiale unilaterale, α = 0.05).
- Test inferenziali: Friedman (con W di Kendall) e confronti a coppie Wilcoxon con correzione FDR, su accuracy (analisi principale) e su specificità, sensibilità e AUROC (esplorativi).

## Output

I file vengono salvati in `out_dir`:

- `console_output.txt`: log completo della console
- `risultati_per_soggetto.csv`: metriche per soggetto, sessione e modello
- `predizioni_grezze.csv`: predizioni, punteggi e flag di update per ogni trial
- `risultati_completi.mat`: tutte le matrici dei risultati

Lo script genera anche i grafici usati nella tesi.

## Dati

I dati grezzi e processati sono archiviati su Drive: https://drive.google.com/drive/folders/1JNLw4Y-qAUun2f1JV9DbGn2mFVeJ1hPW?usp=sharing

Dataset di origine: https://springernature.figshare.com/articles/dataset/Lower_limb_motor_imagery_EEG_dataset_based_on_the_multi-paradigm_and_longitudinal-training_of_stroke_patients/27130299
