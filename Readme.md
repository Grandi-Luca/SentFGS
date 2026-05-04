# SentFGS

SentFGS è un progetto per generare riassunti più fattuali con una selezione iterativa delle frasi candidate. A ogni passo il sistema genera $k$ frasi, le valuta in base alla fattualità, sceglie la migliore e riparte da quella frase.

## Panoramica

Il flusso di lavoro è semplice:

1. generazione di $k$ frasi candidate
2. ranking delle frasi in base alla fattualità
3. selezione della frase migliore
4. ripartenza dal passo 1 usando la frase selezionata

Il repository è organizzato in due aree principali:

- `server/`: pipeline SPRING, script di inferenza e training.
- `client/`: utility, metriche e componenti di supporto.

Il progetto è pensato per essere eseguito con Docker, tramite i due servizi definiti in `docker-compose.yml`.

## Funzionalità principali

- generazione iterativa di frasi per la sintesi
- ranking delle candidate per fattualità
- supporto a metriche AMR e metriche di generazione
- supporto a checkpoint e configurazioni per esperimenti

## Requisiti

- Docker
- Docker Compose
- GPU NVIDIA consigliata per i modelli e le configurazioni già presenti nel compose

## Avvio rapido

Per avviare i container:

```bash
docker compose up --build
```

Il file `docker-compose.yml` avvia:

- il servizio `server`, che esegue `python spring/converterToAMR_server.py`
- il servizio `client`, che esegue `./settings.sh`

## Struttura del repository

- `server/spring/`: codice del modello SPRING e script di training e inferenza
- `server/spring/configs/`: configurazioni degli esperimenti
- `server/spring/bin/`: utility per training, predizione e valutazione
- `client/generation/`: funzioni per generazione e metriche
- `client/weisfeiler_leman_amr_metrics/`: metriche basate su Weisfeiler-Lehman
- `client/sema/`: implementazione AMR e strumenti SEMA
- `client/amr-utils/`: utility per parsing e confronto di grafi AMR

## Uso del progetto

Per dettagli operativi su training, predizione e valutazione, consulta:

- `server/spring/README.md`
- i commenti nei file di configurazione in `server/spring/configs/`

## Licenza

Questo repository include componenti con licenze diverse; vedi `LICENSE.md` alla radice e i file `LICENSE` nelle sottocartelle per i dettagli specifici.