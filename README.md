# IBM RACE 2026 - Team Nexus 🏎️

Benvenuti nella repository ufficiale del **Team Nexus**. Questo progetto è dedicato allo sviluppo e all'ottimizzazione di un'auto a guida autonoma per la competizione **IBM RACE 2026**, con un focus specifico sulle prestazioni nel circuito **Corkscrew Track**.

## 🏁 Obiettivi del Progetto
*   **Standing Start:** Ottimizzazione della partenza da fermo per massimizzare la trazione iniziale.
*   **Fastest Lap:** Algoritmi di guida di precisione per affrontare le sfide tecniche del Corkscrew.
*   **AI Integration:** Utilizzo dei modelli **IBM Granite** per lo sviluppo del software di controllo.

## 🛠️ Requisiti e Ambiente (Conda)
Per garantire la riproducibilità dei risultati, utilizziamo un ambiente Conda dedicato chiamato `torcs-env`.

### Installazione
Se sei un membro del team o un giudice, puoi replicare l'ambiente di sviluppo locale utilizzando il file `environment.yml` incluso:

```bash
# Crea l'ambiente dalla configurazione
conda env create -f environment.yml

# Attiva l'ambiente
conda activate torcs-env
```

## 📂 Struttura della Repository
*   `src/`: Contiene la logica dell'intelligenza artificiale e il controller dell'auto.
*   `livery/`: File grafici della livrea proprietaria per la competizione.
*   `docs/`: Documentazione strategica, log dei test e report sull'uso di IBM Granite.
*   `environment.yml`: Specifiche dei pacchetti e delle versioni dell'ambiente `torcs-env`.

## 🤖 Integrazione IBM Granite
In conformità con il regolamento, il Team Nexus utilizza **IBM Granite Models for Software Development** per:
1.  Generazione di algoritmi per il calcolo della traiettoria ideale.
2.  Refactoring e debugging del controller.
3.  Ottimizzazione delle performance computazionali del codice Python.

## 👥 Il Team
*   **Team Name:** Nexus
*   **Competizione:** IBM RACE 2026
*   **Scadenza Sottomissione:** 1 Luglio 2026

---
*Progetto realizzato nell'ambito della IBM RACE 2026. Tutti i membri del team sono certificati IBM Granite Models for Software Development.*
