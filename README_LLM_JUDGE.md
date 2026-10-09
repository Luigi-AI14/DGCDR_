# Stato Attuale del Sistema LLM-as-a-Judge (G-Eval Alignment)

Questo documento descrive in dettaglio l'architettura, il funzionamento, gli script utilizzati e l'intera cronologia delle migliorie applicate al sistema di valutazione automatica basato su **LLM-as-a-Judge** per il progetto **DGCDR** (*Disentangled Graph Cross-Domain Recommendation*).

---

## 1. Obiettivo e Ruolo del Judge

Il modulo **LLM-as-a-Judge** ha lo scopo di valutare la qualità, la veridicità e la pertinenza delle **spiegazioni di raccomandazione** generate per gli item raccomandati dal modello cross-domain, confrontandole con le **recensioni reali** lasciate dagli utenti sugli item di test/held-out.

Il sistema adotta il paradigma formale **G-Eval** (*Liu et al., EMNLP 2023*), implementando:
1. **Valutazione Contrastiva**: confronto diretto tra le affermazioni della spiegazione e l'esperienza dichiarata dall'utente nella recensione.
2. **Chain-of-Thought (CoT)**: il modello genera prima una motivazione sintetica (1-2 frasi) e successivamente emette il voto.
3. **Calcolo della Probabilità Pesata (Logprobs)**: estrazione della distribuzione di probabilità sui token numerici `1`, `2`, `3`, `4`, `5` condizionata sul ragionamento CoT, calcolando il punteggio continuo pesato:
   $$G\text{-Eval} = \sum_{s=1}^{5} s \cdot p(S = s)$$

---

## 2. File e Script del Sistema

```text
DGCDR_/
├── llm_explainer/
│   ├── judge/
│   │   ├── test_geval_prompts.py          # Script CLI principale di esecuzione del Judge
│   │   ├── shuffle_for_judge.py           # Script dedicato per generare baseline randomizzata (shuffle) e benchmark
│   │   └── evaluate_with_judge.py         # Script storico del Judge (4 metriche legacy)
│   ├── prompts/
│   │   ├── TEST_PROMPT_ALIGNMENT.txt      # System prompt: definizione della metrica, rubrica 1-5, regole e step
│   │   └── TEST_PROMPT_INSTANCE.txt       # Instance prompt: template con placeholder {item_name}, {user_review}, {explanation}
│   └── quintiles.py                       # Modulo di classificazione delle recensioni per lunghezza (Q1-Q5)
├── results/Cloth-Elec/shuffled/           # Dataset generati con spiegazioni randomizzate (derangement negativo)
├── results_judge/                         # Output generati dal Judge (organizzati per dominio)
│   └── Cloth-Elec/
│       ├── geval_Cloth-Elec_alignment_<model>_<timestamp>.json           # Report congruente
│       ├── geval_Cloth-Elec_alignment_<model>_shuffled_<timestamp>.json  # Report baseline shuffled
│       └── benchmark_judge_congruent_vs_shuffled_<timestamp>.md          # Benchmark comparativo contrastivo
└── README_LLM_JUDGE.md                    # Questo documento di specifica e stato attuale
```

### Dettaglio dei Componenti:
* **`test_geval_prompts.py`**:
  * Interroga Ollama tramite endpoint compatibile OpenAI (`/v1/chat/completions`) con parametri deterministici (`temperature=0.0`, `seed=42`, `logprobs=True`, `top_logprobs=20`, `max_tokens=150`).
  * Esegue il filtraggio delle recensioni ultra-brevi (`< 5` parole).
  * Classifica ciascuna recensione nel relativo quintile di lunghezza (**Q1** Micro, **Q2** Short, **Q3** Medium, **Q4** Detailed, **Q5** In-Depth).
  * Stampa a terminale in tempo reale la CoT e le probabilità estratte.
  * Salva i risultati in formato strutturato JSON e in tabelle Markdown riassuntive.
* **`TEST_PROMPT_ALIGNMENT.txt`**:
  * Definisce la metrica unificata **`Alignment`** su scala Likert da 1 a 5.
  * Contiene le linee guida speciali per gestire recensioni off-topic/spedizione/spam.
  * Contiene i 5 step di valutazione generati direttamente dal modello LLM a temperatura 0.
* **`TEST_PROMPT_INSTANCE.txt`**:
  * Fornisce al modello la terna contrastiva ancorata con il nome dell'item:
    ```text
    [Item Name]:
    {item_name}

    [User Review]:
    {user_review}

    [Recommendation Explanation]:
    {explanation}

    Evaluation Form:
    Reasoning:
    ```

---

## 3. Rubrica della Metrica `Alignment` (Scala 1 - 5)

La metrica **Alignment** misura il grado di sovrapposizione e coerenza tra i claim sulle feature formulati dalla spiegazione e quanto attestato nella recensione dell'utente:

| Score | Livello | Definizione Operativa |
| :---: | :--- | :--- |
| **1** | **Hallucinated / Contradictory / Entity Mismatch** | La spiegazione contiene allucinazioni fattuali evidenti, contraddice apertamente l'esperienza dell'utente, oppure inventa feature/scenari d'uso inesistenti. Include esplicitamente il **mismatch ontologico di categoria/dispositivo**: se la recensione parla di un oggetto (es. supporto per auto, cavo, custodia) e la spiegazione descrive un dispositivo diverso/incompatibile (es. prese elettriche, RAM/SSD, soundbar), è tassativamente **Score 1** (mai Score 3). |
| **2** | **Both Review and Explanation Generic** | Sia la recensione che la spiegazione sono vaghe, superficiali e generiche. La recensione esprime solo sentiment generico senza caratteristiche tangibili, e la spiegazione fa affermazioni generiche senza ancoraggio a feature concrete. |
| **3** | **Generic Review vs. Detailed Explanation (Contesto Compatibile)** | La recensione e la spiegazione descrivono contesti di prodotto plausibilmente compatibili, ma la recensione è generica o superficiale (priva di dettagli tecnici o fisici), mentre la spiegazione fa claim dettagliati e specifici. Poiché la recensione è priva di dettagli, i claim specifici della spiegazione non possono essere corroborati. **Si applica SOLO se il tipo di prodotto è compatibile**; non si applica per dispositivi divergenti (Score 1). |
| **4** | **Partial Alignment (At Least One Feature Grounded)** | Almeno una feature concreta, attributo funzionale o claim pratico asserito nella spiegazione è direttamente menzionato e confermato nella recensione. Altri claim possono non essere verificati, ma esiste una chiara corrispondenza su almeno una feature primaria. |
| **5** | **Strong Alignment (Majority/All Claims Grounded)** | La maggioranza o la totalità delle specifiche feature, attributi e benefici evidenziati nella spiegazione sono esplicitamente presenti, verificati e allineati con quanto valutato dall'utente nella recensione. |

---

## 4. Tutte le Migliorie Applicate Fino ad Ora

### Miglioria 1: Sostituzione delle 4 Metriche Legacy con la Nuova Metrica Unificata `Alignment`
* **Problema Precedente**: Il sistema utilizzava 4 metriche separate (*Aspect Coverage*, *Aspect Precision*, *Sentiment Coherence*, *Specificity*). Oltre a richiedere 4 chiamate separate per item, soffrivano di grave **compressione dei punteggi** (oltre il 70-80% dei voti era schiacciato su 4/5) e valutavano la spiegazione in modo isolato, senza un vero confronto contrastivo con l'esperienza dell'utente.
* **Soluzione Implementata**: Rimozione totale delle 4 vecchie metriche e introduzione dell'unica metrica contrastiva **`Alignment`** con scala Likert 1-5 a livelli mutualmente esclusivi.

### Miglioria 2: Ancoraggio dell'Identità del Prodotto tramite `[Item Name]`
* **Ruolo e Funzione**: Il prompt di istanza include `[Item Name]` con il nome/titolo del prodotto raccomandato. Questo fornisce al Judge l'ancoraggio essenziale sulla categoria e sull'identità fisica dell'oggetto:
  * Consente di valutare correttamente recensioni sintetiche (es. *"Works as expected"*) collegandole al prodotto reale.
  * Consente di rilevare con certezza matematica se la spiegazione allucina un dispositivo completamente incompatibile (es. un computer desktop per un cavo USB o un supporto GPS).

### Miglioria 3: Filtraggio delle Recensioni Ultra-Brevi (`< 5` parole)
* **Problema Precedente**: Recensioni costituite da 1-4 parole (es. *"Good"*, *"Works well"*, *"Love it"*) producevano valutazioni artificiali e arbitrarie, distorcendo la media globale.
* **Soluzione Implementata**: Introdotti il parametro `--min_review_words 5` e il conteggio automatico dei token alfabetici. Le recensioni con meno di 5 parole vengono escluse a priori e censite nei report come `Filtered_UltraShort` (69 item scartati nel run completo Cloth-Elec).

### Miglioria 4: Stratificazione in Quintili di Lunghezza (`Q1 - Q5`)
* **Problema Precedente**: Impossibilità di comprendere se un punteggio basso/alto dipendesse dalla bravura del generatore o dalla scarsità di informazioni fornite dall'utente.
* **Soluzione Implementata**: Integrazione del modulo `llm_explainer/quintiles.py`. Le recensioni valide vengono suddivise in 5 quintili calcolati sulla distribuzione empirica del dominio:
  * **Q1 (Micro)**: 5 – 15 parole
  * **Q2 (Short)**: 16 – 33 parole
  * **Q3 (Medium)**: 34 – 61 parole
  * **Q4 (Detailed)**: 62 – 123 parole
  * **Q5 (In-Depth)**: $\ge$ 124 parole
  Ogni report include una tabella di breakdown con conteggi, medie discrete e medie pesate per quintile.

### Miglioria 5: Generazione Automatica degli Evaluation Steps tramite LLM
* **Problema Precedente**: Gli evaluation steps scritti manualmente dall'uomo rischiavano di contenere bias e non rispecchiavano fedelmente la metodologia originale di G-Eval.
* **Soluzione Implementata**: Gli evaluation steps sono stati generati chiamando direttamente `llama3.1:8b` (a temperatura 0.0) a partire dalla definizione della rubrica. Il testo risultante in 5 passaggi logici è stato integrato stabilmente in `TEST_PROMPT_ALIGNMENT.txt`.

### Miglioria 6: Two-Stage CoT con Streaming a Terminale (Senza Inquinamento dei File)
* **Problema Precedente**:
  * Un voto diretto senza ragionamento preliminare portava a decisioni affrettate dell'LLM.
  * Al contempo, salvare testi lunghi di reasoning per centinaia di item gonfiava a dismisura i report JSON/MD rendendoli difficili da consultare.
* **Soluzione Implementata**:
  * Strutturazione dell'output in formato a due stadi:
    ```text
    Reasoning: <1-2 sentences of contrastive analysis>
    Alignment: <1-5>
    ```
  * Aumento del buffer di output a `max_tokens=150`.
  * La CoT generata viene mostrata in streaming in tempo reale su terminale per consentire all'operatore di verificare la qualità del giudizio durante la run.
  * Il campo `reasoning` viene omesso dai file di output finali (.json e .md), mantenendo i dati snelli e focalizzati sulle metriche quantitative.
  * L'estrazione dei logprob avviene esattamente sul token numerico del voto, condizionato sull'intera catena di ragionamento antecedente.

### Miglioria 7: Gestione delle Recensioni Off-Topic, Spedizione e Spam
* **Problema Emerso dall'Analisi di Q3-Q5**: Quando una recensione parla esclusivamente di tempi di consegna, danni del corriere, feedback sul venditore o spam pubblicitario (es. codici sconto per penne), il giudice tendeva a considerare la corretta spiegazione del prodotto come "allucinazione" (Score 1).
* **Soluzione Implementata**: Aggiunta di una linea guida esplicita nel prompt di sistema:
  > *Se la recensione dell'utente si concentra esclusivamente su spedizione, consegna, imballaggio, feedback del venditore, aneddoti personali irrilevanti o spam promozionale, e non tratta feature del prodotto, NON considerare la spiegazione come allucinazione (Score 1). Assegna Score 3 (Review generica vs Spiegazione dettagliata), poiché la recensione manca dei dettagli necessari a corroborare i claim.*

### Miglioria 8: Regola dell'Incompatibilità Ontologica di Entità/Dispositivo (Entity & Category Mismatch $\to$ Score 1)
* **Problema Emerso dall'Analisi della Baseline Shuffled**: Nella valutazione delle spiegazioni randomizzate, il 48% dei mismatch veniva erroneamente classificato con Score 3 anziché Score 1. Il modello interpretava l'assenza di feature nella recensione come "mancanza di dettagli per corroborare i claim" (es. recensione di un supporto per auto a ventosa vs spiegazione di una ciabatta elettrica con prese USB riceveva Score 3 perché la recensione non citava le porte USB).
* **Soluzione Implementata**: Riformulazione di Score 1 e Score 3 in `TEST_PROMPT_ALIGNMENT.txt`:
  * **Score 1** include esplicitamente il *mismatch ontologico*: se la recensione discute un determinato oggetto/funzione e la spiegazione descrive un dispositivo diverso o incompatibile (es. RAM/SSD, ciabatte elettriche, altoparlanti), si assegna tassativamente **Score 1** (mai Score 3).
  * **Score 3** è stato ristretto ai soli casi in cui recensione e spiegazione sono *plausibilmente compatibili* per tipologia di prodotto, ma la recensione è troppo sintetica per confermare claim tecnici specifici.

---

## 5. Stato Attuale dei Risultati (Dataset `Cloth-Elec`)

Dall'ultima esecuzione completa sul dataset di test **Cloth-Elec** (200 utenti, 649 item validi, 69 ultra-brevi filtrati) condotta con `llama3.1:8b`:

### Sintesi Globale
* **Media Aritmetica Globale**: **3.3945 / 5.0**
* **Media Pesata Globale (G-Eval)**: **3.3323 / 5.0**
* **Distribuzione Globale dei Punteggi**:
  * **Score 1**: 48 item (7.4%) — *Allucinazioni reali, cross-product leakage, contraddizioni dirette*
  * **Score 2**: 43 item (6.6%) — *Vaghezza bilaterale*
  * **Score 3**: 248 item (38.2%) — *Recensione d'uso normale vs Spiegazione con specifiche non menzionate*
  * **Score 4**: 224 item (34.5%) — *Almeno una feature verificata (es. cavi resistenti, lenti, supporto)*
  * **Score 5**: 86 item (13.3%) — *Allineamento pressoché totale delle caratteristiche*

### Performance per Quintile

| Quintile | Fascia Parole | Item | Score Medio Discreto | Score Medio Pesato | Distribuzione Prevalente |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1 (Micro)** | 5 – 15w | 134 | 2.8731 | 2.7562 | Dominano Score 2 e 3 (recensioni brevissime) |
| **Q2 (Short)** | 16 – 33w | 133 | 3.3985 | 3.3283 | Transizione verso Score 3 e 4 |
| **Q3 (Medium)** | 34 – 61w | 129 | 3.3178 | 3.2067 | Picco sullo Score 3 (46.5% dei casi) |
| **Q4 (Detailed)**| 62 – 123w | 127 | 3.6693 | 3.6393 | Aumento netto di Score 4 e 5 |
| **Q5 (In-Depth)**| $\ge$ 124w | 126 | 3.7222 | 3.7380 | Massimo allineamento: dominano Score 4 e 5 |

La correlazione positiva tra la lunghezza della recensione e lo score (da 2.75 in Q1 a 3.74 in Q5) dimostra la solidità della metrica contrastiva: all'aumentare dei dettagli forniti dall'utente, il giudice è in grado di verificare un maggior numero di claim tecnici della spiegazione.

---

## 6. Modalità di Esecuzione (CLI)

Per eseguire il benchmark di Alignment con il Judge:

```powershell
# Attiva l'ambiente conda
conda activate dgcdr38

# Test rapido sui primi 5 utenti (mostra CoT in streaming su terminale)
python llm_explainer/judge/test_geval_prompts.py --max_users 5

# Esecuzione completa su tutti i 200 utenti
python llm_explainer/judge/test_geval_prompts.py
```

I file di log e report verranno generati automaticamente all'interno di:
`results_judge/Cloth-Elec/geval_Cloth-Elec_alignment_<modello>_<timestamp>.json` (.md)

---

## 7. Validazione del Judge tramite Shuffle & Negative Baseline (Discriminative Power)

Per validare formalmente che l'LLM Judge stia realmente valutando la coerenza semantica e fattuale (anziché premiare la sola forma grammaticale o soffrire di forte leniency bias), si applica il **test contrastivo di specificità (Negative / Shuffled Baseline)** tramite lo script dedicato **`shuffle_for_judge.py`**.

### 7.1. Principio Scientifico
1. **Coppie Congruenti (Positive Pairs)**: Recensione dell'Item $X$ vs Spiegazione generata per l'Item $X$. Ci si attende punteggi medio-alti (Score 3, 4, 5).
2. **Coppie Mismatched / Shuffled (Negative Baseline)**: Recensione dell'Item $X$ vs Spiegazione generata per un Item diverso $Y$ ($X \neq Y$, da un utente diverso $u_j \neq u_i$).
   * Se il Judge possiede potere discriminativo, deve identificare la mancata aderenza alle caratteristiche e assegnare **Score 1** (*Hallucinated / Contradictory Explanation*) o **Score 2** (*Generic Mismatch*).
   * L'assegnazione di Score 4 o 5 su coppie casuali deve crollare a valori prossimi allo 0%.
   * Il margine contrastivo deve essere strettamente positivo:
     $$\Delta = \bar{S}_{\text{congruent}} - \bar{S}_{\text{shuffled}} > 0 \quad (p < 0.001)$$

### 7.2. Funzionamento di `shuffle_for_judge.py`
Lo script implementa un **derangement perfetto (1-to-1 bijection)** risolto tramite algoritmo di matching bipartito ad assegnamento di costo minimo (`scipy.optimize.linear_sum_assignment`):
* **Zero Collisioni**: nessun item riceve la propria spiegazione, nessun item riceve una spiegazione dello stesso codice prodotto (`item_id`), e nessuna spiegazione proviene dallo stesso utente (`user_id`).
* **Preservazione Distribuzione**: ogni spiegazione reale del dataset viene riutilizzata esattamente una volta, mantenendo inalterata la distribuzione di lunghezze e vocabolario.
* **Metadati Completi**: il dataset generato salva nel JSON e in una card Markdown (`_info.md`) l'ID e il titolo del prodotto di provenienza della spiegazione scambiata.

### 7.3. Comandi di Esecuzione

```powershell
# Attiva l'ambiente conda
conda activate dgcdr38

# 1. Genera il dataset shuffled (default: seed=42, mode=inter_user)
python llm_explainer/judge/shuffle_for_judge.py

# 2. Genera il dataset ed esegue subito il Judge (es. test rapido su 5 utenti)
python llm_explainer/judge/shuffle_for_judge.py --run_judge --max_users 5

# 3. Esegui il Judge direttamente sul dataset shuffled
python llm_explainer/judge/test_geval_prompts.py --results_file results/Cloth-Elec/shuffled/Cloth-Elec_llama3.1_8b_shuffled_inter_user_seed42_<timestamp>.json

# 4. Confronto automatico e calcolo del Benchmark Contrastivo (Margini, Win Rate, t-test, Cohen's d)
python llm_explainer/judge/shuffle_for_judge.py --compare --auto
```

