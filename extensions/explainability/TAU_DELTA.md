# DGCDR, τ e δ

Guida breve. Tutti i concetti e i meccanismi descritti di seguito sono tratti direttamente dal codice di questa repository.

---

## 1. Che cos'è DGCDR

DGCDR è un sistema di raccomandazione **cross-domain**: sfrutta le informazioni di due domini diversi (per esempio, Film e Libri) per migliorare la qualità delle raccomandazioni nel secondo dominio, definito *target*. L'assunto di base è che alcune preferenze di un utente siano valide e coerenti in entrambi i domini.

Per sfruttare questa dinamica, DGCDR scompone le preferenze di ogni utente in due componenti distinte:

- **shared** (`e^c`) — le preferenze comuni a entrambi i domini;
- **specific** (`e^s`) — le preferenze esclusive del dominio target.

A queste due componenti se ne aggiunge una terza: l'embedding di base, generato dalla propagazione sul grafo utente-item, che chiameremo **base** (`e^g`).

---

## 2. Come si genera una raccomandazione

Il processo si divide in tre fasi principali:

**1. Propagazione sul grafo.** La funzione `graph_layer` ([dgcdr.py:228](../../recbole_cdr/model/cross_domain_recommender/dgcdr.py#L228)) propaga le interazioni per un numero di passi pari a `n_layers`, producendo così l'embedding di base `e^g` sia per gli utenti che per gli item.

**2. Separazione (Disentanglement).** La funzione `disentangle_layer` ([dgcdr.py:271](../../recbole_cdr/model/cross_domain_recommender/dgcdr.py#L271)) filtra l'embedding `e^g` utilizzando due gate appresi durante l'addestramento, separandolo così nelle componenti `e^c` ed `e^s`.

**3. Fusione.** La funzione `fuse_and_update` ([dgcdr.py:236](../../recbole_cdr/model/cross_domain_recommender/dgcdr.py#L236)) ricombina le componenti utilizzando dei pesi di attention:

```python
att = torch.cat((b_1, b_2), dim=1)
softed_att = F.softmax(att, dim=1)
e_c = c_1 * common_preference
e_s = c_2 * specific_preference
user_all_embeddings = user_embeddings + e_c + e_s      # attention_mode='all'
```

In termini matematici, l'embedding finale dell'utente si ottiene così:

$$e_u = e^g_u + a_c\,e^c_u + a_s\,e^s_u$$

Questa stessa logica si applica anche agli item. Infine, gli item vengono ordinati in base al loro punteggio decrescente e si selezionano i primi *k*.

---

## 3. Come si calcola il punteggio

Il calcolo avviene tramite un semplice prodotto scalare ([dgcdr.py:609](../../recbole_cdr/model/cross_domain_recommender/dgcdr.py#L609)):

```python
scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
```

$$s(u,i) = \langle e_u,\; e_i \rangle$$

**È proprio questo dettaglio a rendere possibile il resto del ragionamento.** Poiché la fusione avviene tramite una *somma* e il punteggio si calcola tramite un *prodotto scalare*, possiamo applicare la proprietà distributiva:

$$s(u,i) = \underbrace{\langle e^g_u, e_i\rangle}_{C_{base}} + \underbrace{\langle a_c e^c_u, e_i\rangle}_{C_{shared}} + \underbrace{\langle a_s e^s_u, e_i\rangle}_{C_{specific}}$$

In questo modo, il punteggio complessivo si scompone **esattamente** nella somma dei contributi forniti dai tre canali. Non stiamo parlando di un'approssimazione o dell'uso di un modello surrogato: questa è la pura espressione aritmetica del modello stesso.
La funzione `verify_decomposition` ([channels.py:317](channels.py)) si occupa di verificare questa uguaglianza ad ogni esecuzione, bloccando il processo qualora la somma delle parti non coincidesse con il totale.

È importante notare che questi contributi mantengono il **segno**: di conseguenza, un singolo canale può anche far diminuire il punteggio finale.

---

## 4. τ — Il peso sul punteggio

**Che cos'è.** Questo parametro indica in che percentuale il punteggio di *una singola raccomandazione* sia determinato dal canale shared piuttosto che dal canale specific.

**Formula** ([attribution.py:134](attribution.py)):

$$\tau = \frac{|C_{shared}|}{|C_{shared}| + |C_{specific}|} \in [0,1]$$

```python
denom = shared + specific
tau = shared / denom if denom > 0 else 0.0
```

Si utilizzano i valori assoluti perché, come visto, i contributi possono essere negativi: una somma algebrica rischierebbe di azzerare il denominatore, rendendo il rapporto incalcolabile.

**A cosa serve.** Un valore di τ prossimo a 1 indica che la raccomandazione è guidata prevalentemente dal canale shared; se è prossimo a 0, la spinta deriva dal canale specific. Attenzione: τ è definita **sulla singola raccomandazione** e non valuta l'intero spazio latente, a differenza delle metriche tradizionali di disentanglement.

Quando è necessario aggregare i valori su più raccomandazioni, non si fa la media dei singoli rapporti, bensì si sommano separatamente tutti i numeratori e tutti i denominatori ([attribution.py:151](attribution.py)):

```python
numerator = sum(abs(a.user_channel_totals.get(SHARED, 0.0)) for a in attributions)
denominator = sum(a.magnitude for a in attributions)
```

---

## 5. δ — L'impatto sulla decisione

**Che cos'è.** Questo parametro ci dice se e quanto un determinato canale (es. *shared* o *specific*) ha realmente influenzato l'ordinamento finale delle raccomandazioni, e non solo il punteggio numerico.

**Il problema che risolve.** Il punteggio in sé non costituisce la decisione finale; la decisione vera e propria è l'**ordinamento** degli item. Se un canale aggiungesse la stessa quantità al punteggio di tutti gli item, il punteggio complessivo aumenterebbe, ma la classifica resterebbe immutata. Quello che conta davvero, quindi, sono le **differenze relative tra gli item**.

**Formula.** Considerando un item rilevante `i⁺` e uno non rilevante `i⁻`, la differenza di punteggio (il margine) può essere scomposta esattamente come il punteggio stesso:

$$\Delta C_k = C_k(u,i^+) - C_k(u,i^-), \qquad s(u,i^+)-s(u,i^-) = \sum_k \Delta C_k$$

$$\delta_k = \frac{\sum |\Delta C_k|}{\sum_j \sum |\Delta C_j|} \in [0,1]$$

Ecco come avviene il calcolo dei margini nel codice ([discrimination.py:132](discrimination.py)):

```python
return {name: scores[kept].unsqueeze(1) - scores[picked].unsqueeze(0)
        for name, scores in channel_scores.items()}
```

e la conseguente determinazione delle quote ([discrimination.py:245](discrimination.py)):

```python
'delta': absolute[name] / total_abs if total_abs else None,
'delta_disentangled': absolute[name] / disentangled_total,
```

**Attenzione al denominatore.** Il parametro `delta` viene normalizzato considerando tutti e tre i canali, mentre τ considera solamente i canali shared e specific. Per questo motivo, **solo `delta_disentangled` può essere confrontato direttamente con τ.**

**A cosa serve.** Se un canale presenta una τ alta ma una δ bassa, significa che il suo impatto è prevalentemente numerico: gonfia il punteggio ma non influisce sull'ordinamento finale e, di conseguenza, sulla decisione. Questa è un'informazione preziosa che la sola attribuzione del punteggio non poteva rivelare.

Questi parametri vanno interpretati insieme ad altre due metriche fornite dal modulo:

| Metrica | Significato |
|---|---|
| `vote_accuracy` | Con che frequenza il canale "vota" correttamente (il caso base è il 50%) |
| `signed_share` | Indica se l'apporto del canale è d'aiuto o se penalizza il risultato — **può assumere valori negativi** |
| `Spearman` fra canali | Misura se due canali tendono a ordinare gli item nello stesso modo |

Quest'ultima metrica in particolare serve a distinguere un canale **ridondante** da uno semplicemente **debole**: il parametro δ, misurando solo una proporzione, non è in grado di fare questa distinzione da solo.

---

## 6. Integrazione in un framework di explainability

Ci sono quattro motivi principali per cui τ e δ si prestano particolarmente bene a essere integrati in un sistema di explainer; tutti derivano dalla natura esatta della decomposizione.

**1. Esattezza garantita.** Sistemi di explainability post-hoc come LIME o SHAP addestrano un modello surrogato e ne spiegano il comportamento; non c'è garanzia assoluta che il surrogato rifletta fedelmente il modello originale. Nel nostro caso, non si usa alcun surrogato: si ripercorre l'aritmetica reale del modello.

**2. Autoverifica costante.** La funzione `verify_decomposition` ricompone i contributi dei canali e li confronta con l'output reale ad ogni iterazione. Se non è possibile ricostruire esattamente il risultato del modello, la spiegazione non viene neppure generata.

**3. Rifiuto esplicito.** Se si imposta `fuse_mode='concat'`, i canali passano attraverso un MLP (Multi-Layer Perceptron), rendendo la somma non più separabile in modo lineare. Invece di restituire un'approssimazione dubbia, il codice **solleva direttamente un'eccezione**. Lo stesso principio si applica in fase di addestramento (`train()`), dove l'uso del dropout renderebbe la decomposizione non deterministica.

**4. Granularità a livello di singola raccomandazione.** Poiché τ e δ sono definiti sulla specifica coppia (utente, item), è possibile generare spiegazioni personalizzate per l'utente, piuttosto che limitarsi a fornire una tabella riassuntiva globale.

### Flusso di lavoro

```
decomposizione  →  attribuzione del peso (τ)  →  verifica dell'impatto sulla decisione (δ)  →  report
```

Il framework è **interamente numerico**: non genera testo e non dipende da alcun modello linguistico. L'ultimo passaggio produce un report JSON e una tabella Markdown per singola raccomandazione, con i contributi dei canali, τ e δ. Su quel materiale può poggiare un eventuale livello di presentazione, che resta fuori dal framework.

### Cosa comunicare all'utente

- τ alta e δ alta → *"Questo item ti è stato consigliato in base a interessi che hai mostrato in entrambi i domini"* (vero sia numericamente che a livello decisionale).
- τ alta ma δ bassa → Il canale gonfia il punteggio ma non determina la decisione finale: in questo caso, usare la frase precedente sarebbe **fuorviante**.
- τ bassa → *"Questo item ti è stato consigliato basandosi su gusti specifici che hai mostrato in questo dominio"*.

**Un avvertimento importante.** Un valore alto di τ **non** significa che la raccomandazione sia necessariamente migliore. Stiamo misurando un'attribuzione delle cause, non la qualità del risultato.

---

## 7. Comandi Utili

```bash
python -m extensions.explainability.test_decomposition -m saved/<ckpt>.pth
python run_discrimination.py -m saved/<ckpt>.pth -n 300
```

Il primo comando funge da validazione: deve superare tutti i test (**9/9**), altrimenti τ e δ verrebbero calcolati su una decomposizione che non rispecchia fedelmente l'output del modello.

**Vincoli da rispettare:** è necessario che sia impostato `fuse_mode='attention'`, e il parametro τ è calcolabile soltanto per quegli utenti che hanno interazioni (e quindi esistono) in entrambi i domini.
