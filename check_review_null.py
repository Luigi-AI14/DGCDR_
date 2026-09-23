import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor
def check_file(file_path):
    filename = os.path.basename(file_path)
    print(f"[{filename}] Inizio analisi...", flush=True)
    t0 = time.time()
    last_log_time = t0
    
    total_reviews = 0
    missing_key = 0
    is_none = 0
    is_empty_exact = 0
    is_whitespace_only = 0
    valid_text = 0
    json_decode_errors = 0
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            total_reviews += 1
            
            line_str = line.strip()
            if not line_str:
                # Riga completamente vuota nel file
                is_empty_exact += 1
                continue
            try:
                d = json.loads(line_str)
            except Exception:
                json_decode_errors += 1
                continue
            
            if "text" not in d:
                missing_key += 1
            else:
                t = d["text"]
                if t is None:
                    is_none += 1
                elif not isinstance(t, str):
                    # se per caso non è una stringa ma es. dizionario/lista vuota o altro
                    is_none += 1
                elif len(t) == 0:
                    is_empty_exact += 1
                elif len(t.strip()) == 0:
                    is_whitespace_only += 1
                else:
                    valid_text += 1
            
            if total_reviews % 5_000_000 == 0:
                now = time.time()
                rate = total_reviews / (now - t0)
                print(f"[{filename}] Processate {total_reviews:,} righe ({rate:,.0f} righe/s) - "
                      f"senza testo finora: {missing_key + is_none + is_empty_exact + is_whitespace_only:,}", flush=True)
    
    elapsed = time.time() - t0
    total_missing_or_blank = missing_key + is_none + is_empty_exact + is_whitespace_only
    print(f"[{filename}] COMPLETATO in {elapsed:.1f}s! Totale: {total_reviews:,} righe.", flush=True)
    
    return {
        "file": filename,
        "elapsed_seconds": elapsed,
        "total_reviews": total_reviews,
        "valid_text": valid_text,
        "missing_key": missing_key,
        "is_none": is_none,
        "is_empty_exact": is_empty_exact,
        "is_whitespace_only": is_whitespace_only,
        "total_missing_or_blank": total_missing_or_blank,
        "json_decode_errors": json_decode_errors,
    }

def main():
    folder = os.path.join(os.getcwd(), "review_metadata")
    files = [
        os.path.join(folder, "Cloth.jsonl"),
        os.path.join(folder, "Elec.jsonl")
    ]
    
    print(f"Avvio elaborazione parallela per {len(files)} file...", flush=True)
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(check_file, files))
        
    print("\n" + "="*60)
    print("RIASSUNTO RISULTATI")
    print("="*60)
    
    total_all = 0
    total_valid_all = 0
    total_missing_all = 0
    total_missing_key_all = 0
    total_is_none_all = 0
    total_empty_exact_all = 0
    total_whitespace_all = 0
    
    for r in results:
        print(f"\nFile: {r['file']}")
        print(f"  - Recensioni totali: {r['total_reviews']:,}")
        print(f"  - Con testo valido: {r['valid_text']:,}")
        print(f"  - TOTALE senza testo / testo vuoto: {r['total_missing_or_blank']:,}")
        print(f"      * Chiave 'text' assente: {r['missing_key']:,}")
        print(f"      * Valore 'text' null/None: {r['is_none']:,}")
        print(f"      * Stringa vuota ('') esatta: {r['is_empty_exact']:,}")
        print(f"      * Solo spazi bianchi ('   '): {r['is_whitespace_only']:,}")
        if r['json_decode_errors'] > 0:
            print(f"      * Errori decodifica JSON: {r['json_decode_errors']:,}")
        print(f"  - Tempo impiegato: {r['elapsed_seconds']:.1f} secondi")
        
        total_all += r['total_reviews']
        total_valid_all += r['valid_text']
        total_missing_all += r['total_missing_or_blank']
        total_missing_key_all += r['missing_key']
        total_is_none_all += r['is_none']
        total_empty_exact_all += r['is_empty_exact']
        total_whitespace_all += r['is_whitespace_only']
    print("\n" + "="*60)
    print("TOTALE COMPLESSIVO:")
    print(f"  - Recensioni totali analizzate: {total_all:,}")
    print(f"  - Con testo valido: {total_valid_all:,}")
    print(f"  - Recensioni con testo mancante/vuoto: {total_missing_all:,}")
    print(f"      * Chiave assente: {total_missing_key_all:,}")
    print(f"      * Valore null: {total_is_none_all:,}")
    print(f"      * Stringa vuota: {total_empty_exact_all:,}")
    print(f"      * Solo spazi bianchi: {total_whitespace_all:,}")
    print(f"  - Tempo totale: {time.time() - t_start:.1f} secondi")
    print("="*60)
if __name__ == "__main__":
    main()