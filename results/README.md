# Sačuvani rezultati

Ovaj direktorijum sadrži samo male, tekstualne sažetke završenog eksperimenta koji su pogodni za Git. Veliki checkpointi, Grad-CAM slike, preprocessing cache i ostali generisani artefakti namerno nisu deo repozitorijuma.

- `final_baseline/metrics.json` — zbirne OOF metrike i bootstrap intervali.
- `final_baseline/fold_metrics.csv` — rezultati pet spoljašnjih foldova.
- `final_baseline/calibration_evaluation.json` — evaluacija kalibracije.
- `dataset_summary.csv` — sažetak korišćenog skupa.
- `verification/static_checks.json` — sažetak statičkih provera izvornog projekta.

Završno testiranje na izvornom macOS okruženju imalo je 119 uspešnih testova i 0 grešaka. Sirovi logovi nisu uključeni jer sadrže lokalne korisničke i privremene putanje.

Modeli se moraju ponovo istrenirati ili odvojeno preuzeti iz privatnog skladišta. Ovi fajlovi nisu klinička validacija.

Patient-level redovi, tokeni, datumi pregleda, putanje i pojedinačne predikcije namjerno nisu objavljeni. Zbirni `metrics.json` bilježi da patient leakage nije otkriven.
