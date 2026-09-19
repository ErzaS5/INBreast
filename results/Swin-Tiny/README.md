# Finalni Swin-Tiny rezultati

Ovaj folder sadrži mali, Git-friendly izbor rezultata finalnog 5-fold eksperimenta.
Kompletni generisani izlazi ostaju lokalno u `artifacts/phase2_swin_thresholds/`, dok
checkpointi, velike heatmape i pojedinačne predikcije nisu uključeni.

## Metrike

- `metrics/threshold_comparison.json` — poređenje pragova `youden_j` i `max_f1`.
- `metrics/crossval_metrics.json` — zbirne OOF metrike 5-fold evaluacije.
- `metrics/fold_metrics.csv` — metrike po spoljašnjem foldu.

Za izabrani `youden_j` prag glavne OOF vrijednosti su:

- senzitivnost: 0.6600
- specifičnost: 0.8092
- balanced accuracy: 0.7346
- F1: 0.5893
- ROC-AUC: 0.7668
- PR-AUC: 0.6120

## Slike

- `figures/` sadrži zbirne klasifikacione krive i matricu konfuzije.
- `gradcam_examples/` sadrži dva anonimizovana true-positive slučaja, sa CC i MLO
  prikazom za svaki slučaj. Svaka comparison slika prikazuje originalni snimak,
  XML masku, Grad-CAM i njihov preklop.

Grad-CAM je pomoć za interpretaciju klasifikacije, a ne segmentacioni model. Zbog
toga su njegove aktivacione regije šire od XML maski i IoU ne treba tumačiti kao
rezultat precizne segmentacije.

Rezultati su generisani 19.09.2026. i nisu namijenjeni kliničkoj upotrebi.
