# Phase 1 — poređenje backbone-a

Eksperiment je izvršen na istih 202 CC/MLO parova i pet identičnih patient-grouped outer foldova, sa seedom 42. Preprocessing, podjele, checkpoint kriterijum (`pr_auc`), broj epoha, early stopping i threshold strategija ostali su isti. Grad-CAM je isključen jer nije dio klasifikacionog poređenja.

| Backbone | Parametri | ROC-AUC | PR-AUC | Senzitivnost | Specifičnost | Balanced accuracy | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny (raniji baseline) | 27,523,963 | 0.7424 | 0.5231 | 0.9400 | 0.3882 | 0.6641 | 0.4947 | 0.3071 |
| ResNet-18 | 11,179,585 | 0.5089 | 0.2827 | 0.9000 | 0.1513 | 0.5257 | 0.4018 | 0.0641 |
| DenseNet-121 | 6,960,001 | 0.5034 | 0.2529 | 0.9000 | 0.1053 | 0.5026 | 0.3896 | 0.0074 |

Metrike senzitivnosti/specifičnosti koriste threshold iz nezavisnog unutrašnjeg threshold holdouta svakog folda. Rezultati na pragu 0.5 takođe su sačuvani u lokalnim, Git-ignored artefaktima.

## Zaključak

Ni ResNet-18 ni DenseNet-121 nisu dostigli Swin-Tiny u ovom unaprijed definisanom Phase 1 protokolu. Oba CNN-a imaju pooled ROC-AUC približno 0.50, pa ih u ovoj konfiguraciji ne treba smatrati zamjenom za Swin. Swin se zato ne uklanja.

Ovo poređenje ne dokazuje da CNN backbone-i generalno ne mogu raditi dobro. Njihov validation loss je često rastao neposredno nakon odmrzavanja encodera, što ukazuje da bi zaseban, manji encoder learning rate mogao biti opravdan kao novi eksperiment. Takvo podešavanje više ne bi bilo čisto poređenje samo arhitekture i mora se evidentirati kao zasebna faza.

Lokalni artefakti sa checkpointima i redovnim predikcijama ostaju u `artifacts/` i nisu namijenjeni za Git. Ovaj dokument sadrži samo agregatne rezultate.
