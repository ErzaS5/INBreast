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

## Phase 1B — duže zamrzavanje i diskriminativni learning rate

Encoder je zamrznut pet umjesto dvije epohe. Klasifikaciona glava koristi LR `2e-5`, encoder nakon odmrzavanja `2e-6`, a plateau scheduler i early-stopping brojač se resetuju na granici faza. Zbog batch size 2, pretrained BatchNorm running statistike ostaju zamrznute i nakon odmrzavanja težina. Foldovi, seed i svi evaluacioni holdouti ostali su isti.

| Backbone | ROC-AUC | PR-AUC | Senzitivnost | Specifičnost | Balanced accuracy | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 Phase 1A | 0.5089 | 0.2827 | 0.9000 | 0.1513 | 0.5257 | 0.4018 | 0.0641 |
| ResNet-18 Phase 1B | 0.4995 | 0.2450 | 0.8200 | 0.1250 | 0.4725 | 0.3661 | -0.0687 |
| DenseNet-121 Phase 1A | 0.5034 | 0.2529 | 0.9000 | 0.1053 | 0.5026 | 0.3896 | 0.0074 |
| DenseNet-121 Phase 1B | 0.6200 | 0.3061 | 0.9000 | 0.3750 | 0.6375 | 0.4737 | 0.2573 |

ResNet nije profitirao od konzervativnijeg protokola. DenseNet jeste: poboljšao je sve glavne tuned-threshold metrike i u ovoj fazi nadmašio ResNet, ali je ostao ispod ranijeg Swin baseline-a. DenseNet rezultat je i dalje nestabilan među foldovima (fold-mean ROC-AUC standardna devijacija `0.1591`, PR-AUC `0.2354`), pa ga ne treba proglasiti konačnim pobjednikom bez ponavljanja preko više seedova.

DenseNet na fiksnom pragu 0.5 predviđa sve primjere kao negativne; dobar ranking se vidi tek uz prag iz nezavisnog unutrašnjeg threshold skupa. To znači da je sljedeći prioritet kalibracija izlaznih vjerovatnoća, a ne dodatno produžavanje zamrznute faze.

## Phase 1C — DenseNet kalibracija i tri seeda

ResNet-18 je nakon Phase 1B isključen iz daljih eksperimenata. DenseNet-121 je ponovljen sa seedovima 42, 43 i 44. Svaki outer fold sada ima dodatni, patient-disjoint calibration holdout i koristi temperature scaling; BatchNorm running statistike ostaju zamrznute tokom cijelog fine-tuninga.

| Seed | ROC-AUC | PR-AUC | Brier | ECE | Senzitivnost | Specifičnost | Balanced accuracy | F1 | MCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6278 | 0.3431 | 0.1828 | 0.0367 | 0.9000 | 0.3289 | 0.6145 | 0.4569 | 0.2220 |
| 43 | 0.5139 | 0.2498 | 0.2059 | 0.0832 | 0.8200 | 0.2303 | 0.5251 | 0.3942 | 0.0526 |
| 44 | 0.5495 | 0.3012 | 0.1858 | 0.0233 | 0.8600 | 0.2697 | 0.5649 | 0.4216 | 0.1315 |
| **Sredina ± sample SD** | **0.5637 ± 0.0582** | **0.2981 ± 0.0468** | **0.1915 ± 0.0126** | **0.0477 ± 0.0314** | **0.8600 ± 0.0400** | **0.2763 ± 0.0497** | **0.5682 ± 0.0448** | **0.4242 ± 0.0314** | **0.1354 ± 0.0848** |

Temperature scaling je prihvaćen u 14 od 15 foldova. Temperature su uglavnom bile između 1.3 i 4.3, ali je jedan fold dostigao gornju granicu `54.598`; to je upozorenje da je calibration holdout od približno devet pacijenata premalen za stabilnu procjenu temperature. Kalibracija je znatno smanjila Brier score i ECE, ali prag 0.5 i dalje gotovo uvijek daje samo negativnu klasu. Zato se operativni prag mora birati na nezavisnom threshold holdoutu, kako je već implementirano.

Konačan zaključak Phase 1C je da DenseNet ima signal iznad slučajnog rangiranja, ali rezultat zavisi od seeda i još je ispod Swin baseline-a. Dalje podešavanje na ovom istom skupu povećalo bi rizik od overfittinga na evaluacioni protokol; naredni korak treba da bude ili više podataka/eksterna validacija, ili unaprijed definisana ograničena promjena loss-a bez biranja na osnovu outer-test rezultata.
