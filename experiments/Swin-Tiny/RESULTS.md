# Swin-Tiny — poređenje klasifikacionih pragova

Jedan five-fold patient-grouped CV trening sa seedom 42 korišten je za oba načina izbora praga. U svakom foldu prag je izabran isključivo na unutrašnjem, patient-disjoint threshold holdoutu i zatim primijenjen na netaknuti outer-test fold. Zbog toga razlika između strategija nije posljedica drugačijih težina ili test leakage-a.

| Strategija | Senzitivnost | Specifičnost | Balanced accuracy | Precision | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `youden_j` | 0.6600 | 0.8092 | **0.7346** | 0.5323 | **0.5893** | **0.4391** |
| `max_f1` | 0.5800 | **0.8553** | 0.7176 | **0.5686** | 0.5743 | 0.4324 |

Obje strategije dijele isti ranking rezultat: ROC-AUC `0.7668` i PR-AUC `0.6120`. `youden_j` je preporučena glavna radna tačka jer daje veću balanced accuracy, F1 i MCC uz manji gubitak senzitivnosti. `max_f1` ostaje korisna alternativna radna tačka kada je veća specifičnost važnija.

Za poređenje, raniji `min_sensitivity=0.90` protokol imao je senzitivnost `0.94`, specifičnost `0.3882` i balanced accuracy `0.6641`. Novi rezultat pokazuje da je veliki dio problema niske specifičnosti dolazio od namjerno vrlo osjetljivog praga, a ne od potpunog odsustva diskriminativnog signala u modelu.

Pragovi po foldovima nisu isti i ne treba ih prosječiti u jedan univerzalni prag. Za budući finalni model prag se mora ponovo odabrati na njegovom zasebnom development threshold skupu.
