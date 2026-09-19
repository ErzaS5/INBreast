# Swin-Tiny — poređenje klasifikacionih pragova

Jedan five-fold patient-grouped CV trening sa seedom 42 korišten je za oba načina izbora praga. U svakom foldu prag je izabran isključivo na unutrašnjem, patient-disjoint threshold holdoutu i zatim primijenjen na netaknuti outer-test fold. Zbog toga razlika između strategija nije posljedica drugačijih težina ili test leakage-a.

| Strategija | Senzitivnost | Specifičnost | Balanced accuracy | Precision | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `youden_j` | 0.6600 | 0.8092 | **0.7346** | 0.5323 | **0.5893** | **0.4391** |
| `max_f1` | 0.5800 | **0.8553** | 0.7176 | **0.5686** | 0.5743 | 0.4324 |

Obje strategije dijele isti ranking rezultat: ROC-AUC `0.7668` i PR-AUC `0.6120`. `youden_j` je preporučena glavna radna tačka jer daje veću balanced accuracy, F1 i MCC uz manji gubitak senzitivnosti. `max_f1` ostaje korisna alternativna radna tačka kada je veća specifičnost važnija.

Za poređenje, raniji `min_sensitivity=0.90` protokol imao je senzitivnost `0.94`, specifičnost `0.3882` i balanced accuracy `0.6641`. Novi rezultat pokazuje da je veliki dio problema niske specifičnosti dolazio od namjerno vrlo osjetljivog praga, a ne od potpunog odsustva diskriminativnog signala u modelu.

Pragovi po foldovima nisu isti i ne treba ih prosječiti u jedan univerzalni prag. Za budući finalni model prag se mora ponovo odabrati na njegovom zasebnom development threshold skupu.

## Grad-CAM dijagnostika

Geometrijsko preslikavanje iz 384×384 model inputa u originalni DICOM prostor je provjereno i ispravno. Početni loš rezultat nije predstavljao tipično ponašanje modela: stari selektor je prvo birao FN i FP greške i svih osam dostupnih mjesta popunio bez ijednog TP primjera. Na šest pozitivnih FN prikaza Dice je bio praktično nula, što je očekivano jer je model već propustio pozitivan signal.

Na svih sedam TP parova prvog outer folda, odnosno 14 CC/MLO prikaza, klasični Grad-CAM daje:

| Skup | Dice mean | Dice median | IoU mean | IoU median |
| --- | ---: | ---: | ---: | ---: |
| CC | 0.4202 | 0.3330 | 0.2997 | 0.1997 |
| MLO | 0.4537 | 0.5241 | 0.3361 | 0.3551 |
| Sve | **0.4369** | **0.3785** | **0.3179** | **0.2344** |

`LayerCAM` (`Dice=0.0745`) i `HiResCAM` (`Dice=0.0897`) nisu pobijedili klasični Grad-CAM (`Dice=0.0972`) na istom miješanom FN/FP dijagnostičkom podskupu, pa ostaju podržani samo za transparentno eksperimentisanje. Pretposljednji Swin stage takođe nije povećao prosječni Dice i zato je zadržan standardni posljednji stage. Padding se isključuje i heatmapa se ponovo normalizuje isključivo unutar stvarnog modelskog sadržaja.

Selektor sada uzorkuje tražene kategorije round-robin umjesto da sve primjere uzme iz prve kategorije. Za izvještaj treba prikazati TP i FN odvojeno: TP Grad-CAM pokazuje gdje je model pronašao koristan signal, dok FN prikazuje zašto se propuštena lezija ne može naknadno „popraviti“ vizualizacijom.
