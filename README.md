# INbreast: paired-view klasifikacija i Grad-CAM lokalizacija

Projekat klasifikuje jednu dojku pomoću uparenih `CC` i `MLO` mamograma. Isti pretrained Swin-Tiny encoder obrađuje obe projekcije, feature-i se spajaju, a jedna glava predviđa `BI-RADS 1–3 = 0` ili `BI-RADS 4–6 = 1`. Grad-CAM se računa zasebno za obe projekcije i poredi sa XML ROI maskama kroz Dice i IoU.

## Podaci

`--data-root` može pokazivati na bilo koji folder iznad INbreast sadržaja. Pipeline rekurzivno pronalazi INbreast XLS/XLSX/CSV, DICOM fajlove i tumorske XML maske. `PectoralMuscle` anotacije se isključuju.

Par čine CC i MLO projekcija istog `PatientID`-a i iste strane dojke. Nepotpuni parovi se ne koriste. Ako se BI-RADS projekcija razlikuje, koristi se viši nalaz i konflikt se evidentira. Podela je 85/15 po pacijentima, bez test skupa.

## Instalacija i pokretanje

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python train.py --data-root "/putanja/INbreast" --mode prepare
python train.py --data-root "/putanja/INbreast" --mode sanity
python train.py --data-root "/putanja/INbreast" --mode train --epochs 25
python train.py --data-root "/putanja/INbreast" --mode train --epochs 40 --resume artifacts/best.pt
python train.py --data-root "/putanja/INbreast" --mode evaluate
python train.py --data-root "/putanja/INbreast" --mode predict
```

Vrednosti mogu biti u `config.yaml`: `python train.py --config config.yaml --mode train`. Ako nema GPU memorije koristiti `--size 224 --batch-size 1`; za lokalni Windows koristiti `--workers 0`.

## Izlazi

- `metadata_images.csv`, `metadata_pairs.csv`, `dataset_summary.csv`
- `sanity/paired_overlays.png`
- `best.pt`, `history.csv`, `training_curves.png`
- `metrics.json`, `localization_metrics.csv`
- `gradcam/*.png` sa heatmapom i zelenom ground-truth konturom

Primarne metrike koriste klasifikacioni prag `0.5`. Dodatno se prikazuje prag podešen na validaciji. Grad-CAM prag bira se na train maskama, a Dice/IoU računaju na validation maskama.

`history.csv` sadrži `train_score` i `val_score` (F1, u opsegu `[0, 1]`).
`train_loss` i `val_loss` su BCE loss vrijednosti i namjerno nisu normalizovane, jer
loss nije vjerovatnoća niti score i može biti veći od 1. Predikcija modela prolazi
kroz sigmoid funkciju, pa je dobijena vjerovatnoća uvijek u opsegu `[0, 1]`.

## Ograničenja

- Ovo nije segmentacioni model; lokalizacija je post-hoc Grad-CAM.
- Validacija služi za izbor checkpointa jer po zahtevu nema test skupa.
- Rezultati nisu klinički validirani.
- Pretrained model pri prvom pokretanju zahteva internet ili postojeći `timm` cache.
