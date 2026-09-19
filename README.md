# INbreast: breast-level CC/MLO klasifikacija

Jedan primer je **jedna dojka**, predstavljena tačno jednim CC i jednim MLO mamogramom. Shared encoder obrađuje dva prikaza i vraća jednu verovatnoću, binarnu predikciju i klasifikacioni prag. Podržani backbone-i su Swin-Tiny, ResNet-18 i DenseNet-121. Leva i desna dojka su odvojeni primeri; dovoljan je jedan kompletan par. Četiri projekcije pacijenta nisu jedan model input. Nema patient-level fusion niti patient-level predikcije.

Pacijent služi isključivo za grupisanje svih strana i svih datuma radi sprečavanja leakage-a, stratifikacije i bootstrap statistike. Identitet para je `patient token + acquisition date + side`; konkretni identifikatori nisu dio javne dokumentacije.

## Labela i značenje verovatnoće

Zadržana definicija je **BI-RADS 1–3 = 0; BI-RADS 4–6, uključujući 4a/b/c = 1**. Ako projekcije daju različite binarne oznake, labela dojke je viša oznaka, uz audit konflikta. BI-RADS nije histološka potvrda maligniteta: model procenjuje ovu radiološku labelu, pa njegov izlaz ne treba tumačiti kao klinički validiranu verovatnoću raka. `probability_kind` označava `raw` ili `calibrated`; visoka verovatnoća ne garantuje tačan nalaz.

## Dataset

Dataset se ne čuva u Git repozitorijumu. Podrazumevana prenosiva putanja u `config.yaml` je `data/INbreast Release 1.0`. Možete napraviti tu lokalnu strukturu ili proslediti sopstvenu putanju pomoću `--data-root`. Originalni DICOM/XML fajlovi se samo čitaju i nikada se ne menjaju.

```text
data/INbreast Release 1.0/
  INbreast.xls                 # ili INbreast.csv / .xlsx
  AllDICOMs/*.dcm
  AllXML/*.xml                 # lesion ROI; PectoralMuscle se isključuje
  AllROI/                     # originalni dodatni podaci ostaju netaknuti
```

INbreast DICOM headeri nemaju PatientID, datum ni projekciju u ovoj kopiji. Token se uzima iz naziva fajla, datum iz tabele, a strana/projekcija proveravaju kroz dostupne header/filename/table vrednosti. Oznaka `ML` se, kao u prethodnim fazama, normalizuje na MLO. Dataset root mora označavati jednu kanonsku kopiju; duplirani image ID prekida pripremu.

Ponovljene ekspozicije ostaju u auditu. Kanonski izbor je: najviša binarna labela, zatim ROI, zatim najniži numerički image ID. Nepotpune dojke su u `incomplete_pairs.csv`; sve isključene slike i razlozi su u `excluded_images.csv`. Razlike u BI-RADS kategoriji beleže se zasebno od konflikta binarne labele.

## Okruženje i testovi

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python environment.py
python -m pytest
python train.py --config config.yaml --mode prepare
```

Ako je dataset izvan projekta, nije potrebno kopirati ga. Putanju zadajte na komandnoj liniji, na primer:

```powershell
python train.py --config config.yaml --mode prepare --data-root "D:\Datasets\INbreast Release 1.0"
```

Lokalne putanje možete čuvati i u kopiji `config.local.yaml`; taj fajl je isključen iz Gita. `artifacts/`, modeli, cache i medicinski podaci su takođe isključeni. Mali sažeci već izvršenog eksperimenta nalaze se u `results/`.

### macOS/Linux

Podržani opseg je Python 3.12–3.13; provereno lokalno: Python 3.13.7, PyTorch 2.14.0, torchvision 0.29.0, timm 1.0.29, macOS arm64. Tačne runtime verzije su u `requirements.txt`, pytest u `requirements-dev.txt`, a kompletan snapshot svih instaliranih biblioteka u `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export MPLCONFIGDIR="${TMPDIR:-/tmp}/inbreast-matplotlib"
export HF_HOME="$PWD/artifacts/model_cache"
python environment.py
pytest
```

Lokalni CUDA nije dostupan, pa CUDA kombinacija ovde nije verifikovana. Colab ispisuje stvarne Python/PyTorch/CUDA/driver verzije; za GPU runtime instalirajte odgovarajući PyTorch wheel za taj runtime i zabeležite eventualno odstupanje od lock fajla. Apple MPS je podržan kada ga proces može pristupiti; ograničen sandbox može ga prikazati kao nedostupan. CPU je takođe podržan. Pretrained težine pri prvom pokretanju zahtevaju internet, a zatim se koriste iz cache-a.

Za potpun snapshot istog macOS/Python okruženja koristite `python -m pip install -r requirements-lock.txt`. Runtime/dev fajlovi pin-uju direktne zavisnosti; lock pin-uje i tranzitivne. CUDA runtime može zahtevati dodatne platform-specific pakete, pa njegova reprodukcija mora sačuvati sopstveni environment snapshot.

Ova kopija sadrži nekompresovane DICOM slike i ne zahteva dodatne decodere. Za drugu transfer syntax proverite decoder kroz `pydicom`; opcione decoder biblioteke nisu automatski dodate.

## Konfiguracija

Precedence je **CLI argument > YAML konfiguracija > podrazumevana vrednost**. YAML se ne učitava automatski: koristite `--config config.yaml`. Nepoznati ključevi, nevalidni opsezi, nepodržan backbone i negrupisani split prekidaju rad pre treninga. Isti parametar ne sme biti istovremeno zadat kao flat YAML ključ i nested ključ. Koristite `python train.py --help` za sve opcije.

`metadata_schema_version=2`, `preprocessing_version=2`, arhitektura `shared_encoder_cc_mlo_late_fusion`, verzija 2. Crop je određen intenzitetom, nezavisno od ROI-ja. Float BOX resize čuva veoma sitne ROI-je bez 8-bitnog zaokruživanja. Nije uveden novi VOI/windowing postupak. ImageNet mean/std ostaje isti.

Konfiguracije arhitektonskih eksperimenata grupisane su po modelu da root projekta ostane pregledan:

```text
experiments/
  Swin-Tiny/{threshold_comparison.yaml,RESULTS.md}
  DenseNet-121/{baseline,stable,calibrated}.yaml
  ResNet-18/{baseline,stable}.yaml
  PHASE1_RESULTS.md
```

Phase 1 baseline konfiguracije koriste iste foldove, seed, preprocessing i trening protokol kao početni Swin eksperiment; razlikuju se samo backbone i output direktorijum. DenseNet `stable` i `calibrated` konfiguracije dokumentuju kasnije eksperimente sa odvojenim encoder/head learning rate-om, zamrznutim BatchNorm statistikama i temperature calibration. ResNet-18 i DenseNet-121 više nisu kandidati za dodatno podešavanje, ali ostaju reproduktivno dokumentovani. Seedovi 43 i 44 za kalibrisani DenseNet pokreću se CLI override-om za `--seed` i `--output-dir`.

Swin threshold poređenje bira `youden_j` i `max_f1` pragove na istom nezavisnom threshold holdoutu svakog folda, bez ponovnog treninga za drugu strategiju. Rezultati i preporučena radna tačka nalaze se uz konfiguraciju u `experiments/Swin-Tiny/RESULTS.md`.

## Priprema i sanity

```bash
python train.py --config config.yaml --mode prepare
python train.py --config config.yaml --mode sanity
python verify_dataset.py --data-root "data/INbreast Release 1.0"
```

`prepare` je jedini režim koji automatski generiše metadata. Ostali režimi učitavaju postojeći `metadata_pairs.csv` i strogo proveravaju potrebne fajlove, identitet prikaza i grouping. Putanja se može promeniti preko `--metadata-pairs`; eksplicitni `--rebuild-metadata` ponovo priprema tabelu. API `load_metadata_pairs(..., check_files=False)` služi samo za prenosiv pregled CSV-a; svi izvršni režimi koriste `check_files=True`.

Metadata sadrži `pair_id`, `patient_id`, `patient_token`, `acquisition_date`, `exam_id`, `side`, `label`, `split`, `label_conflict`, `birads_conflict`, `birads`, obe verzije i za svaki prikaz: image ID, DICOM/XML putanju, ROI flag, patient/date/side/view, BI-RADS, rezoluciju, broj kandidata i razlog izbora. Pair ID je jedinstven. Ponovna upotreba DICOM-a u drugim parovima zahteva `dicom_reuse_reason` i prolazak svih ostalih identitetskih provera.

## Evaluaciona metodologija

Osnovni grouped holdout ima disjunktne pacijente za `train`, `val` (isključivo checkpoint/early stopping/scheduler), `threshold` (isključivo klasifikacioni prag) i `test` (konačna procena). Opcioni temperature scaling dodaje poseban `calibration` skup. Podrazumevane frakcije su 0.15 checkpoint, 0.15 threshold i 0.20 test; kalibracija dodaje 0.10 kada je uključena. Frakcije su ciljevi, a zaokruživanje je na nivou pacijenta.

Preferirana završna procena je **spoljašnji stratifikovani grouped CV**. Svaki razvojni fold se zasebno deli na train/checkpoint/threshold/optional calibration pacijente; spoljašnji test fold ne učestvuje ni u jednoj odluci. Ovo nije puna nested hyperparameter CV: nema pretrage hiperparametara i postoje nezavisni unutrašnji holdout skupovi. Jedan baseline se trenira po spoljašnjem foldu. Stratifikaciona oznaka označava prisustvo pozitivne dojke radi raspodele pacijenata; ona nije izlaz modela niti konačna patient-level labela.

`split_audit.json` i `crossval/patient_leakage_audit.json` čuvaju broj pacijenata/parova/klasa/strana, identitete i njihove SHA-256 hashove, kao i sve prazne preseke. Ako nema dovoljno pacijenata za obe klase u svakom potrebnom holdoutu/fold-u, rad se prekida bez random fallback-a. Sve fold podele se proveravaju pre prvog treninga.

## Smoke, resume i finalni trening

Smoke služi samo proveri pipeline-a; njegove metrike nisu finalni rezultat. Odvojeni output direktorijumi sprečavaju mešanje eksperimenata.

```bash
python train.py --config config.yaml --mode train --size 224 --epochs 2 --output-dir artifacts/smoke --no-pretrained
python train.py --config config.yaml --mode train --size 224 --epochs 3 --output-dir artifacts/smoke --no-pretrained --resume artifacts/smoke/last.pt
python train.py --config config.yaml --mode evaluate --output-dir artifacts/smoke --eval-split val --bootstrap-iterations 100
python train.py --config config.yaml --mode predict --output-dir artifacts/smoke --pair-id "<PAIR_ID>"

# Finalni baseline od početka; tek posle pytest, metadata, leakage i smoke provera.
# Stvarno izvršena MPS five-fold konfiguracija: python train.py --config final_experiment.yaml
python train.py --config config.yaml --mode train --output-dir artifacts/final_holdout --epochs 25
python train.py --config config.yaml --mode evaluate --output-dir artifacts/final_holdout
python train.py --config config.yaml --mode crossval --output-dir artifacts/final_baseline --folds 5 --epochs 25
```

Za nastavak koristite `last.pt`. Restore uključuje optimizer, scheduler, GradScaler, Python/NumPy/Torch/CUDA/MPS RNG i oba DataLoader generatora. Persistent workers su isključeni da bi se worker RNG rekonstruisao pri svakoj epohi. Resume proverava metadata/split hash i relevantnu konfiguraciju; istorija se čuva. Stariji checkpoint ne sme prepisati noviju istoriju u istom direktorijumu. U novom direktorijumu se kopira upravo resume stanje, a ne noviji sibling last.pt. Periodični snapshot sa starijim najboljim modelom zahteva odgovarajući istorijski best.pt; bez njega nema tihog izbora novijih težina. Ako je checkpoint već iza traženog broja epoha, nema novog treninga.

Prethodni checkpointi bez kompatibilnih verzija/arhitekture/provenance se odbijaju. `best.pt` i `last.pt` ostaju odvojeni. `checkpoint_every: 0` čuva ta dva fajla; pozitivan interval dodaje `epoch_NNN.pt`. Prag/kalibrator se zasebno fituju na nezavisnim holdoutima za konkretne težine oba checkpointa.

Checkpoint metrika je konfigurabilna: `pr_auc` je početni izbor za neuravnotežene klase; ROC-AUC meri ranking obe klase, sensitivity zanemaruje cenu false positives, F1 zavisi od praga, a validation loss meri logit loss. Sve ove odluke koriste samo checkpoint holdout. Konačni test rezultat ne bira metricu.

Checkpoint i `experiment.json` čuvaju arhitekturu, backbone, input size, dropout, jedan izlaz, normalizaciju, label definiciju, verzije, prag i kalibraciju, konfiguraciju, seed, uređaj, biblioteke/Python/CUDA, normalizovan metadata frame SHA-256 i Git hash kada postoji. Ovaj direktorijum nema Git repozitorijum pa je commit `null`.

**Učitavajte samo pouzdane checkpoint fajlove.** `torch.load(weights_only=False)` koristi pickle koji može izvršiti kod; CLI daje upozorenje pre svakog takvog učitavanja.

## Prag i kalibracija

```yaml
threshold:
  strategy: min_sensitivity # fixed, max_f1, youden_j, min_sensitivity
  fixed_value: 0.5
  minimum_sensitivity: 0.90
calibration:
  method: none              # opciono temperature
  bins: 10
```

Pretraga koristi jedinstvene verovatnoće, susedne floating-point pragove i krajeve [0,1], sa pravilom `p >= threshold`. `min_sensitivity` bira najbolju specifičnost među pragovima koji dostižu cilj, a zatim najveći prag radi determinističkog tie-a. Cilj se odnosi na threshold holdout; ne garantuje istu senzitivnost na test skupu. Na skupu sa obe klase prag 0 uvek može postići senzitivnost 1, uz potencijalno specifičnost 0. Ostvaren rezultat, strategija, izvor i upozorenja čuvaju se u `threshold_selection.json`.

Temperature scaling koristi originalne model logits (ne gubi ih zbog saturacije sigmoid verovatnoća) i fituje se samo na nezavisnom calibration holdoutu. Primena zahteva niži Brier i NLL na tom skupu; inače probability ostaje raw. To je dijagnostika fitovanja, a nepristrasna procena pre/posle nalazi se samo na test/outer rezultatima. Ni kalibrator ni prag se ne prepravljaju posle posmatranja test rezultata. Podrazumevani baseline ne primenjuje kalibraciju; ipak beleži Brier, ECE i reliability diagram.

## Evaluate i predict

```bash
python train.py --config config.yaml --mode evaluate --output-dir artifacts/final_holdout
# Samo development dijagnostika, eksplicitno označena kao nefinalna:
python train.py --config config.yaml --mode evaluate --output-dir artifacts/final_holdout --eval-split val
python train.py --config config.yaml --mode predict --output-dir artifacts/final_holdout --pair-id "<PAIR_ID>"
python train.py --config config.yaml --mode predict --output-dir artifacts/final_holdout --cc-path "/putanja/CC.dcm" --mlo-path "/putanja/MLO.dcm"
```

Direktan inference zahteva samo dva validna DICOM-a iste osobe/strane istog datuma kada je dostupan u headeru ili postojećoj INbreast tabeli pored DICOM-a. Čitanje te tabele je opciono i ne generiše metadata fajlove. Labela i XML nisu obavezni. Ako datum nije dostupan iz oba izvora, ostaje prazan uz eksplicitno upozorenje da datum nije proveren; nema izmišljenog datuma. Veličina modelskog ulaza se učitava iz checkpointa.

Metrike su breast-level: sensitivity/recall, specificity, precision/PPV, NPV, accuracy, balanced accuracy, F1, MCC, ROC-AUC, PR-AUC (average precision), Brier, ECE i 2×2 confusion matrix. Nedefinisane vrednosti su `null` uz upozorenje u JSON-u. Prediction CSV sadrži identitet para/pacijenta, datum, stranu, labelu, raw/reported probability, prediction, threshold, fold, image ID i ROI podatke.

95% percentile confidence intervali koriste patient-cluster bootstrap: uzorkovanje pacijenata sa ponavljanjem i uključivanje svih njihovih dojki/datuma, fiksan seed, podrazumevano 2000 iteracija. Nedefinisane bootstrap metrike se preskaču uz broj validnih/nedefinisanih iteracija. Modeli, foldovi i pragovi se ne refituju tokom bootstrapa; intervali ne obuhvataju svu neizvesnost treninga. Fold mean/std koristi sample standard deviation (`ddof=1`).

## Grad-CAM i analiza grešaka

```bash
python train.py --config config.yaml --mode evaluate --output-dir artifacts/final_holdout --gradcam-enabled --gradcam-examples 8 --gradcam-categories FN FP --heatmap-threshold 0.5
```

Grad-CAM je opciona post-hoc **heatmapa**; thresholded Grad-CAM region nije izlaz segmentacionog modela. Target layer i layout biraju se prema backbone-u (Swin NHWC, CNN NCHW), za pozitivnu klasu (1), sa tačno dva hook poziva. Konstantna mapa daje nule; nevalidne aktivacije/gradijenti se odbijaju.

Geometrija čuva originalnu rezoluciju, crop, resize i padding. Heatmapa se vraća sa modelskog kvadrata preko sadržaja bez paddinga na originalnu DICOM sliku. Lokalizacija koristi originalni XML ROI i računa Dice/IoU samo za prikaze sa validnim nepraznim ROI-jem, nezavisno od klasifikacione labele. Prag heatmape je unapred konfigurisan, ne fituje se na train/test slikama. Broj primera je ograničen; svi neobrađeni prikazi i razlozi su u coverage izveštaju. Rezultati ograničenog podskupa nisu procena lokalizacije celog dataseta.

Čuvaju se originalna slika, model input, model-square heatmapa i overlay uključujući padding, original-space heatmapa sa geometrijom, thresholded region, ground-truth ROI, overlay i poređenje. `padding_heat_fraction` je opisna dijagnostika saliency mase, a ne dokaz uzročne prečice. `error_analysis.json` i FN/FP/TP/TN CSV-ovi sadrže najpouzdanije greške, najmanje sigurne predikcije i podgrupe po BI-RADS/strani/rezoluciji/ROI-ju/broju pregleda/ponovljenim ekspozicijama. Vizuelni pregled overlay-a treba tražiti prečice u tekstu, ivicama, pozadini, paddingu, artefaktima skenera i pektoralnom mišiću. Preklapanje Grad-CAM-a sa lezijom nije dokaz klinički ispravnog učenja.

## Artefakti i Colab

- Priprema: `metadata_images.csv`, `metadata_pairs.csv`, `metadata_audit.json`, `dataset_summary.csv`, `incomplete_pairs.csv`, `excluded_images.csv`, `metadata_exclusions.csv`, `repeated_exposure_decisions.csv`, `split_audit.json`.
- Trening: `best.pt`, `last.pt`, opciono `epoch_NNN.pt`, `history.csv`, `training_curves.png`, `experiment.json`, `experiment_config.yaml`, `threshold_selection.json`, `calibration.json`.
- Evaluacija: `metrics.json`, `predictions.csv`, `confusion_matrix.png`, ROC/PR CSV i plot, `calibration_evaluation.json`, `reliability_diagram.png`, error-analysis JSON i CSV, localization CSV i coverage u metrics JSON-u.
- CV: svaki fold ima svoje artefakte; zbirni `crossval/oof_predictions.csv`, `fold_metrics.csv`, `metrics.json`, `patient_leakage_audit.json` i sve evaluacione vizualizacije.
- Grad-CAM: `original_images/`, `model_inputs/`, `heatmaps/`, `gradcam/`, `gradcam_regions/`, `ground_truth_masks/`, `comparisons/`.
- Provera: početni/prošireni/finalni pytest log/XML, source hash manifest, potvrda očuvanih 202 parova i README/Colab validacije u `artifacts/verification/`.

`INbreast_Colab.ipynb` koristi aktuelni source paket koji se učita u Colab, proverava GPU i verzije, pokreće pytest pre pripreme, odvaja smoke/finalne direktorijume i podržava resume. Finalni CV je eksplicitno uključen odvojenim switch-em. Notebook ne klonira staru udaljenu verziju preko trenutnog koda.

Skup je mali, pacijenti su korelisani, BI-RADS je proxy labela, a unutrašnji holdout skupovi i intervali mogu biti nestabilni. Baseline i prag nisu klinički validirani. Sledeći eksperiment treba predefinisati pre gledanja novih test rezultata; najpre proveriti kalibraciju na nezavisnim grupama ili nezavisnom spoljašnjem datasetu, uz istu CC/MLO jedinicu klasifikacije.

Metodološki izvori: grupisanje povezanih primera opisuje [scikit-learn grouped CV dokumentacija](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data); temperature scaling na izdvojenoj validaciji opisuje [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html).
