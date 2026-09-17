# INbreast — završni izveštaj


Završen je svež, petostruki stratifikovani grouped CV postojećeg CC/MLO baseline-a. Konačna procena obuhvata **202 dojke, 107 pacijenata i 116 pregleda**. Svaki par ima tačno jednu out-of-fold predikciju. Senzitivnost je **0.9400**, specifičnost **0.3882**, ROC-AUC **0.7424**, PR-AUC **0.5231**. Ovo su rezultati spoljašnjih test foldova, a ne smoke ili checkpoint validation metrike.

Jedinica ostaje jedna dojka sa tačno dva prikaza iste osobe, datuma i strane. Leva i desna dojka su zasebni primeri; 27 pacijenata sa samo jednom validnom dojkom je uključeno. Nema četvoroslikovnog ulaza, patient-level fusion, konačne patient-level predikcije niti agregacije leve/desne verovatnoće. Pacijent služi grupisanju i statistici. Labela je očuvana: BI-RADS 1–3 = 0, 4–6 = 1, viša projekcijska oznaka po dojci. To je radiološka proxy labela, bez tvrdnje o histološki potvrđenom malignitetu.

**Pregled početnog stanja i očuvanje prethodnog rada**


Pre implementacije pregledani su svi navedeni izvorni fajlovi, originalni testovi, notebook, konfiguracija i prethodni DICOM/XML/metadata audit. Git repozitorijum nije prisutan; commit je null. Početni kod i artefakti sačuvani su u `artifacts/verification/original_project/` i `original_artifacts/`. Potvrđeni su prepare-only metadata generation, direktan inference, shared preprocessing, best/last checkpointi, schema/preprocessing verzija 2, identitet token+datum+strana i deterministička selekcija ekspozicija. Njihovo ponašanje je očuvano. DICOM audit nije nepotrebno ponovljen.

Dokazane greške/regresije su popravljene: raniji evaluate tok pozivao je nedefinisanu predikcijsku funkciju; integer BOX resize mogao je izgubiti vrlo sitan ROI; nedostajala je stroga validacija identiteta parova; premala stratifikacija imala je tihi random fallback; checkpoint izbor i prag koristili su isti skup; Grad-CAM se oslanjao na layout heuristiku i square metrike; resume u novom direktorijumu mogao je preuzeti noviji sibling last checkpoint. Image preprocessing, ImageNet normalizacija, labela, model baseline i pravilo ekspozicija nisu metodološki promenjeni. Nije uveden VOI/windowing.

| Fajl | Konačna izmena |
| --- | --- |
| data.py | Stroga metadata/file/identity/schema/leakage validacija, dokumentovane ekskluzije, patient holdouts, geometrija i deterministički preprocessing cache. |
| preprocessing.py | NaN/Inf provere, geometrija i povratak na originalni prostor; float BOX čuva sitne ROI-je. |
| model.py | Isti shared Swin late-fusion baseline, konfigurabilan backbone, oblik/batch/finite/device/output provere, arhitektura verzija 1. |
| train.py | Odvojeni selection skupovi, kompletan resume/provenance, checkpoint retention, svi režimi, grouped outer CV, raw logits, native i square CAM dijagnostika. |
| evaluate.py | Četiri threshold strategije, sve klasifikacione metrike, ECE/NLL, opcioni temperature scaling, patient-cluster bootstrap, localization summary. |
| gradcam.py | Eksplicitni NHWC layout/dva hook poziva/target provere; heatmap i region, original/model prikazi i geometrija. |
| configuration.py (nov) | CLI > YAML > default, schema i opsezi, nepoznati ključevi i ungrouped režim se odbijaju. |
| reporting.py (nov) | CSV i plot artefakti, kalibracija, FN/FP/TP/TN i podgrupe; stabilan JSON sa null. |
| environment.py, verify_dataset.py, prepare_cache.py (novi) | Environment snapshot, read-only izvorni hashovi, deterministički preprocessing cache. |
| dicom_audit.py, validate_xml.py | Postojeći audit kod očuvan; XML audit ponovljen radi potvrde stvarne ROI resize popravke. |
| tests/test_pipeline.py, tests/test_validation.py, pytest.ini | Originalnih 19 testova plus regresije/edge cases i realan E2E. CV fixture proširen jer osam pacijenata ne podržava potrebne disjunktne stratifikovane skupove. |
| config.yaml, final_experiment.yaml | Potpuna validirana konfiguracija i tačna, izvršena finalna konfiguracija. |
| requirements*.txt | Tačne runtime verzije, odvojeni pytest i potpun environment lock. |
| README.md, INbreast_Colab.ipynb | UTF-8, aktuelni kod, svi tokovi/ograničenja, GPU/environment provere i jasno odvojeni smoke/finalni režimi. |
| WORK_REPORT.md, INbreast_source.zip, artifacts/verification/*.py | Izveštaj, aktuelni Colab source paket i transparentna post-run verifikacija bez promene modela/odluka. |

**Testovi i redosled izvršenja**


| Skup | Prikupljeno/uspešno | Neuspešno | Preskočeno | Upozorenja | Trajanje |
| --- | --- | --- | --- | --- | --- |
| Početni kompletan pytest | 19 / 19 | 0 | 0 | 0 | 33.51 s |
| Konačni kompletan pytest | 119 / 119 | 0 | 0 | 27 očekivanih pickle upozorenja | 30.41 s |

Pytest upozorenja su eksplicitne poruke pre učitavanja pouzdanih lokalno napravljenih torch/pickle checkpointa; nije skrivena bibliotečka greška. Dodati testovi obuhvataju prazne/nepotpune/duplirane metadata, fajlove i identitete prikaza, strane/datume/pacijente, više datuma i obe dojke, leakage, ekspozicije/BI-RADS, jednu klasu, premalu stratifikaciju, unlabeled direktan inference, kompatibilnost, resume i očuvanje istorije, veću checkpoint epohu, NaN/Inf, verovatnoće, XML clipping/sitni ROI, CLI/YAML, split/loader RNG reproduktivnost i realan E2E. Realan Swin forward, Grad-CAM i CLI prepare/sanity/train/evaluate/predict/crossval su prošli. CPU resume test poredi kontinuirani i prekinuti trening uključujući stvarne težine i istoriju.

Pre dugog treninga prošli su metadata/file audit i svih pet preflight splitova, sanity pregled dve negativne/dve pozitivne dojke, realni MPS smoke dve epohe i resume do treće, evaluate validation dijagnostika i oba predict toka bez obavezne XML/labele. Smoke istorija [1,2] je očuvana i produžena na [1,2,3]; optimizer/scheduler/scaler/RNG/oba loader generatora i MPS RNG su sačuvani. Pretrained Swin forward pri 384 takođe je potvrđen. Tek zatim je pokrenut finalni trening od početka. Syntax/import/UTF-8/notebook code-cell provere su uspešne. Notebook nije izvršen na udaljenom Colab GPU-u.

Logovi i JUnit XML su u `artifacts/verification/pytest_initial.*` i `pytest_final.*`; `smoke_*`, `pretrained_forward.log`, `static_checks.json`, `final_cv_preflight.json` i `smoke_resume_verification.json` nose odvojene dokaze. Smoke metrike nisu konačan rezultat.

**Finalni metadata i izvorni podaci**


| Stavka | Broj |
| --- | --- |
| Originalni DICOM / podržane označene slike | 410 / 409 |
| Pacijenti / pregledi u označenim slikama | 108 / 117 |
| Pacijenti / pregledi u validnim parovima | 107 / 116 |
| CC/MLO parovi / leva / desna dojka | 202 / 104 / 98 |
| Pozitivni / negativni parovi | 50 / 152 |
| Parovi sa ROI / CC ROI / MLO ROI | 169 / 169 / 168 |
| Pacijenti sa više datuma / obe validne dojke / jednom dojkom | 9 / 80 / 27 |
| Nepotpune dojke / isključene slike / repeated kandidati | 3 / 6 / 4 |
| Konflikt binarne labele / raw BI-RADS konflikta | 0 / 2 |

BI-RADS raspodela parova: 2: 108, 1: 33, 5: 25, 4c: 11, 3: 11, 4a: 6, 6: 4, 4b: 4. Isključeno: 3 slike bez counterpart-a, 2 neizabrane ponovljene ekspozicije, 1 nepodržani FB prikaz. Footer/blank redovi tabele su auditovani u `metadata_exclusions.csv`; nepotpune dojke imaju zaseban razlog u `incomplete_pairs.csv`. Izbor ekspozicija ostaje najviša binarna labela, zatim ROI, zatim najniži numerički image ID. Svih 202 pair ID-ja, labela, odabranih image ID-ja i ROI flagova identično je početnom metadata (`pairing_invariance.json`).

Kanonski dataset je tokom izvornog eksperimenta bio u lokalnom direktorijumu koji nije deo repozitorijuma. DICOM headeri su anonimni i nemaju token/date/view; dostupni naziv i originalna XLS tabela daju identitet, uz stroge provere. Datum ove kopije ima mesečnu rezoluciju YYYYMM. ML se normalizuje na MLO kao ranije. Direktan inference može čitati postojeću originalnu tabelu radi datuma, bez pripreme metadata/labele/XML-a; kada datum nije dostupan, to se eksplicitno prijavljuje.

Završna read-only SHA-256 provera potvrđuje **956 nepromenjenih izvornih DICOM/XML/XLS/CSV fajlova**, uključujući svih **410 DICOM hashova** identičnih ranijem auditu. Originali i druga kopija nisu menjani ni brisani. XML validacija: 343 fajla, 338 OK, 5 poznatih upozorenja za clipping tačaka, 0 problema/grešaka/nedostajućih DICOM-a, 0 ROI odsečenih crop-om i 0 izgubljenih resize-om; 17 audit overlay-a.

**Metodologija i dokaz odsustva patient leakage-a**


Pet spoljašnjih stratifikovanih patient-grouped foldova daju jednu predikciju po dojci. Sve dojke i svi datumi iste osobe su u jednom spoljašnjem test foldu. Unutar svakog razvoja pacijenti su razdvojeni na train, checkpoint/early stopping/scheduler validation (cilj 0.15) i threshold holdout (cilj 0.15). Kalibracija je unapred isključena za ovaj baseline. Opcioni temperature režim uvodi zaseban calibration holdout. Spoljašnji test ne učestvuje u izboru checkpointa, praga, kalibraciji ili hiperparametrima.

Primena je outer grouped CV sa unutrašnjim disjunktnim holdoutima, a ne puna nested pretraga hiperparametara. Jedan unapred definisan baseline, bez poređenja arhitektura ili optimizacije hiperparametara na outer rezultatima. Time se izbegava veliko smanjenje već malih selection skupova i višestruki trening svih unutrašnjih foldova. Stratifikacioni patient max label služi samo raspodeli grupa, a ne patient-level izlazu. Nedovoljna stratifikacija prekida rad bez tihog random fallback-a. Grupisanje korelisanih primera je opisano u [scikit-learn dokumentaciji](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

| Fold | Uloga | Pacijenti | Parovi | + | − | L | R | Patient SHA-256 (prefix) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | threshold | 13 | 27 | 6 | 21 | 12 | 15 | 00a927f742f4… |
| 1 | train | 59 | 111 | 28 | 83 | 58 | 53 | eb42b66c8ecb… |
| 1 | val | 13 | 23 | 6 | 17 | 13 | 10 | 5a48b4797f65… |
| 1 | outer_test | 22 | 41 | 10 | 31 | 21 | 20 | dfd0e2bcea52… |
| 2 | threshold | 13 | 23 | 6 | 17 | 11 | 12 | 9b5e9384fcb2… |
| 2 | train | 59 | 110 | 28 | 82 | 58 | 52 | 2ade31292f9b… |
| 2 | val | 13 | 24 | 6 | 18 | 13 | 11 | 39e04077cecc… |
| 2 | outer_test | 22 | 45 | 10 | 35 | 22 | 23 | 9abd968099da… |
| 3 | threshold | 13 | 23 | 6 | 17 | 11 | 12 | 537da1050e63… |
| 3 | train | 60 | 112 | 27 | 85 | 58 | 54 | 255f873cacd5… |
| 3 | val | 13 | 27 | 6 | 21 | 15 | 12 | 91a3fd069faa… |
| 3 | outer_test | 21 | 40 | 11 | 29 | 20 | 20 | c2bc2d7fcd02… |
| 4 | threshold | 13 | 22 | 7 | 15 | 11 | 11 | 100fcafefa33… |
| 4 | train | 60 | 116 | 27 | 89 | 60 | 56 | d1a626af5239… |
| 4 | val | 13 | 24 | 6 | 18 | 13 | 11 | 8d012bb9a663… |
| 4 | outer_test | 21 | 40 | 10 | 30 | 20 | 20 | 731db06187df… |
| 5 | threshold | 13 | 28 | 7 | 21 | 13 | 15 | 8fac7d5855ea… |
| 5 | train | 60 | 116 | 28 | 88 | 59 | 57 | af91dba622ae… |
| 5 | val | 13 | 22 | 6 | 16 | 11 | 11 | 2ac791f85076… |
| 5 | outer_test | 21 | 36 | 9 | 27 | 21 | 15 | 49e7738ee6d1… |

Puni ID-jevi/hashovi i svi prazni preseci su u `patient_leakage_audit.json` i pojedinačnim `split_audit.json`. `final_result_verification.json` dodatno proverava checkpoint patient_roles, development metadata hash, disjunktnost outer pacijenata, tačno 202 jedinstvena para, 107 pacijenata, identične labele i tačno jedan outer fold po pacijentu. Prediction CSV oznake split=test/evaluation_role=outer_test dopunjene su posle rada procesa, uz očuvanje svakog originalnog probability/prediction/threshold tekstualnog polja; stari prepare holdout je zadržan u prepared_holdout_split.

**Konfiguracija, okruženje i training istorija**


Tačna konfiguracija je `final_experiment.yaml`: pretrained shared `swin_tiny_patch4_window7_224` encoder pri input size 384; dropout 0.3, batch 2, lr 2e-5, weight decay 1e-4, augmentacije uključene, prve 2 epohe zamrznut backbone, najviše 25 epoha/fold, PR-AUC checkpoint izbor, patience 5, plateau scheduler, seed 42 za split i 43–47 za fold trening, workers 0. Uređaj je Apple MPS; AMP se primenjuje samo na CUDA. Checkpoint retention best.pt/last.pt, bez periodičnih dodataka (checkpoint_every=0). Grad-CAM 4 para/fold, unapred fiksiran heatmap prag 0.5. Cache svih 404 prikaza je validiran, sa 337 nepraznih ROI-ja.

PR-AUC je unapred izabran zbog neravnoteže 50/152 i nezavisnosti od klasifikacionog praga. Konfiguracija podržava F1/sensitivity/ROC-AUC/validation loss; sensitivity sama ne opisuje false positives, a F1 zavisi od praga. Nije menjan kriterijum nakon test rezultata.

| Fold | Seed | Best epoha | Poslednja epoha | Best validation PR-AUC (selection) | Development metadata SHA-256 (prefix) |
| --- | --- | --- | --- | --- | --- |
| 1 | 43 | 18 | 23 | 0.9762 | 43b852bdddc9… |
| 2 | 44 | 11 | 16 | 0.8857 | e89a182f9187… |
| 3 | 45 | 7 | 12 | 0.7881 | 4f3c69a60649… |
| 4 | 46 | 21 | 25 | 0.9583 | 95f7e32750bb… |
| 5 | 47 | 1 | 6 | 0.5451 | 5cd2983f5637… |

Selection PR-AUC u prethodnoj tabeli objašnjava izbor težina i nije konačna procena. Cele istorije i krive su u `fold_N/history.csv` i `training_curves.png`. best/last imaju arhitekturu/verziju, backbone, veličinu, dropout, normalizaciju, jedan izlaz, label definiciju, metadata/preprocessing verziju 2, prag/kalibraciju, hash/roles/config/environment i kompletno resume stanje. Postprocessing je završen zasebno za best i last težine. Best checkpointi nisu menjani tokom dopune CAM artefakata; SHA-256 je sačuvan u završnoj verifikaciji.

Provereno okruženje: Python 3.13.7, macOS arm64, torch 2.14.0, torchvision 0.29.0, timm 1.0.29. Podržan Python opseg 3.12–3.13; sve tačne biblioteke su u requirements.txt i potpunom requirements-lock.txt. Runtime i pytest dependencies su odvojeni. Lokalni CUDA nije dostupan (cuda_version=null) i CUDA kombinacija nije lokalno verifikovana; Colab ispisuje stvarne runtime/driver verzije. Originali su nekompresovani, pa decoder dodaci nisu potrebni. Pretrained težine su u lokalnom Hugging Face cache-u; prvo preuzimanje dalo je informativno upozorenje o anonimnom pristupu. Git commit je null.

**Klasifikacioni pragovi i finalni breast-level rezultati**


Svaki best checkpoint koristi min_sensitivity sa ciljem 0.90 na nezavisnom threshold holdoutu. Kandidati su jedinstvene verovatnoće i precizni susedni/krajnji pragovi; bira se najbolja specifičnost, zatim najviši prag. Pravilo klasifikacije je p >= threshold. Prag se ne bira na checkpoint ili outer test pacijentima. Cilj je ostvaren na holdoutu svih foldova, bez upozorenja, ali nema garantovane test senzitivnosti.

| Fold | Prag | Threshold holdout sensitivity | Threshold holdout specificity | Izvor |
| --- | --- | --- | --- | --- |
| 1 | 0.0004610978940036148 | 1.0000 | 0.2857 | independent_threshold_holdout |
| 2 | 0.0044945525005459785 | 1.0000 | 0.8824 | independent_threshold_holdout |
| 3 | 0.001527990447357297 | 1.0000 | 0.1765 | independent_threshold_holdout |
| 4 | 0.00012119224993512034 | 1.0000 | 0.4667 | independent_threshold_holdout |
| 5 | 0.3595356047153473 | 1.0000 | 0.2381 | independent_threshold_holdout |

| Metrika | OOF, prag svakog folda | 95% patient-cluster CI | Fold mean ± sample SD | OOF pri 0.5 |
| --- | --- | --- | --- | --- |
| sensitivity | 0.9400 | 0.8684 – 1.0000 | 0.9378 ± 0.0570 | 0.3800 |
| specificity | 0.3882 | 0.3007 – 0.4788 | 0.3753 ± 0.2093 | 0.8816 |
| precision | 0.3357 | 0.2617 – 0.4118 | 0.3473 ± 0.0794 | 0.5135 |
| npv | 0.9516 | 0.8955 – 1.0000 | 0.9451 ± 0.0599 | 0.8121 |
| accuracy | 0.5248 | 0.4495 – 0.6011 | 0.5180 ± 0.1501 | 0.7574 |
| balanced_accuracy | 0.6641 | 0.6100 – 0.7189 | 0.6566 ± 0.1024 | 0.6308 |
| f1 | 0.4947 | 0.4072 – 0.5756 | 0.5026 ± 0.0806 | 0.4368 |
| mcc | 0.3071 | 0.2082 – 0.4044 | 0.2996 ± 0.1573 | 0.2919 |
| roc_auc | 0.7424 | 0.6598 – 0.8177 | 0.7823 ± 0.1498 | 0.7424 |
| pr_auc | 0.5231 | 0.4094 – 0.6544 | 0.6733 ± 0.2265 | 0.5231 |
| brier_score | 0.2135 | 0.1691 – 0.2607 | 0.2144 ± 0.1033 | 0.2135 |
| ece | 0.1973 | 0.1561 – 0.2558 | 0.2065 ± 0.1090 | 0.1973 |

Confusion matrix za pragove foldova, redovi stvarna 0/1 i kolone predikovana 0/1: [[59, 93], [3, 47]]. Pri fiksnom pragu 0.5: [[134, 18], [31, 19]]. PR-AUC je sklearn average precision. OOF ranking metrike kombinuju predikcije različitih modela; zato su zasebno prikazani fold mean/std i svi fold rezultati, posebno zbog različitih skala verovatnoća.

| Fold | Parovi | Sensitivity | Specificity | PPV | NPV | Accuracy | Balanced accuracy | F1 | MCC | ROC-AUC | PR-AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 41 | 0.9000 | 0.2258 | 0.2727 | 0.8750 | 0.3902 | 0.5629 | 0.4186 | 0.1363 | 0.8000 | 0.7188 | 0.3644 | 0.3700 |
| 2 | 45 | 0.9000 | 0.7143 | 0.4737 | 0.9615 | 0.7556 | 0.8071 | 0.6207 | 0.5171 | 0.8486 | 0.7340 | 0.1459 | 0.1469 |
| 3 | 40 | 1.0000 | 0.2069 | 0.3235 | 1.0000 | 0.4250 | 0.6034 | 0.4889 | 0.2587 | 0.7806 | 0.7251 | 0.2538 | 0.2605 |
| 4 | 40 | 1.0000 | 0.4333 | 0.3704 | 1.0000 | 0.5750 | 0.7167 | 0.5405 | 0.4006 | 0.9433 | 0.8978 | 0.0963 | 0.0971 |
| 5 | 36 | 0.8889 | 0.2963 | 0.2963 | 0.8889 | 0.4444 | 0.5926 | 0.4444 | 0.1852 | 0.5391 | 0.2908 | 0.2117 | 0.1578 |

Senzitivnost 0.94 predstavlja 47/50 pozitivnih dojki, uz 3 FN; specifičnost 0.3882 predstavlja 59/152 negativnih dojki, uz 93 FP. Donja 95% granica senzitivnosti je 0.8684, ispod cilja 0.90. Nije potvrđena pouzdana minimalna senzitivnost 0.90 za buduće podatke. Peti fold je najslabiji po rankingu (ROC-AUC 0.5391, PR-AUC 0.2908); njegov best checkpoint je iz prve epohe, kada je backbone zamrznut, a head treniran. To pokazuje nestabilnost malog checkpoint holdouta i ovog ranog training protokola, bez naknadnog menjanja pravila. Model nije klinički validiran.

95% percentile CI koristi 2000 patient-cluster bootstrap iteracija: pacijenti se uzorkuju sa ponavljanjem, pa se uključuju sve njihove dojke/datumi. Seed 42 zbirno, 43–47 pojedinačno. Čuvaju se broj validnih i nedefinisanih iteracija po metrici. Ovo su intervali uslovljeni već fitovanim modelima, splitovima i pragovima; trening se ne refituje i puna neizvesnost model selection-a nije obuhvaćena. Single-class metrike ostaju null sa upozorenjem; proces se ne ruši.

**Kalibracija**


| Procena | Metod | Primena | Brier | ECE, 10 bins | NLL |
| --- | --- | --- | --- | --- | --- |
| Pre | raw | ne | 0.2135 | 0.1973 | 1.0367 |
| Prijavljeno / posle | none | ne | 0.2135 | 0.1973 | 1.0367 |

Svih 202 prijavljenih probability vrednosti je raw. Pre/posle su identični jer kalibrator nije fitovan niti automatski primenjen u ovom unapred definisanom baseline-u. Nije tvrđeno da je temperature scaling poboljšao ovaj finalni eksperiment. Opcioni temperature scaling je implementiran i E2E testiran: fituje originalne logits na zasebnom calibration holdoutu, bira pozitivan T minimizacijom NLL, a primenu prihvata samo uz niži Brier i NLL na tom fit skupu. To je apparent diagnostic, a nepristrasna pre/posle procena ostaje outer test. Metod opisuje [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html). Reliability diagram i svi zauzeti/prazni binovi su u calibration_evaluation.json i reliability_diagram.png. Raw probability i confidence ne predstavljaju kliničku sigurnost.

**Grad-CAM, lokalizacija i pregled prečica**


| Prikaz | Evaluirani ROI heatmaps | Dice mean | Dice median | IoU mean | IoU median |
| --- | --- | --- | --- | --- | --- |
| cc | 18 | 0.0302 | 0.0000 | 0.0173 | 0.0000 |
| mlo | 18 | 0.0231 | 0.0000 | 0.0127 | 0.0000 |
| all | 36 | 0.0266 | 0.0000 | 0.0150 | 0.0000 |

| Fold | CC slike sa ROI | MLO slike sa ROI | Generisano | Evaluirano | Preskočeno | Razlozi |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 33 | 33 | 8 | 8 | 74 | {"example_limit_or_category_filter": 74} |
| 2 | 37 | 36 | 8 | 8 | 82 | {"example_limit_or_category_filter": 82} |
| 3 | 38 | 38 | 8 | 8 | 72 | {"example_limit_or_category_filter": 72} |
| 4 | 34 | 34 | 8 | 6 | 74 | {"example_limit_or_category_filter": 72, "no_roi": 2} |
| 5 | 27 | 27 | 8 | 6 | 66 | {"example_limit_or_category_filter": 64, "no_roi": 2} |

Ukupno dostupno 169 CC i 168 MLO ROI prikaza. Generisano je ograničenih 40 heatmapa za 20 odabranih outer test parova. Izbor daje prioritet FN, zatim FP/TP/TN, pa udaljenosti raw p od 0.5 i stabilnom pair ID-ju; nije reprezentativan localization uzorak. Samo validni neprazni XML ROI prikazi ulaze u Dice/IoU, u originalnoj DICOM rezoluciji i bez square padding-a. Heatmap prag 0.5 nije optimizovan na train/test slikama. Grad-CAM je pozitivan target class=1, poslednji Swin block norm1, NHWC, dva hook poziva. Region nije segmentation model output.

Po završetku treninga ista četiri ranije izabrana para/fold ponovo su obrađena istim best težinama i istom geometrijom/pragom, samo da bi se sačuvala dodatna model-square heatmapa/overlay sa padding-om. Originalni native Dice/IoU i verovatnoće su provereni kao saglasni; originalni CSV je sačuvan u verification/original_localization/. Nema novih izbora, fitovanja, promenjenih predikcija ili checkpointa. NPZ sadrži model i original-space heatmapu i geometriju; odvojeni su originalna slika, model input, ROI, thresholded region i overlay/comparison.

Ograničeni vizuelni pregled je završen i dokumentovan u `visual_review.json`, uključujući original i model-square prikaze iz svih pet foldova. Pregledano je 7 original-space comparison prikaza i 7 model-square overlay-a iz svih pet foldova. Vidljive su aktivacije uz ivice/pektoralnu oblast, u originalnoj crnoj pozadini i u dodatoj padding oblasti kod pojedinih primera. To označava hipoteze za kontrolisanu perturbacionu proveru, bez potvrde uzročnosti ili odsustva ostalih prečica.

Za svih 40 sačuvanih model-square CAM-ova, mean padding_heat_fraction = 0.2248, raspon 0.0000–0.9031. `padding_diagnostics.csv` odvojeno čuva padding area fraction i srednji CAM intenzitet u padding-u/sadržaju, jer masa zavisi i od površine padding-a. Ovaj nalaz ne dokazuje da promene padding-a menjaju predikciju. Na primer, fold 2 FN MLO ima aktivaciju u originalnoj crnoj pozadini, ali padding_heat_fraction=0; te dve oblasti nisu poistovećene.

- Fold 1, `<redacted_patient>_201001_L`, MLO, FN: Najveća CAM aktivacija je uz gornju ivicu i donju/perifernu oblast dojke; pragovani region ne preklapa prikazane XML ROI-je (Dice=0). Pektoralna/gornja oblast i crop ivica zaslužuju dodatnu kontrolu.

- Fold 1, `<redacted_patient>_200802_L`, MLO, FP: Široka aktivacija obuhvata deo jasno vidljivih kalcifikovanih ROI oblasti i okolno tkivo; BI-RADS labela je negativna. Preklapanje ne objašnjava ispravnost klasifikacije i region je mnogo širi od ROI-ja. Vidljive su i aktivacije u levom padding-u (22.5% CAM mase).

- Fold 2, `<redacted_patient>_200901_R`, MLO, FN: Najizraženija aktivacija je na donjoj ivici dojke i delom crnoj okolini; XML ROI u donjoj desnoj oblasti uglavnom nije obuhvaćen. Native-space Dice je praktično nula.

- Fold 3, `<redacted_patient>_201001_L`, CC, FP: Aktivacija je u centralnom tkivu i ispod sitnih XML ROI tačaka; pragovani region ne preklapa ROI (Dice=0). U prikazu nije vidljiv jasan tekstualni marker, ali odsustvo svih prečica nije dokazano.

- Fold 4, `<redacted_patient>_201001_L`, MLO, FP: Najjača aktivacija je u gornjoj MLO oblasti uz pektoralnu/crop ivicu, sa slabijom perifernom aktivacijom. Sitni XML ROI nije preklopljen (Dice=0); ovo je signal za kontrolu prečica, bez uzročnog zaključka.

- Fold 5, `<redacted_patient>_200901_R`, MLO, FN: CAM ističe uske gornje/periferne oblasti i deo crne pozadine unutar crop-a, dok centralni XML ROI nije preklopljen (Dice=0). Potrebna je kontrola osetljivosti na pozadinu/crop/padding; saliency nije dokaz uzročnosti. Model-square pregled pokazuje izrazitu aktivaciju izvan sadržaja; padding CAM masa je 87.1%.

- Fold 5, `<redacted_patient>_200901_R`, CC, FN: Original-space CAM ističe oblast bradavice i periferno tkivo; centralni XML ROI je van thresholded region-a (Dice=0). Pregledan original-space comparison, a ne model-square CC overlay.

- Fold 4, `<redacted_patient>_200902_R`, CC, FP: Dodatno pregledan model-square overlay zbog najveće padding CAM mase: 90.3% je izvan validnog sadržaja. Vidljive su jake izolovane oblasti u padding-u. ROI nedostaje pa Dice/IoU nisu računati. Ne predstavlja uzročni dokaz da padding određuje predikciju.

padding_heat_fraction meri relativnu CAM masu u padding-u, bez uzročnog zaključka. Pregled ne može isključiti sve scanner/crop/resolution/tekstualne prečice; pektoralni mišić nije evaluiran kao lesion ROI. Prazan ground-truth prikaz u slučaju has_roi=0 označava odsustvo anotacije, a ne validiranu background-only masku; takvi slučajevi imaju null Dice/IoU. Vizuelno preklapanje lezije nije dokaz klinički ispravnog učenja. Nizak Dice/IoU zahteva oprez i ne koristi se za izbor klasifikatora.

**Strukturisana analiza grešaka**


Kategorije na OOF parovima: {"FP": 93, "TN": 59, "TP": 47, "FN": 3}. Sve FN/FP/TP/TN predikcije su u zasebnim CSV-ovima. most_confident_errors.csv sortira pogrešne slučajeve po verovatnoći predikovane klase; least_confident_predictions.csv koristi istu confidence definiciju, a least_certain_predictions.csv udaljenost od konkretnog fold praga. Ta dva pojma mogu znatno odstupati kada prag nije 0.5.

| Par | Fold | Tip | BI-RADS | p | Prag | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| <redacted_patient>_200802_L | 1 | FP | 2 | 1.0000 | 0.0004610978940036 | 1.0000 |
| <redacted_patient>_201001_L | 1 | FP | 2 | 1.0000 | 0.0004610978940036 | 1.0000 |
| <redacted_patient>_200902_R | 1 | FP | 2 | 0.9998 | 0.0004610978940036 | 0.9998 |
| <redacted_patient>_201001_L | 1 | FN | 4b | 0.0004 | 0.0004610978940036 | 0.9996 |
| <redacted_patient>_200902_L | 1 | FP | 3 | 0.9994 | 0.0004610978940036 | 0.9994 |
| <redacted_patient>_200901_R | 2 | FN | 4c | 0.0024 | 0.0044945525005459 | 0.9976 |
| <redacted_patient>_200802_L | 1 | FP | 2 | 0.9959 | 0.0004610978940036 | 0.9959 |
| <redacted_patient>_201001_R | 1 | FP | 2 | 0.9937 | 0.0004610978940036 | 0.9937 |
| <redacted_patient>_200901_R | 1 | FP | 2 | 0.9912 | 0.0004610978940036 | 0.9912 |
| <redacted_patient>_200901_R | 1 | FP | 2 | 0.9859 | 0.0004610978940036 | 0.9859 |

Sva tri false-negative slučaja:

| Par | Fold | Strana | BI-RADS | p | Prag |
| --- | --- | --- | --- | --- | --- |
| <redacted_patient>_201001_L | 1 | L | 4b | 0.0004 | 0.0004610978940036 |
| <redacted_patient>_200901_R | 2 | R | 4c | 0.0024 | 0.0044945525005459 |
| <redacted_patient>_200901_R | 5 | R | 4c | 0.3522 | 0.3595356047153473 |

Pet predikcija najbližih konkretnom fold pragu:

| Par | Fold | Stvarna labela | Predikcija | p | Prag | Udaljenost |
| --- | --- | --- | --- | --- | --- | --- |
| <redacted_patient>_201001_L | 3 | 0 | 1 | 0.001531125511974 | 0.0015279904473572 | 3.135064616800005e-06 |
| <redacted_patient>_201001_L | 3 | 0 | 0 | 0.0015242202207446 | 0.0015279904473572 | 3.770226612599941e-06 |
| <redacted_patient>_200902_R | 4 | 0 | 0 | 0.0001165715730166 | 0.0001211922499351 | 4.620676918500006e-06 |
| <redacted_patient>_201001_L | 4 | 0 | 1 | 0.0001290786021854 | 0.0001211922499351 | 7.886352250299986e-06 |
| <redacted_patient>_201001_R | 4 | 0 | 1 | 0.0001294401154154 | 0.0001211922499351 | 8.247865480300008e-06 |

Sve podgrupe su deskriptivne, bez dodatne model selection procedure i bez grupnih intervala za veoma male kategorije. Jednoklasni podskupovi imaju null AUC i upozorenja.

| birads | Parovi | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 33 | null | 0.5152 | 0.0000 | null | null |
| 2 | 108 | null | 0.3426 | 0.0000 | null | null |
| 3 | 11 | null | 0.4545 | 0.0000 | null | null |
| 4a | 6 | 1.0000 | null | 1.0000 | null | null |
| 4b | 4 | 0.7500 | null | 0.8571 | null | null |
| 4c | 11 | 0.8182 | null | 0.9000 | null | null |
| 5 | 25 | 1.0000 | null | 1.0000 | null | null |
| 6 | 4 | 1.0000 | null | 1.0000 | null | null |

| side | Parovi | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| L | 104 | 0.9630 | 0.2597 | 0.4727 | 0.6811 | 0.5155 |
| R | 98 | 0.9130 | 0.5200 | 0.5250 | 0.8075 | 0.5582 |

| resolution | Parovi | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 3328x2560x3328x2560 | 119 | 0.9259 | 0.4130 | 0.4717 | 0.6985 | 0.4639 |
| 3328x2560x4084x3328 | 1 | null | 1.0000 | null | null | null |
| 4084x3328x4084x3328 | 82 | 0.9565 | 0.3390 | 0.5238 | 0.8018 | 0.6519 |

| has_roi | Parovi | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| False | 33 | null | 0.5152 | 0.0000 | null | null |
| True | 169 | 0.9400 | 0.3529 | 0.5402 | 0.7323 | 0.5558 |

| multiple_exams | Parovi | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| False | 171 | 0.9574 | 0.3548 | 0.5233 | 0.7352 | 0.5641 |
| True | 31 | 0.6667 | 0.5357 | 0.2222 | 0.8333 | 0.3131 |

| repeated_exposure | Parovi | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| False | 200 | 0.9388 | 0.3841 | 0.4894 | 0.7371 | 0.5119 |
| True | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Tačne komande za reprodukciju i artefakti**


Iz korena projekta, sa originalnim datasetom na dokumentovanoj putanji:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export HF_HOME="$PWD/artifacts/model_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/inbreast-matplotlib"
python environment.py
pytest
python train.py --config config.yaml --mode prepare
python train.py --config config.yaml --mode sanity
python verify_dataset.py --data-root "../INbreast Release 1.0"
python train.py --config config.yaml --mode train --size 224 --epochs 2 --output-dir artifacts/smoke --no-pretrained
python train.py --config config.yaml --mode train --size 224 --epochs 3 --output-dir artifacts/smoke --no-pretrained --resume artifacts/smoke/last.pt
python train.py --config config.yaml --mode evaluate --output-dir artifacts/smoke --eval-split val --bootstrap-iterations 100
python train.py --config config.yaml --mode predict --output-dir artifacts/smoke --pair-id "<redacted_patient>_200902_L"
python prepare_cache.py --metadata-pairs artifacts/metadata_pairs.csv --cache-dir artifacts/preprocessed --size 384
# Fresh output; do not mix with existing final results.
python -u train.py --config final_experiment.yaml > artifacts/final_training.log 2>&1
python artifacts/verification/finalize_outputs.py
python verify_dataset.py --data-root "../INbreast Release 1.0"
```

Finalni predict primeri, koji rekonstruišu model iz best checkpointa i ne fituju nikakve parametre:

```bash
python train.py --config final_experiment.yaml --mode predict --checkpoint artifacts/final_baseline/crossval/fold_1/best.pt --output-dir artifacts/final_predict_pair --pair-id "<redacted_patient>_201001_L" --no-gradcam-enabled
python train.py --config final_experiment.yaml --mode predict --checkpoint artifacts/final_baseline/crossval/fold_1/best.pt --output-dir artifacts/final_predict_direct --cc-path "/putanja/CC.dcm" --mlo-path "/putanja/MLO.dcm" --no-gradcam-enabled
```

Stvarno izvršena završna provera oba inference ulaza je `python artifacts/verification/check_final_inference.py`: par `<redacted_patient>_201001_L`, probability 0.7665117383003235, threshold 0.0004610978940036148, prediction 1. Identitet i rezultat su saglasni OOF; direktni ulaz ima label=null i prazne XML putanje. Dokaz je `final_inference_verification.json`, a izlazi su u artifacts/final_predict_pair i final_predict_direct.

Glavni artefakti: metadata_images/pairs.csv, metadata_audit.json, dataset_summary.csv, incomplete_pairs.csv, excluded_images.csv, metadata_exclusions.csv, repeated_exposure_decisions.csv, sanity/paired_overlays.png; crossval/metadata_pairs_cv.csv, patient_leakage_audit.json, oof_predictions.csv, predictions.csv, fold_metrics.csv, metrics.json; svaki fold best.pt/last.pt/history/curves/config/environment/split audit, development/outer CSV, threshold/calibration JSON i classification/localization CSV/plot artefakti. Zbirni i pojedinačni confusion matrix, ROC/PR CSV+plot, calibration/reliability, error-analysis i FN/FP/TP/TN tabele su sačuvani. CAM original_images/model_inputs/heatmaps/gradcam/gradcam_regions/ground_truth_masks/comparisons su odvojeni. Verification čuva source hashove, početni/finalni pytest log/XML, XML audit, smoke/resume/pretrained/syntax/import dokaze, originalne snapshotove i završne provere.

`artifacts/verification/artifact_manifest.csv` navodi svaki generisani fajl i veličinu; `INbreast_source.zip` sadrži aktuelni kod/testove/konfiguraciju/dokumentaciju/notebook za upload u Colab, bez dataseta, cache-a i checkpointa. README i notebook dokumentuju holdout evaluate/resume tok; izvršena završna procena ovog rada je pet-fold CV. Pouzdane lokalne checkpoint-e učitavajte uz eksplicitno pickle upozorenje. Colab notebook je sintaksno validiran, ali njegova CUDA izvršivost nije ovde potvrđena.

**Poznata ograničenja i sledeći eksperiment**


Skup je mali i neuravnotežen; selection holdouti imaju malo pozitivnih pacijenata, pragovi i fold rezultati su nestabilni, a raw probabilities mogu biti loše kalibrisane. Pooled OOF AUC je osetljiv na razlike skale različitih fold modela. Grupni bootstrap ne resampluje trening i ne daje kliničku validaciju. BI-RADS nije histologija, mesec nije pun acquisition date, anonimni headeri ograničavaju nezavisnu identitetsku potvrdu. Nema spoljašnjeg dataseta, potpuno nested hiperparametarske pretrage ili provere CUDA determinističnosti. Lokalizacija je mali, greškama obogaćen podskup i post-hoc heatmapa, bez segmentacionog modela. Vizuelni pregled ne dokazuje odsustvo prečica.

Sledeći eksperiment treba unapred definisati sa istim CC/MLO baseline-om, posebnim patient-grouped calibration holdoutom i temperature scaling-om, uz nepromenjen checkpoint kriterijum i sensitivity threshold strategiju. Paralelno predefinisati kontrolisanu proveru osetljivosti na padding/crnu pozadinu i crop bez uklanjanja tkiva. Procenu poboljšanja kalibracije/screening ponašanja uraditi na novim nezavisnim outer podacima ili spoljašnjem datasetu; postojeće OOF rezultate ne koristiti za iterativno podešavanje. Pozitivan temperature scaling čuva ranking unutar jednog modela i sam ne popravlja njegovu diskriminaciju. Eventualnu promenu warmup/early-stopping pravila zbog slabog petog folda tretirati kao novi predefinisani protokol i oceniti na novim podacima. Dodatne arhitekture razmatrati tek uz kontrolisanu, unapred navedenu metodologiju i uporedivu jedinicu dojke.
