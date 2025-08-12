✓ Configuration loaded

============================================================
STEP 1: Loading Data
============================================================
Loading training data...
Loading test data...
✓ Train shape: (574945, 341)
✓ Test shape: (107, 336)
✓ Using 11 common IMU columns
✓ Train-only columns: 5 columns
✓ Test-only columns: 0 columns

============================================================
STEP 2: Preparing Sequences
============================================================
✓ Prepared 8151 training sequences
✓ Prepared 2 test sequences

============================================================
STEP 3: Feature Engineering
============================================================
Extracting training features...
Processing sequence 1/8151...
Processing sequence 101/8151...
Processing sequence 201/8151...
Processing sequence 301/8151...
Processing sequence 401/8151...
Processing sequence 501/8151...
Processing sequence 601/8151...
Processing sequence 701/8151...
Processing sequence 801/8151...
Processing sequence 901/8151...
Processing sequence 1001/8151...
Processing sequence 1101/8151...
Processing sequence 1201/8151...
Processing sequence 1301/8151...
Processing sequence 1401/8151...
Processing sequence 1501/8151...
Processing sequence 1601/8151...
Processing sequence 1701/8151...
Processing sequence 1801/8151...
Processing sequence 1901/8151...
Processing sequence 2001/8151...
Processing sequence 2101/8151...
Processing sequence 2201/8151...
Processing sequence 2301/8151...
Processing sequence 2401/8151...
Processing sequence 2501/8151...
Processing sequence 2601/8151...
Processing sequence 2701/8151...
Processing sequence 2801/8151...
Processing sequence 2901/8151...
Processing sequence 3001/8151...
Processing sequence 3101/8151...
Processing sequence 3201/8151...
Processing sequence 3301/8151...
Processing sequence 3401/8151...
Processing sequence 3501/8151...
Processing sequence 3601/8151...
Processing sequence 3701/8151...
Processing sequence 3801/8151...
Processing sequence 3901/8151...
Processing sequence 4001/8151...
Processing sequence 4101/8151...
Processing sequence 4201/8151...
Processing sequence 4301/8151...
Processing sequence 4401/8151...
Processing sequence 4501/8151...
Processing sequence 4601/8151...
Processing sequence 4701/8151...
Processing sequence 4801/8151...
Processing sequence 4901/8151...
Processing sequence 5001/8151...
Processing sequence 5101/8151...
Processing sequence 5201/8151...
Processing sequence 5301/8151...
Processing sequence 5401/8151...
Processing sequence 5501/8151...
Processing sequence 5601/8151...
Processing sequence 5701/8151...
Processing sequence 5801/8151...
Processing sequence 5901/8151...
Processing sequence 6001/8151...
Processing sequence 6101/8151...
Processing sequence 6201/8151...
Processing sequence 6301/8151...
Processing sequence 6401/8151...
Processing sequence 6501/8151...
Processing sequence 6601/8151...
Processing sequence 6701/8151...
Processing sequence 6801/8151...
Processing sequence 6901/8151...
Processing sequence 7001/8151...
Processing sequence 7101/8151...
Processing sequence 7201/8151...
Processing sequence 7301/8151...
Processing sequence 7401/8151...
Processing sequence 7501/8151...
Processing sequence 7601/8151...
Processing sequence 7701/8151...
Processing sequence 7801/8151...
Processing sequence 7901/8151...
Processing sequence 8001/8151...
Processing sequence 8101/8151...
✓ Extracted features shape: (8151, 360)
Extracting test features...
Processing sequence 1/2...
✓ Extracted features shape: (2, 360)

============================================================
STEP 4: Model Training
============================================================

============================================================
Training LightGBM with 5-fold cross-validation
============================================================
Number of features: 359
Number of samples: 8151
Number of classes: 18

--- Training Fold 1 ---
Train size: 6623, Val size: 1528
Training until validation scores don't improve for 100 rounds
[10] train's multi_logloss: 1.99464 valid's multi_logloss: 2.1428
[20] train's multi_logloss: 1.63235 valid's multi_logloss: 1.87066
[30] train's multi_logloss: 1.38646 valid's multi_logloss: 1.69366
[40] train's multi_logloss: 1.20426 valid's multi_logloss: 1.5709
[50] train's multi_logloss: 1.0626 valid's multi_logloss: 1.4791
[60] train's multi_logloss: 0.948062 valid's multi_logloss: 1.41002
[70] train's multi_logloss: 0.853417 valid's multi_logloss: 1.35754
[80] train's multi_logloss: 0.773317 valid's multi_logloss: 1.31249
[90] train's multi_logloss: 0.704002 valid's multi_logloss: 1.27571
[100] train's multi_logloss: 0.643771 valid's multi_logloss: 1.24651
[110] train's multi_logloss: 0.591316 valid's multi_logloss: 1.22302
[120] train's multi_logloss: 0.544502 valid's multi_logloss: 1.20165
[130] train's multi_logloss: 0.503114 valid's multi_logloss: 1.18423
[140] train's multi_logloss: 0.466227 valid's multi_logloss: 1.16898
[150] train's multi_logloss: 0.432236 valid's multi_logloss: 1.15459
[160] train's multi_logloss: 0.401905 valid's multi_logloss: 1.14371
[170] train's multi_logloss: 0.373836 valid's multi_logloss: 1.13469
[180] train's multi_logloss: 0.348513 valid's multi_logloss: 1.12529
[190] train's multi_logloss: 0.325683 valid's multi_logloss: 1.11663
[200] train's multi_logloss: 0.30446 valid's multi_logloss: 1.11072
[210] train's multi_logloss: 0.284558 valid's multi_logloss: 1.10267
[220] train's multi_logloss: 0.266308 valid's multi_logloss: 1.09648
[230] train's multi_logloss: 0.249403 valid's multi_logloss: 1.09194
[240] train's multi_logloss: 0.233939 valid's multi_logloss: 1.0877
[250] train's multi_logloss: 0.219714 valid's multi_logloss: 1.08456
[260] train's multi_logloss: 0.206531 valid's multi_logloss: 1.08074
[270] train's multi_logloss: 0.194103 valid's multi_logloss: 1.07745
[280] train's multi_logloss: 0.182571 valid's multi_logloss: 1.07497
[290] train's multi_logloss: 0.171672 valid's multi_logloss: 1.07211
[300] train's multi_logloss: 0.161515 valid's multi_logloss: 1.06996
[310] train's multi_logloss: 0.151932 valid's multi_logloss: 1.06857
[320] train's multi_logloss: 0.143052 valid's multi_logloss: 1.06633
[330] train's multi_logloss: 0.134799 valid's multi_logloss: 1.06468
[340] train's multi_logloss: 0.127063 valid's multi_logloss: 1.06371
[350] train's multi_logloss: 0.119671 valid's multi_logloss: 1.06246
[360] train's multi_logloss: 0.112937 valid's multi_logloss: 1.06124
[370] train's multi_logloss: 0.106493 valid's multi_logloss: 1.06004
[380] train's multi_logloss: 0.100612 valid's multi_logloss: 1.05991
[390] train's multi_logloss: 0.0949703 valid's multi_logloss: 1.05891
[400] train's multi_logloss: 0.0895684 valid's multi_logloss: 1.05739
[410] train's multi_logloss: 0.0845374 valid's multi_logloss: 1.05785
[420] train's multi_logloss: 0.0798526 valid's multi_logloss: 1.05762
[430] train's multi_logloss: 0.0754343 valid's multi_logloss: 1.05748
[440] train's multi_logloss: 0.0712965 valid's multi_logloss: 1.05761
[450] train's multi_logloss: 0.0674305 valid's multi_logloss: 1.05747
[460] train's multi_logloss: 0.0637529 valid's multi_logloss: 1.05771
[470] train's multi_logloss: 0.0602716 valid's multi_logloss: 1.05711
[480] train's multi_logloss: 0.0570144 valid's multi_logloss: 1.05626
[490] train's multi_logloss: 0.0539167 valid's multi_logloss: 1.05624
[500] train's multi_logloss: 0.0510268 valid's multi_logloss: 1.05713
[510] train's multi_logloss: 0.0483694 valid's multi_logloss: 1.05752
[520] train's multi_logloss: 0.0458214 valid's multi_logloss: 1.05794
[530] train's multi_logloss: 0.0434408 valid's multi_logloss: 1.05826
[540] train's multi_logloss: 0.0412266 valid's multi_logloss: 1.05853
[550] train's multi_logloss: 0.0391097 valid's multi_logloss: 1.05932
[560] train's multi_logloss: 0.0371559 valid's multi_logloss: 1.0601
[570] train's multi_logloss: 0.0352897 valid's multi_logloss: 1.06077
[580] train's multi_logloss: 0.0335121 valid's multi_logloss: 1.06121
Early stopping, best iteration is:
[487] train's multi_logloss: 0.054826 valid's multi_logloss: 1.05588
Fold 1 training completed. Best iteration: 487
Fold 1 - Competition Score: 0.7580 (Binary F1: 0.9804, Macro F1: 0.5357)

--- Training Fold 2 ---
Train size: 6519, Val size: 1632
Training until validation scores don't improve for 100 rounds
[10] train's multi_logloss: 1.97135 valid's multi_logloss: 2.23491
[20] train's multi_logloss: 1.60067 valid's multi_logloss: 1.99864
[30] train's multi_logloss: 1.35317 valid's multi_logloss: 1.84694
[40] train's multi_logloss: 1.17084 valid's multi_logloss: 1.73892
[50] train's multi_logloss: 1.02911 valid's multi_logloss: 1.65783
[60] train's multi_logloss: 0.914453 valid's multi_logloss: 1.59757
[70] train's multi_logloss: 0.820726 valid's multi_logloss: 1.55151
[80] train's multi_logloss: 0.741756 valid's multi_logloss: 1.51547
[90] train's multi_logloss: 0.673952 valid's multi_logloss: 1.48547
[100] train's multi_logloss: 0.614696 valid's multi_logloss: 1.46088
[110] train's multi_logloss: 0.562925 valid's multi_logloss: 1.43998
[120] train's multi_logloss: 0.517131 valid's multi_logloss: 1.42193
[130] train's multi_logloss: 0.476851 valid's multi_logloss: 1.40854
[140] train's multi_logloss: 0.440448 valid's multi_logloss: 1.39674
[150] train's multi_logloss: 0.407542 valid's multi_logloss: 1.38711
[160] train's multi_logloss: 0.377872 valid's multi_logloss: 1.38071
[170] train's multi_logloss: 0.35103 valid's multi_logloss: 1.37446
[180] train's multi_logloss: 0.326781 valid's multi_logloss: 1.36854
[190] train's multi_logloss: 0.304515 valid's multi_logloss: 1.36365
[200] train's multi_logloss: 0.283846 valid's multi_logloss: 1.35874
[210] train's multi_logloss: 0.265004 valid's multi_logloss: 1.35597
[220] train's multi_logloss: 0.247466 valid's multi_logloss: 1.35167
[230] train's multi_logloss: 0.231567 valid's multi_logloss: 1.34903
[240] train's multi_logloss: 0.216806 valid's multi_logloss: 1.34654
[250] train's multi_logloss: 0.203116 valid's multi_logloss: 1.34457
[260] train's multi_logloss: 0.190441 valid's multi_logloss: 1.34349
[270] train's multi_logloss: 0.178439 valid's multi_logloss: 1.34233
[280] train's multi_logloss: 0.16736 valid's multi_logloss: 1.33999
[290] train's multi_logloss: 0.15704 valid's multi_logloss: 1.33944
[300] train's multi_logloss: 0.147394 valid's multi_logloss: 1.33804
[310] train's multi_logloss: 0.138363 valid's multi_logloss: 1.33711
[320] train's multi_logloss: 0.129902 valid's multi_logloss: 1.33669
[330] train's multi_logloss: 0.122073 valid's multi_logloss: 1.33611
[340] train's multi_logloss: 0.114823 valid's multi_logloss: 1.33573
[350] train's multi_logloss: 0.107936 valid's multi_logloss: 1.33592
[360] train's multi_logloss: 0.101644 valid's multi_logloss: 1.33667
[370] train's multi_logloss: 0.0957037 valid's multi_logloss: 1.33663
[380] train's multi_logloss: 0.0900839 valid's multi_logloss: 1.3363
[390] train's multi_logloss: 0.0848649 valid's multi_logloss: 1.33601
[400] train's multi_logloss: 0.0799843 valid's multi_logloss: 1.33628
[410] train's multi_logloss: 0.075339 valid's multi_logloss: 1.33709
[420] train's multi_logloss: 0.0710417 valid's multi_logloss: 1.33787
[430] train's multi_logloss: 0.0670889 valid's multi_logloss: 1.33913
[440] train's multi_logloss: 0.0633517 valid's multi_logloss: 1.33973
Early stopping, best iteration is:
[344] train's multi_logloss: 0.111991 valid's multi_logloss: 1.33553
Fold 2 training completed. Best iteration: 344
Fold 2 - Competition Score: 0.7111 (Binary F1: 0.9578, Macro F1: 0.4643)

--- Training Fold 3 ---
Train size: 6526, Val size: 1625
Training until validation scores don't improve for 100 rounds
[10] train's multi_logloss: 1.97443 valid's multi_logloss: 2.22541
[20] train's multi_logloss: 1.6047 valid's multi_logloss: 1.99026
[30] train's multi_logloss: 1.35478 valid's multi_logloss: 1.83335
[40] train's multi_logloss: 1.17051 valid's multi_logloss: 1.7243
[50] train's multi_logloss: 1.02893 valid's multi_logloss: 1.6441
[60] train's multi_logloss: 0.914245 valid's multi_logloss: 1.58043
[70] train's multi_logloss: 0.819571 valid's multi_logloss: 1.53139
[80] train's multi_logloss: 0.740017 valid's multi_logloss: 1.49408
[90] train's multi_logloss: 0.671782 valid's multi_logloss: 1.46453
[100] train's multi_logloss: 0.611734 valid's multi_logloss: 1.4396
[110] train's multi_logloss: 0.559697 valid's multi_logloss: 1.42056
[120] train's multi_logloss: 0.513709 valid's multi_logloss: 1.40347
[130] train's multi_logloss: 0.473129 valid's multi_logloss: 1.38985
[140] train's multi_logloss: 0.437481 valid's multi_logloss: 1.38082
[150] train's multi_logloss: 0.404969 valid's multi_logloss: 1.37071
[160] train's multi_logloss: 0.375559 valid's multi_logloss: 1.36275
[170] train's multi_logloss: 0.349572 valid's multi_logloss: 1.35656
[180] train's multi_logloss: 0.325184 valid's multi_logloss: 1.35015
[190] train's multi_logloss: 0.302957 valid's multi_logloss: 1.34462
[200] train's multi_logloss: 0.28251 valid's multi_logloss: 1.34049
[210] train's multi_logloss: 0.264062 valid's multi_logloss: 1.3373
[220] train's multi_logloss: 0.246759 valid's multi_logloss: 1.33458
[230] train's multi_logloss: 0.230568 valid's multi_logloss: 1.33195
[240] train's multi_logloss: 0.215936 valid's multi_logloss: 1.33027
[250] train's multi_logloss: 0.202475 valid's multi_logloss: 1.32926
[260] train's multi_logloss: 0.189654 valid's multi_logloss: 1.3277
[270] train's multi_logloss: 0.177786 valid's multi_logloss: 1.32482
[280] train's multi_logloss: 0.166797 valid's multi_logloss: 1.32375
[290] train's multi_logloss: 0.156665 valid's multi_logloss: 1.32296
[300] train's multi_logloss: 0.146998 valid's multi_logloss: 1.32231
[310] train's multi_logloss: 0.13818 valid's multi_logloss: 1.32161
[320] train's multi_logloss: 0.129874 valid's multi_logloss: 1.32106
[330] train's multi_logloss: 0.122195 valid's multi_logloss: 1.32174
[340] train's multi_logloss: 0.114996 valid's multi_logloss: 1.32223
[350] train's multi_logloss: 0.108276 valid's multi_logloss: 1.32143
[360] train's multi_logloss: 0.101961 valid's multi_logloss: 1.32194
[370] train's multi_logloss: 0.0959673 valid's multi_logloss: 1.32248
[380] train's multi_logloss: 0.0903877 valid's multi_logloss: 1.32161
[390] train's multi_logloss: 0.0851506 valid's multi_logloss: 1.3219
[400] train's multi_logloss: 0.0802024 valid's multi_logloss: 1.32372
[410] train's multi_logloss: 0.075605 valid's multi_logloss: 1.32392
[420] train's multi_logloss: 0.0712527 valid's multi_logloss: 1.32458
Early stopping, best iteration is:
[323] train's multi_logloss: 0.127502 valid's multi_logloss: 1.32095
Fold 3 training completed. Best iteration: 323
Fold 3 - Competition Score: 0.7100 (Binary F1: 0.9518, Macro F1: 0.4681)

--- Training Fold 4 ---
Train size: 6519, Val size: 1632
Training until validation scores don't improve for 100 rounds
[10] train's multi_logloss: 1.96747 valid's multi_logloss: 2.26564
[20] train's multi_logloss: 1.59664 valid's multi_logloss: 2.03512
[30] train's multi_logloss: 1.34879 valid's multi_logloss: 1.88793
[40] train's multi_logloss: 1.16592 valid's multi_logloss: 1.77919
[50] train's multi_logloss: 1.02343 valid's multi_logloss: 1.69706
[60] train's multi_logloss: 0.910014 valid's multi_logloss: 1.6319
[70] train's multi_logloss: 0.816504 valid's multi_logloss: 1.5813
[80] train's multi_logloss: 0.737511 valid's multi_logloss: 1.54262
[90] train's multi_logloss: 0.669197 valid's multi_logloss: 1.51269
[100] train's multi_logloss: 0.610047 valid's multi_logloss: 1.48613
[110] train's multi_logloss: 0.559095 valid's multi_logloss: 1.46677
[120] train's multi_logloss: 0.514488 valid's multi_logloss: 1.44865
[130] train's multi_logloss: 0.474124 valid's multi_logloss: 1.43291
[140] train's multi_logloss: 0.438317 valid's multi_logloss: 1.42033
[150] train's multi_logloss: 0.405873 valid's multi_logloss: 1.41073
[160] train's multi_logloss: 0.376611 valid's multi_logloss: 1.40276
[170] train's multi_logloss: 0.350363 valid's multi_logloss: 1.39558
[180] train's multi_logloss: 0.326321 valid's multi_logloss: 1.38974
[190] train's multi_logloss: 0.304258 valid's multi_logloss: 1.38533
[200] train's multi_logloss: 0.284343 valid's multi_logloss: 1.38058
[210] train's multi_logloss: 0.265555 valid's multi_logloss: 1.3757
[220] train's multi_logloss: 0.24831 valid's multi_logloss: 1.37298
[230] train's multi_logloss: 0.232636 valid's multi_logloss: 1.36944
[240] train's multi_logloss: 0.217884 valid's multi_logloss: 1.36637
[250] train's multi_logloss: 0.204359 valid's multi_logloss: 1.36395
[260] train's multi_logloss: 0.191587 valid's multi_logloss: 1.36131
[270] train's multi_logloss: 0.179549 valid's multi_logloss: 1.35948
[280] train's multi_logloss: 0.168552 valid's multi_logloss: 1.35889
[290] train's multi_logloss: 0.158387 valid's multi_logloss: 1.35846
[300] train's multi_logloss: 0.148698 valid's multi_logloss: 1.35716
[310] train's multi_logloss: 0.139681 valid's multi_logloss: 1.35717
[320] train's multi_logloss: 0.131268 valid's multi_logloss: 1.3561
[330] train's multi_logloss: 0.123506 valid's multi_logloss: 1.3556
[340] train's multi_logloss: 0.11626 valid's multi_logloss: 1.35491
[350] train's multi_logloss: 0.109389 valid's multi_logloss: 1.35511
[360] train's multi_logloss: 0.102903 valid's multi_logloss: 1.35569
[370] train's multi_logloss: 0.0968141 valid's multi_logloss: 1.35548
[380] train's multi_logloss: 0.0910632 valid's multi_logloss: 1.35595
[390] train's multi_logloss: 0.085794 valid's multi_logloss: 1.35608
[400] train's multi_logloss: 0.0807993 valid's multi_logloss: 1.35646
[410] train's multi_logloss: 0.0761932 valid's multi_logloss: 1.35696
[420] train's multi_logloss: 0.0717626 valid's multi_logloss: 1.35722
[430] train's multi_logloss: 0.0676361 valid's multi_logloss: 1.35926
[440] train's multi_logloss: 0.0637851 valid's multi_logloss: 1.35984
Early stopping, best iteration is:
[347] train's multi_logloss: 0.111387 valid's multi_logloss: 1.35489
Fold 4 training completed. Best iteration: 347
Fold 4 - Competition Score: 0.7028 (Binary F1: 0.9493, Macro F1: 0.4564)

--- Training Fold 5 ---
Train size: 6417, Val size: 1734
Training until validation scores don't improve for 100 rounds
[10] train's multi_logloss: 1.97094 valid's multi_logloss: 2.25727
[20] train's multi_logloss: 1.59958 valid's multi_logloss: 2.02153
[30] train's multi_logloss: 1.35119 valid's multi_logloss: 1.86708
[40] train's multi_logloss: 1.16689 valid's multi_logloss: 1.75753
[50] train's multi_logloss: 1.02479 valid's multi_logloss: 1.68036
[60] train's multi_logloss: 0.910429 valid's multi_logloss: 1.62009
[70] train's multi_logloss: 0.816233 valid's multi_logloss: 1.57191
[80] train's multi_logloss: 0.736543 valid's multi_logloss: 1.5361
[90] train's multi_logloss: 0.669172 valid's multi_logloss: 1.50663
[100] train's multi_logloss: 0.609704 valid's multi_logloss: 1.48064
[110] train's multi_logloss: 0.558628 valid's multi_logloss: 1.46044
[120] train's multi_logloss: 0.512906 valid's multi_logloss: 1.44322
[130] train's multi_logloss: 0.472484 valid's multi_logloss: 1.42878
[140] train's multi_logloss: 0.436193 valid's multi_logloss: 1.41618
[150] train's multi_logloss: 0.403753 valid's multi_logloss: 1.40658
[160] train's multi_logloss: 0.374813 valid's multi_logloss: 1.39895
[170] train's multi_logloss: 0.348176 valid's multi_logloss: 1.39274
[180] train's multi_logloss: 0.323957 valid's multi_logloss: 1.38793
[190] train's multi_logloss: 0.301372 valid's multi_logloss: 1.38243
[200] train's multi_logloss: 0.280868 valid's multi_logloss: 1.37962
[210] train's multi_logloss: 0.262028 valid's multi_logloss: 1.37695
[220] train's multi_logloss: 0.244763 valid's multi_logloss: 1.37377
[230] train's multi_logloss: 0.22878 valid's multi_logloss: 1.37237
[240] train's multi_logloss: 0.213856 valid's multi_logloss: 1.3695
[250] train's multi_logloss: 0.200126 valid's multi_logloss: 1.36752
[260] train's multi_logloss: 0.187443 valid's multi_logloss: 1.36685
[270] train's multi_logloss: 0.17583 valid's multi_logloss: 1.36611
[280] train's multi_logloss: 0.164866 valid's multi_logloss: 1.36565
[290] train's multi_logloss: 0.154636 valid's multi_logloss: 1.36497
[300] train's multi_logloss: 0.145243 valid's multi_logloss: 1.36529
[310] train's multi_logloss: 0.136437 valid's multi_logloss: 1.36528
[320] train's multi_logloss: 0.128022 valid's multi_logloss: 1.36469
[330] train's multi_logloss: 0.120189 valid's multi_logloss: 1.36553
[340] train's multi_logloss: 0.112871 valid's multi_logloss: 1.36501
[350] train's multi_logloss: 0.106237 valid's multi_logloss: 1.36475
[360] train's multi_logloss: 0.099948 valid's multi_logloss: 1.36427
[370] train's multi_logloss: 0.0939851 valid's multi_logloss: 1.36339
[380] train's multi_logloss: 0.0884457 valid's multi_logloss: 1.36362
[390] train's multi_logloss: 0.0832374 valid's multi_logloss: 1.36339
[400] train's multi_logloss: 0.0784496 valid's multi_logloss: 1.36293
[410] train's multi_logloss: 0.0739829 valid's multi_logloss: 1.36318
[420] train's multi_logloss: 0.0697228 valid's multi_logloss: 1.36373
[430] train's multi_logloss: 0.0657348 valid's multi_logloss: 1.36483
[440] train's multi_logloss: 0.0620008 valid's multi_logloss: 1.36546
[450] train's multi_logloss: 0.0585393 valid's multi_logloss: 1.36562
[460] train's multi_logloss: 0.0553168 valid's multi_logloss: 1.36589
[470] train's multi_logloss: 0.0522576 valid's multi_logloss: 1.36619
[480] train's multi_logloss: 0.0493674 valid's multi_logloss: 1.36714
[490] train's multi_logloss: 0.0466461 valid's multi_logloss: 1.36911
[500] train's multi_logloss: 0.0440682 valid's multi_logloss: 1.36884
[510] train's multi_logloss: 0.0416788 valid's multi_logloss: 1.37059
Early stopping, best iteration is:
[417] train's multi_logloss: 0.0709792 valid's multi_logloss: 1.36282
Fold 5 training completed. Best iteration: 417
Fold 5 - Competition Score: 0.7111 (Binary F1: 0.9539, Macro F1: 0.4684)

============================================================
CROSS-VALIDATION RESULTS
============================================================
Overall Competition Score: 0.7182 ± 0.0199
Overall Binary F1: 0.9583
Overall Macro F1: 0.4781
Fold scores: ['0.7580', '0.7111', '0.7100', '0.7028', '0.7111']
============================================================

============================================================
EVALUATION SUMMARY
============================================================
Competition Score: 0.7182

- Binary F1 (BFRB vs non-BFRB): 0.9583
- Macro F1 (BFRB classes): 0.4781
  Overall Accuracy: 0.5561

BFRB vs Non-BFRB:

- True BFRB: 5113
- Predicted BFRB: 5264
- Correct BFRB: 4972
- # Correct Non-BFRB: 2746

============================================================
STEP 5: Saving Results
============================================================
✓ Saved 5 models to results/run_20250812_224240/models

Top 20 Most Important Features:
feature importance
acc_y_seg3_std 1757.6
acc_x_seg3_std 1501.8
acc_z_seg3_std 1177.4
acc_y_seg3_mean 1164.0
acc_z_min 1016.2
acc_y_max 972.0
rot_w_n_changes 928.6
acc_magnitude_seg3_std 926.0
rot_z_n_changes 875.0
rot_x_seg3_mean 815.6
rot_z_seg3_std 813.2
rot_x_seg3_std 774.2
rot_x_n_changes 769.4
acc_y_diff_std 764.6
rot_w_seg3_std 746.8
acc_y_seg2_to_seg3 746.0
acc_z_seg3_mean 736.4
acc_x_std 727.4
acc_y_q75 727.0
acc_y_last 725.8

✓ Results saved to results/run_20250812_224240

============================================================
STEP 6: Generating Test Predictions
============================================================
✓ Submission saved to results/run_20250812_224240/submission.csv

============================================================
TRAINING COMPLETE
============================================================
✓ Final CV Score: 0.7186 ± 0.0199
✓ Results saved to: results/run_20250812_224240
✓ Models trained: 5
✓ Test predictions generated: 2
============================================================
