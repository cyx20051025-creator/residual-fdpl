# Cross-endpoint calibration diagnostic

A 117-statistic Ridge map is fitted on 16 calibration scenes from the
calibrator checkpoint and applied to 13 held-out scenes from the
recipient checkpoint. Scores are exact pre-clipping PSNR changes from
the cached base MSE and reference offsets. They do not include clipping
or perceptual metrics.

| Mode | Calibrator | Recipient | ΔPSNR [95% split interval] | Positive | Worst | Mean offset norm |
|---|---:|---:|---:|---:|---:|---:|
| capped | NAFNet-ft62 | NAFNet-ft62 | +0.6638 [+0.6046, +0.7375] | 10/10 | +0.4978 | 0.0168 |
| uncapped | NAFNet-ft62 | NAFNet-ft62 | +1.2255 [+1.1298, +1.3256] | 10/10 | +1.0039 | 0.0509 |
| capped | NAFNet-ft62 | NAFNet-ft63 | +0.6782 [+0.6180, +0.7495] | 10/10 | +0.5143 | 0.0169 |
| uncapped | NAFNet-ft62 | NAFNet-ft63 | +1.2473 [+1.1494, +1.3475] | 10/10 | +1.0237 | 0.0510 |
| capped | NAFNet-ft62 | NAFNet-ft64 | +0.6476 [+0.5874, +0.7224] | 10/10 | +0.4658 | 0.0167 |
| uncapped | NAFNet-ft62 | NAFNet-ft64 | +1.2201 [+1.1210, +1.3206] | 10/10 | +0.9839 | 0.0510 |
| capped | NAFNet-ft62 | NAFNet-official | -0.2158 [-0.3793, -0.0470] | 2/10 | -0.6648 | 0.0131 |
| uncapped | NAFNet-ft62 | NAFNet-official | -0.0144 [-0.2514, +0.2139] | 4/10 | -0.7068 | 0.0243 |
| capped | NAFNet-ft62 | SwinIR-M | +0.3580 [+0.2879, +0.4299] | 10/10 | +0.1614 | 0.0149 |
| uncapped | NAFNet-ft62 | SwinIR-M | +0.8480 [+0.7636, +0.9371] | 10/10 | +0.6036 | 0.0443 |
| capped | NAFNet-ft62 | Restormer | +0.0898 [-0.1773, +0.3480] | 5/10 | -0.6895 | 0.0140 |
| uncapped | NAFNet-ft62 | Restormer | +0.3887 [+0.0209, +0.7356] | 7/10 | -0.7112 | 0.0270 |
| capped | NAFNet-ft63 | NAFNet-ft62 | +0.6640 [+0.6048, +0.7370] | 10/10 | +0.4994 | 0.0168 |
| uncapped | NAFNet-ft63 | NAFNet-ft62 | +1.2237 [+1.1291, +1.3220] | 10/10 | +1.0017 | 0.0510 |
| capped | NAFNet-ft63 | NAFNet-ft63 | +0.6784 [+0.6194, +0.7506] | 10/10 | +0.5160 | 0.0169 |
| uncapped | NAFNet-ft63 | NAFNet-ft63 | +1.2456 [+1.1505, +1.3471] | 10/10 | +1.0214 | 0.0511 |
| capped | NAFNet-ft63 | NAFNet-ft64 | +0.6479 [+0.5841, +0.7193] | 10/10 | +0.4672 | 0.0167 |
| uncapped | NAFNet-ft63 | NAFNet-ft64 | +1.2184 [+1.1194, +1.3146] | 10/10 | +0.9819 | 0.0511 |
| capped | NAFNet-ft63 | NAFNet-official | -0.2005 [-0.3710, -0.0304] | 2/10 | -0.6681 | 0.0131 |
| uncapped | NAFNet-ft63 | NAFNet-official | +0.0048 [-0.2302, +0.2323] | 5/10 | -0.7118 | 0.0245 |
| capped | NAFNet-ft63 | SwinIR-M | +0.3571 [+0.2838, +0.4283] | 10/10 | +0.1579 | 0.0149 |
| uncapped | NAFNet-ft63 | SwinIR-M | +0.8450 [+0.7564, +0.9329] | 10/10 | +0.5970 | 0.0443 |
| capped | NAFNet-ft63 | Restormer | +0.1006 [-0.1734, +0.3669] | 5/10 | -0.7243 | 0.0141 |
| uncapped | NAFNet-ft63 | Restormer | +0.4036 [+0.0435, +0.7656] | 7/10 | -0.7568 | 0.0273 |
| capped | NAFNet-ft64 | NAFNet-ft62 | +0.6639 [+0.6052, +0.7346] | 10/10 | +0.5055 | 0.0169 |
| uncapped | NAFNet-ft64 | NAFNet-ft62 | +1.2228 [+1.1289, +1.3235] | 10/10 | +1.0061 | 0.0511 |
| capped | NAFNet-ft64 | NAFNet-ft63 | +0.6784 [+0.6219, +0.7487] | 10/10 | +0.5221 | 0.0169 |
| uncapped | NAFNet-ft64 | NAFNet-ft63 | +1.2447 [+1.1470, +1.3450] | 10/10 | +1.0259 | 0.0512 |
| capped | NAFNet-ft64 | NAFNet-ft64 | +0.6478 [+0.5864, +0.7235] | 10/10 | +0.4736 | 0.0167 |
| uncapped | NAFNet-ft64 | NAFNet-ft64 | +1.2178 [+1.1188, +1.3175] | 10/10 | +0.9867 | 0.0512 |
| capped | NAFNet-ft64 | NAFNet-official | -0.2097 [-0.3635, -0.0459] | 2/10 | -0.6438 | 0.0131 |
| uncapped | NAFNet-ft64 | NAFNet-official | -0.0041 [-0.2309, +0.2248] | 4/10 | -0.6797 | 0.0244 |
| capped | NAFNet-ft64 | SwinIR-M | +0.3569 [+0.2895, +0.4308] | 10/10 | +0.1634 | 0.0150 |
| uncapped | NAFNet-ft64 | SwinIR-M | +0.8466 [+0.7605, +0.9367] | 10/10 | +0.6036 | 0.0444 |
| capped | NAFNet-ft64 | Restormer | +0.1061 [-0.1623, +0.3594] | 5/10 | -0.6835 | 0.0140 |
| uncapped | NAFNet-ft64 | Restormer | +0.4092 [+0.0590, +0.7479] | 7/10 | -0.7043 | 0.0270 |
| capped | NAFNet-official | NAFNet-ft62 | +0.6428 [+0.5940, +0.7073] | 10/10 | +0.5511 | 0.0182 |
| uncapped | NAFNet-official | NAFNet-ft62 | +0.6030 [+0.2381, +0.9584] | 8/10 | -0.3921 | 0.0690 |
| capped | NAFNet-official | NAFNet-ft63 | +0.6584 [+0.6123, +0.7223] | 10/10 | +0.5695 | 0.0183 |
| uncapped | NAFNet-official | NAFNet-ft63 | +0.6319 [+0.2824, +0.9698] | 8/10 | -0.3428 | 0.0690 |
| capped | NAFNet-official | NAFNet-ft64 | +0.6256 [+0.5764, +0.6864] | 10/10 | +0.5279 | 0.0181 |
| uncapped | NAFNet-official | NAFNet-ft64 | +0.6143 [+0.2603, +0.9603] | 8/10 | -0.3362 | 0.0690 |
| capped | NAFNet-official | NAFNet-official | +0.1387 [+0.0364, +0.2373] | 8/10 | -0.1367 | 0.0129 |
| uncapped | NAFNet-official | NAFNet-official | +0.4495 [+0.3466, +0.5560] | 10/10 | +0.2009 | 0.0318 |
| capped | NAFNet-official | SwinIR-M | +0.3607 [+0.2818, +0.4460] | 10/10 | +0.1476 | 0.0161 |
| uncapped | NAFNet-official | SwinIR-M | +0.6235 [+0.4752, +0.7691] | 10/10 | +0.2086 | 0.0611 |
| capped | NAFNet-official | Restormer | +0.3898 [+0.1255, +0.6350] | 7/10 | -0.2761 | 0.0150 |
| uncapped | NAFNet-official | Restormer | +0.8345 [+0.5042, +1.1515] | 9/10 | -0.0827 | 0.0367 |
| capped | SwinIR-M | NAFNet-ft62 | +0.6215 [+0.5546, +0.7035] | 10/10 | +0.4785 | 0.0185 |
| uncapped | SwinIR-M | NAFNet-ft62 | +0.5894 [+0.3548, +0.8270] | 9/10 | -0.0551 | 0.0667 |
| capped | SwinIR-M | NAFNet-ft63 | +0.6380 [+0.5700, +0.7187] | 10/10 | +0.4962 | 0.0185 |
| uncapped | SwinIR-M | NAFNet-ft63 | +0.6077 [+0.3612, +0.8690] | 9/10 | -0.0360 | 0.0670 |
| capped | SwinIR-M | NAFNet-ft64 | +0.6011 [+0.5281, +0.6849] | 10/10 | +0.4402 | 0.0184 |
| uncapped | SwinIR-M | NAFNet-ft64 | +0.5947 [+0.3471, +0.8404] | 9/10 | -0.0577 | 0.0669 |
| capped | SwinIR-M | NAFNet-official | -0.2126 [-0.3182, -0.0935] | 2/10 | -0.4269 | 0.0143 |
| uncapped | SwinIR-M | NAFNet-official | -0.1176 [-0.2965, +0.0794] | 3/10 | -0.5116 | 0.0290 |
| capped | SwinIR-M | SwinIR-M | +0.3848 [+0.3119, +0.4551] | 10/10 | +0.1622 | 0.0155 |
| uncapped | SwinIR-M | SwinIR-M | +0.8909 [+0.8087, +0.9671] | 10/10 | +0.6441 | 0.0517 |
| capped | SwinIR-M | Restormer | +0.3366 [+0.1229, +0.5265] | 8/10 | -0.3363 | 0.0159 |
| uncapped | SwinIR-M | Restormer | +0.5427 [+0.2311, +0.8195] | 9/10 | -0.5411 | 0.0317 |
| capped | Restormer | NAFNet-ft62 | +0.5971 [+0.5027, +0.6880] | 10/10 | +0.2939 | 0.0176 |
| uncapped | Restormer | NAFNet-ft62 | +1.0021 [+0.8727, +1.1370] | 10/10 | +0.6829 | 0.0533 |
| capped | Restormer | NAFNet-ft63 | +0.6110 [+0.5130, +0.7029] | 10/10 | +0.3051 | 0.0176 |
| uncapped | Restormer | NAFNet-ft63 | +1.0226 [+0.8932, +1.1559] | 10/10 | +0.7020 | 0.0534 |
| capped | Restormer | NAFNet-ft64 | +0.5825 [+0.4906, +0.6731] | 10/10 | +0.2887 | 0.0175 |
| uncapped | Restormer | NAFNet-ft64 | +1.0003 [+0.8691, +1.1353] | 10/10 | +0.6918 | 0.0535 |
| capped | Restormer | NAFNet-official | +0.0246 [-0.1454, +0.1590] | 6/10 | -0.5940 | 0.0149 |
| uncapped | Restormer | NAFNet-official | +0.2474 [+0.0452, +0.4147] | 9/10 | -0.4863 | 0.0354 |
| capped | Restormer | SwinIR-M | +0.2926 [+0.1937, +0.3919] | 9/10 | -0.0013 | 0.0175 |
| uncapped | Restormer | SwinIR-M | +0.7086 [+0.5663, +0.8437] | 10/10 | +0.2945 | 0.0496 |
| capped | Restormer | Restormer | +0.9175 [+0.8350, +1.0165] | 10/10 | +0.7647 | 0.0170 |
| uncapped | Restormer | Restormer | +1.3484 [+1.2412, +1.4730] | 10/10 | +1.0470 | 0.0430 |

Same-checkpoint cells are the reference fits. The cross-checkpoint
cells test whether a calibration map learned from one frozen
endpoint supplies the right image-statistic mapping for another.
All three endpoints reuse the same scenes and splits and are not
independent populations.
