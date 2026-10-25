# Quantitative Kinematic Distribution Similarity Report (0 to 5 Hours)

## 1. Executive Summary
This report assesses the mathematical and physical similarity between regime-matched experimental live-cell tracking datasets and GPU simulation trajectories restricted strictly to the **first 5 hours of tracking** (`FRAME <= 1800`).

### Quality-Control Filtering Applied:
- **Cancer Cell Tracks**: Retains entire trajectories only if total tracking duration $\ge 900$ s (15 minutes).
- **Immune / T-Cell Tracks**: Retains entire trajectories only if total tracking duration $\ge 3600$ s (1 hour).
- **Time Window**: Restricts downstream statistical evaluations to $0 \le \text{FRAME} \le 1800$.

### Key Findings (0 - 5 Hours):
- **Best Overall Agreement**: **Experimental Non-Killing T-Cell vs Non-Killing T-Cell** (Composite Index = **0.062**).
- **Worst Overall Agreement**: **Experimental Non-Killing Cancer vs Non-Killing Cancer** (Composite Index = **0.196**).
- **Most Accurately Reproduced Category**: **Local Motility** (Mean Category Composite = **0.028**).
- **Top Calibration Targets**: `PATH_EFFICIENCY` (Mean Composite = **0.518**) and `DISPLACEMENT_FROM_ORIGIN` (Mean Composite = **0.500**).

---

## 2. Dataset Information and Observational Metadata (0-5 hrs)
| Dataset Description | Analyzed Observations | Unique Retained Tracks | Obs / Track |
| :--- | :--- | :--- | :--- |
| Experimental Killing Cancer | 504,835 | 5,575 | 90.6 |
| Experimental Non-Killing Cancer | 408,262 | 2,979 | 137.0 |
| Experimental Killing T-Cell | 282,489 | 2,220 | 127.2 |
| Experimental Non-Killing T-Cell | 357,159 | 3,971 | 89.9 |
| Killing Cancer | 6,335 | 101 | 62.7 |
| Killing T-Cell | 10,050 | 73 | 137.7 |
| Non-Killing Cancer | 21,308 | 166 | 128.4 |
| Non-Killing T-Cell | 5,103 | 54 | 94.5 |


---

## 3. Overall Comparison Summary (0 - 5 Hours)
| Comparison                                            |   Overall_Composite_Score |   Spatial_Behavior |   Local_Motility |   Velocity |   Accumulated_Motility |   Long_Term_Motility |   Rank |
|:------------------------------------------------------|--------------------------:|-------------------:|-----------------:|-----------:|-----------------------:|---------------------:|-------:|
| Experimental Non-Killing T-Cell vs Non-Killing T-Cell |                 0.0622861 |           0.125471 |       0.00849057 | 0.00980638 |              0.0344926 |             0.13317  |      1 |
| Experimental Killing T-Cell vs Killing T-Cell         |                 0.103278  |           0.229162 |       0.0153361  | 0.0158714  |              0.0454558 |             0.210562 |      2 |
| Experimental Killing Cancer vs Killing Cancer         |                 0.179342  |           0.28075  |       0.04142    | 0.0401956  |              0.133446  |             0.400897 |      3 |
| Experimental Non-Killing Cancer vs Non-Killing Cancer |                 0.195868  |           0.427171 |       0.0462243  | 0.0461801  |              0.0685495 |             0.391213 |      4 |

---

## 4. Feature Category Similarity Matrix
|                      |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| Spatial Behavior     |                                       0.28075   |                                               0.427171  |                                       0.229162  |                                              0.125471   |
| Local Motility       |                                       0.04142   |                                               0.0462243 |                                       0.0153361 |                                              0.00849057 |
| Velocity             |                                       0.0401956 |                                               0.0461801 |                                       0.0158714 |                                              0.00980638 |
| Accumulated Motility |                                       0.133446  |                                               0.0685495 |                                       0.0454558 |                                              0.0344926  |
| Long-Term Motility   |                                       0.400897  |                                               0.391213  |                                       0.210562  |                                              0.13317    |

---

## 5. Primary Physical Realism Diagnostics (Raw Physical Units)
### Experimental Killing Cancer vs Killing Cancer
- **Largest Spatial Discrepancy (Wasserstein)**: `POSITION_Y` with **84.502 µm** (Exp Mean: 758.168 µm vs Sim Mean: 839.434 µm).
- **Largest Velocity Discrepancy (Wasserstein)**: `SPEED` with **0.010 µm/s** (Exp Mean: 0.052 µm/s vs Sim Mean: 0.043 µm/s).

### Experimental Non-Killing Cancer vs Non-Killing Cancer
- **Largest Spatial Discrepancy (Wasserstein)**: `POSITION_X` with **164.616 µm** (Exp Mean: 712.280 µm vs Sim Mean: 876.875 µm).
- **Largest Velocity Discrepancy (Wasserstein)**: `AVERAGE_SPEED` with **0.019 µm/s** (Exp Mean: 0.047 µm/s vs Sim Mean: 0.054 µm/s).

### Experimental Killing T-Cell vs Killing T-Cell
- **Largest Spatial Discrepancy (Wasserstein)**: `DISTANCE_TRAVELED` with **94.988 µm** (Exp Mean: 443.997 µm vs Sim Mean: 535.660 µm).
- **Largest Velocity Discrepancy (Wasserstein)**: `SPEED` with **0.008 µm/s** (Exp Mean: 0.086 µm/s vs Sim Mean: 0.086 µm/s).

### Experimental Non-Killing T-Cell vs Non-Killing T-Cell
- **Largest Spatial Discrepancy (Wasserstein)**: `POSITION_X` with **101.698 µm** (Exp Mean: 745.943 µm vs Sim Mean: 840.851 µm).
- **Largest Velocity Discrepancy (Wasserstein)**: `AVERAGE_SPEED` with **0.030 µm/s** (Exp Mean: 0.118 µm/s vs Sim Mean: 0.128 µm/s).



---

## 6. Feature Rankings by Regime Comparison (Best Match to Worst Match)

### Experimental Killing Cancer vs Killing Cancer
|   Rank | Feature                          | Units         |   Composite_Score |   Raw_Wasserstein |   Std_Wasserstein |    Std_KS |   Std_JSD |
|-------:|:---------------------------------|:--------------|------------------:|------------------:|------------------:|----------:|----------:|
|      1 | VEL_X                            | µm/s          |         0.0220358 |        0.00535553 |         0.0894001 | 0.0350895 | 0.0143845 |
|      2 | DX_FROM_PREVIOUS_POINT           | µm            |         0.0228788 |        0.0554052  |         0.091374  | 0.0357606 | 0.0150798 |
|      3 | VEL_Y                            | µm/s          |         0.0265584 |        0.00590287 |         0.0964645 | 0.0423725 | 0.0156482 |
|      4 | DY_FROM_PREVIOUS_POINT           | µm            |         0.0270602 |        0.0613298  |         0.0988475 | 0.0430707 | 0.0157821 |
|      5 | SPEED                            | µm/s          |         0.0719928 |        0.00963632 |         0.141027  | 0.120055  | 0.0265006 |
|      6 | DISPLACEMENT_FROM_PREVIOUS_POINT | µm            |         0.074321  |        0.0994238  |         0.143331  | 0.121616  | 0.0287999 |
|      7 | POSITION_X                       | µm            |         0.0751119 |       48.972      |         0.107442  | 0.0854594 | 0.0579931 |
|      8 | DT_ACC                           | s             |         0.106323  |     1701          |         0.360245  | 0.172034  | 0.0296095 |
|      9 | POSITION_Y                       | µm            |         0.119533  |       84.5019     |         0.187135  | 0.130766  | 0.088762  |
|     10 | DX_ACC                           | µm            |         0.137863  |       41.9168     |         0.357168  | 0.199943  | 0.057722  |
|     11 | DISTANCE_TRAVELED                | µm            |         0.153897  |       71.3399     |         0.382395  | 0.222255  | 0.0647202 |
|     12 | DY_ACC                           | µm            |         0.156152  |       45.379      |         0.371985  | 0.229188  | 0.0637164 |
|     13 | AVERAGE_SPEED                    | µm/s          |         0.298464  |        0.00831932 |         0.365392  | 0.329074  | 0.208941  |
|     14 | DX_FROM_ORIGIN                   | µm            |         0.320948  |       15.5776     |         2.42636   | 0.223093  | 0.209269  |
|     15 | DY_FROM_ORIGIN                   | µm            |         0.331478  |       12.7746     |         2.09261   | 0.258379  | 0.217745  |
|     16 | DISPLACEMENT_FROM_ORIGIN         | µm            |         0.55668   |       22.5067     |         3.57101   | 0.512197  | 0.298814  |
|     17 | PATH_EFFICIENCY                  | dimensionless |         0.75033   |        0.24125    |         2.3997    | 0.718022  | 0.508264  |

### Experimental Non-Killing Cancer vs Non-Killing Cancer
|   Rank | Feature                          | Units         |   Composite_Score |   Raw_Wasserstein |   Std_Wasserstein |    Std_KS |     Std_JSD |
|-------:|:---------------------------------|:--------------|------------------:|------------------:|------------------:|----------:|------------:|
|      1 | DT_ACC                           | s             |         0         |       94.083      |         0.0182336 | 0.0133754 | 0.000266631 |
|      2 | DX_FROM_PREVIOUS_POINT           | µm            |         0.0222833 |        0.0577338  |         0.111657  | 0.0317625 | 0.0159685   |
|      3 | VEL_X                            | µm/s          |         0.0226014 |        0.00581293 |         0.113306  | 0.0321623 | 0.0160768   |
|      4 | VEL_Y                            | µm/s          |         0.0349413 |        0.00583116 |         0.0988428 | 0.0564442 | 0.0181517   |
|      5 | DY_FROM_PREVIOUS_POINT           | µm            |         0.0350258 |        0.0585987  |         0.0986413 | 0.056776  | 0.0180521   |
|      6 | POSITION_Y                       | µm            |         0.0593575 |       26.3356     |         0.058306  | 0.0581222 | 0.0563223   |
|      7 | SPEED                            | µm/s          |         0.0809976 |        0.0101761  |         0.173086  | 0.122905  | 0.0364528   |
|      8 | DISPLACEMENT_FROM_PREVIOUS_POINT | µm            |         0.0813638 |        0.101486   |         0.171     | 0.123756  | 0.0365089   |
|      9 | DY_ACC                           | µm            |         0.0933745 |       68.1848     |         0.429385  | 0.0996744 | 0.0583406   |
|     10 | DISTANCE_TRAVELED                | µm            |         0.102621  |      107.402      |         0.468145  | 0.114078  | 0.0599728   |
|     11 | DX_ACC                           | µm            |         0.112274  |       71.1035     |         0.526989  | 0.13181   | 0.0587502   |
|     12 | POSITION_X                       | µm            |         0.137643  |      164.616      |         0.382562  | 0.192728  | 0.0612268   |
|     13 | AVERAGE_SPEED                    | µm/s          |         0.366665  |        0.0193485  |         0.942643  | 0.320198  | 0.288373    |
|     14 | DX_FROM_ORIGIN                   | µm            |         0.513692  |       32.0617     |         5.41006   | 0.291571  | 0.293894    |
|     15 | DY_FROM_ORIGIN                   | µm            |         0.564085  |       36.9446     |         6.34108   | 0.29039   | 0.321698    |
|     16 | PATH_EFFICIENCY                  | dimensionless |         0.704353  |        0.171528   |         2.26604   | 0.702275  | 0.456703    |
|     17 | DISPLACEMENT_FROM_ORIGIN         | µm            |         0.861076  |       53.4404     |         9.5065    | 0.59958   | 0.381932    |

### Experimental Killing T-Cell vs Killing T-Cell
|   Rank | Feature                          | Units         |   Composite_Score |   Raw_Wasserstein |   Std_Wasserstein |    Std_KS |    Std_JSD |
|-------:|:---------------------------------|:--------------|------------------:|------------------:|------------------:|----------:|-----------:|
|      1 | DY_FROM_PREVIOUS_POINT           | µm            |        0.00642697 |        0.044011   |         0.0524251 | 0.0163528 | 0.00608416 |
|      2 | VEL_Y                            | µm/s          |        0.00654716 |        0.00435752 |         0.0523838 | 0.0174252 | 0.00549644 |
|      3 | DX_FROM_PREVIOUS_POINT           | µm            |        0.0124885  |        0.049534   |         0.0601009 | 0.0277437 | 0.00669895 |
|      4 | VEL_X                            | µm/s          |        0.0129798  |        0.004953   |         0.060635  | 0.0289172 | 0.00657319 |
|      5 | DT_ACC                           | s             |        0.0189771  |      356.393      |         0.0819952 | 0.046378  | 0.00198132 |
|      6 | DISPLACEMENT_FROM_PREVIOUS_POINT | µm            |        0.027093   |        0.0816104  |         0.103106  | 0.0531969 | 0.00830372 |
|      7 | SPEED                            | µm/s          |        0.0280871  |        0.00809729 |         0.103421  | 0.0560422 | 0.00775065 |
|      8 | DY_ACC                           | µm            |        0.0537964  |       57.6007     |         0.243529  | 0.0875567 | 0.0167105  |
|      9 | DISTANCE_TRAVELED                | µm            |        0.0584077  |       94.9883     |         0.259836  | 0.095234  | 0.0173303  |
|     10 | DX_ACC                           | µm            |        0.0635939  |       64.3946     |         0.279028  | 0.103635  | 0.0181498  |
|     11 | POSITION_Y                       | µm            |        0.0663041  |       45.8382     |         0.103635  | 0.0873873 | 0.0433842  |
|     12 | POSITION_X                       | µm            |        0.0978413  |       51.6985     |         0.115029  | 0.102122  | 0.0802137  |
|     13 | AVERAGE_SPEED                    | µm/s          |        0.106323   |        0.00683661 |         0.212053  | 0.118365  | 0.076236   |
|     14 | DX_FROM_ORIGIN                   | µm            |        0.261868   |       24.5254     |         0.910322  | 0.274699  | 0.163195   |
|     15 | DY_FROM_ORIGIN                   | µm            |        0.274821   |       29.7032     |         1.20126   | 0.230988  | 0.19887    |
|     16 | DISPLACEMENT_FROM_ORIGIN         | µm            |        0.444976   |       41.8279     |         1.35107   | 0.513144  | 0.246751   |
|     17 | PATH_EFFICIENCY                  | dimensionless |        0.466955   |        0.105972   |         1.09242   | 0.535505  | 0.277975   |

### Experimental Non-Killing T-Cell vs Non-Killing T-Cell
|   Rank | Feature                          | Units         |   Composite_Score |   Raw_Wasserstein |   Std_Wasserstein |    Std_KS |    Std_JSD |
|-------:|:---------------------------------|:--------------|------------------:|------------------:|------------------:|----------:|-----------:|
|      1 | DX_FROM_PREVIOUS_POINT           | µm            |        0.00580836 |        0.0566669  |         0.0533034 | 0.015822  | 0.00547712 |
|      2 | VEL_X                            | µm/s          |        0.007207   |        0.00663399 |         0.0640379 | 0.0175904 | 0.00575898 |
|      3 | VEL_Y                            | µm/s          |        0.00934104 |        0.00631439 |         0.056304  | 0.0234114 | 0.00522885 |
|      4 | DY_FROM_PREVIOUS_POINT           | µm            |        0.00979091 |        0.0612245  |         0.0521989 | 0.0251091 | 0.0049103  |
|      5 | DISPLACEMENT_FROM_PREVIOUS_POINT | µm            |        0.00987244 |        0.0642024  |         0.0628144 | 0.0251994 | 0.0044011  |
|      6 | SPEED                            | µm/s          |        0.0128711  |        0.0073348  |         0.076719  | 0.0295993 | 0.00505454 |
|      7 | DT_ACC                           | s             |        0.0185871  |      305.708      |         0.103296  | 0.0355535 | 0.00805021 |
|      8 | DX_ACC                           | µm            |        0.0374961  |       36.7755     |         0.187134  | 0.0592363 | 0.0153053  |
|      9 | DISTANCE_TRAVELED                | µm            |        0.0375982  |       60.9966     |         0.185681  | 0.0573644 | 0.0168881  |
|     10 | DY_ACC                           | µm            |        0.0473945  |       41.1704     |         0.182539  | 0.0763218 | 0.0183189  |
|     11 | POSITION_Y                       | µm            |        0.0667091  |       56.2072     |         0.127847  | 0.0682581 | 0.0564958  |
|     12 | DX_FROM_ORIGIN                   | µm            |        0.131283   |       10.3119     |         0.280833  | 0.196965  | 0.053927   |
|     13 | DISPLACEMENT_FROM_ORIGIN         | µm            |        0.138218   |        8.78574    |         0.226381  | 0.209154  | 0.0586236  |
|     14 | DY_FROM_ORIGIN                   | µm            |        0.141343   |        9.72754    |         0.277308  | 0.203163  | 0.0649787  |
|     15 | POSITION_X                       | µm            |        0.149801   |      101.698      |         0.233638  | 0.183379  | 0.0944693  |
|     16 | PATH_EFFICIENCY                  | dimensionless |        0.150615   |        0.0496465  |         0.37778   | 0.227063  | 0.056499   |
|     17 | AVERAGE_SPEED                    | µm/s          |        0.211298   |        0.0300477  |         0.571829  | 0.19831   | 0.159319   |


---

## 7. Raw Physical Unit Metric Tables

### Raw Wasserstein Distance ($W_1$ in physical units)
|                                  |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| POSITION_X                       |                                     48.972      |                                            164.616      |                                     51.6985     |                                            101.698      |
| POSITION_Y                       |                                     84.5019     |                                             26.3356     |                                     45.8382     |                                             56.2072     |
| DX_FROM_PREVIOUS_POINT           |                                      0.0554052  |                                              0.0577338  |                                      0.049534   |                                              0.0566669  |
| DY_FROM_PREVIOUS_POINT           |                                      0.0613298  |                                              0.0585987  |                                      0.044011   |                                              0.0612245  |
| DISPLACEMENT_FROM_PREVIOUS_POINT |                                      0.0994238  |                                              0.101486   |                                      0.0816104  |                                              0.0642024  |
| DX_FROM_ORIGIN                   |                                     15.5776     |                                             32.0617     |                                     24.5254     |                                             10.3119     |
| DY_FROM_ORIGIN                   |                                     12.7746     |                                             36.9446     |                                     29.7032     |                                              9.72754    |
| DISPLACEMENT_FROM_ORIGIN         |                                     22.5067     |                                             53.4404     |                                     41.8279     |                                              8.78574    |
| DX_ACC                           |                                     41.9168     |                                             71.1035     |                                     64.3946     |                                             36.7755     |
| DY_ACC                           |                                     45.379      |                                             68.1848     |                                     57.6007     |                                             41.1704     |
| DT_ACC                           |                                   1701          |                                             94.083      |                                    356.393      |                                            305.708      |
| DISTANCE_TRAVELED                |                                     71.3399     |                                            107.402      |                                     94.9883     |                                             60.9966     |
| PATH_EFFICIENCY                  |                                      0.24125    |                                              0.171528   |                                      0.105972   |                                              0.0496465  |
| VEL_X                            |                                      0.00535553 |                                              0.00581293 |                                      0.004953   |                                              0.00663399 |
| VEL_Y                            |                                      0.00590287 |                                              0.00583116 |                                      0.00435752 |                                              0.00631439 |
| SPEED                            |                                      0.00963632 |                                              0.0101761  |                                      0.00809729 |                                              0.0073348  |
| AVERAGE_SPEED                    |                                      0.00831932 |                                              0.0193485  |                                      0.00683661 |                                              0.0300477  |

### Raw Kolmogorov-Smirnov Statistic ($D_{\text{KS}}$)
|                                  |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| POSITION_X                       |                                       0.0854594 |                                               0.192728  |                                       0.102122  |                                               0.183379  |
| POSITION_Y                       |                                       0.130766  |                                               0.0581222 |                                       0.0873873 |                                               0.0682581 |
| DX_FROM_PREVIOUS_POINT           |                                       0.0357606 |                                               0.0317625 |                                       0.0277437 |                                               0.015822  |
| DY_FROM_PREVIOUS_POINT           |                                       0.0430707 |                                               0.056776  |                                       0.0163528 |                                               0.0251091 |
| DISPLACEMENT_FROM_PREVIOUS_POINT |                                       0.121616  |                                               0.123756  |                                       0.0531969 |                                               0.0251994 |
| DX_FROM_ORIGIN                   |                                       0.223093  |                                               0.291571  |                                       0.274699  |                                               0.196965  |
| DY_FROM_ORIGIN                   |                                       0.258379  |                                               0.29039   |                                       0.230988  |                                               0.203163  |
| DISPLACEMENT_FROM_ORIGIN         |                                       0.512197  |                                               0.59958   |                                       0.513144  |                                               0.209154  |
| DX_ACC                           |                                       0.199943  |                                               0.13181   |                                       0.103635  |                                               0.0592363 |
| DY_ACC                           |                                       0.229188  |                                               0.0996744 |                                       0.0875567 |                                               0.0763218 |
| DT_ACC                           |                                       0.172034  |                                               0.0133754 |                                       0.046378  |                                               0.0355535 |
| DISTANCE_TRAVELED                |                                       0.222255  |                                               0.114078  |                                       0.095234  |                                               0.0573644 |
| PATH_EFFICIENCY                  |                                       0.718022  |                                               0.702275  |                                       0.535505  |                                               0.227063  |
| VEL_X                            |                                       0.0350895 |                                               0.0321623 |                                       0.0289172 |                                               0.0175904 |
| VEL_Y                            |                                       0.0423725 |                                               0.0564442 |                                       0.0174252 |                                               0.0234114 |
| SPEED                            |                                       0.120055  |                                               0.122905  |                                       0.0560422 |                                               0.0295993 |
| AVERAGE_SPEED                    |                                       0.329074  |                                               0.320198  |                                       0.118365  |                                               0.19831   |

### Raw Jensen-Shannon Divergence ($\text{JSD}$ in bits)
|                                  |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| POSITION_X                       |                                       0.106757  |                                             0.0952554   |                                      0.113163   |                                              0.132823   |
| POSITION_Y                       |                                       0.158794  |                                             0.0743431   |                                      0.0691544  |                                              0.092167   |
| DX_FROM_PREVIOUS_POINT           |                                       0.012065  |                                             0.0142357   |                                      0.00512594 |                                              0.0017291  |
| DY_FROM_PREVIOUS_POINT           |                                       0.0128569 |                                             0.0160657   |                                      0.00436644 |                                              0.00158254 |
| DISPLACEMENT_FROM_PREVIOUS_POINT |                                       0.0276015 |                                             0.0369709   |                                      0.00831306 |                                              0.0019353  |
| DX_FROM_ORIGIN                   |                                       0.204547  |                                             0.283928    |                                      0.158687   |                                              0.0511147  |
| DY_FROM_ORIGIN                   |                                       0.214484  |                                             0.299099    |                                      0.196064   |                                              0.0628679  |
| DISPLACEMENT_FROM_ORIGIN         |                                       0.29482   |                                             0.376768    |                                      0.245676   |                                              0.0593858  |
| DX_ACC                           |                                       0.0575128 |                                             0.0586733   |                                      0.0183035  |                                              0.0148611  |
| DY_ACC                           |                                       0.0638814 |                                             0.0582628   |                                      0.016697   |                                              0.0170541  |
| DT_ACC                           |                                       0.0295239 |                                             0.000280969 |                                      0.00201392 |                                              0.00811612 |
| DISTANCE_TRAVELED                |                                       0.0646347 |                                             0.0599548   |                                      0.0173811  |                                              0.0160485  |
| PATH_EFFICIENCY                  |                                       0.498173  |                                             0.456026    |                                      0.277309   |                                              0.0556764  |
| VEL_X                            |                                       0.0118676 |                                             0.0143438   |                                      0.00513701 |                                              0.00289088 |
| VEL_Y                            |                                       0.0126194 |                                             0.0160301   |                                      0.00429636 |                                              0.0019305  |
| SPEED                            |                                       0.0272928 |                                             0.0370221   |                                      0.00830885 |                                              0.00230055 |
| AVERAGE_SPEED                    |                                       0.17461   |                                             0.266647    |                                      0.0683497  |                                              0.126131   |

---

## 8. Experimental Reference Standardized Z-Score Tables

### Standardized Wasserstein Distance ($W_1$ in $\sigma_{\text{exp}}$ units)
|                                  |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| POSITION_X                       |                                       0.107442  |                                               0.382562  |                                       0.115029  |                                               0.233638  |
| POSITION_Y                       |                                       0.187135  |                                               0.058306  |                                       0.103635  |                                               0.127847  |
| DX_FROM_PREVIOUS_POINT           |                                       0.091374  |                                               0.111657  |                                       0.0601009 |                                               0.0533034 |
| DY_FROM_PREVIOUS_POINT           |                                       0.0988475 |                                               0.0986413 |                                       0.0524251 |                                               0.0521989 |
| DISPLACEMENT_FROM_PREVIOUS_POINT |                                       0.143331  |                                               0.171     |                                       0.103106  |                                               0.0628144 |
| DX_FROM_ORIGIN                   |                                       2.42636   |                                               5.41006   |                                       0.910322  |                                               0.280833  |
| DY_FROM_ORIGIN                   |                                       2.09261   |                                               6.34108   |                                       1.20126   |                                               0.277308  |
| DISPLACEMENT_FROM_ORIGIN         |                                       3.57101   |                                               9.5065    |                                       1.35107   |                                               0.226381  |
| DX_ACC                           |                                       0.357168  |                                               0.526989  |                                       0.279028  |                                               0.187134  |
| DY_ACC                           |                                       0.371985  |                                               0.429385  |                                       0.243529  |                                               0.182539  |
| DT_ACC                           |                                       0.360245  |                                               0.0182336 |                                       0.0819952 |                                               0.103296  |
| DISTANCE_TRAVELED                |                                       0.382395  |                                               0.468145  |                                       0.259836  |                                               0.185681  |
| PATH_EFFICIENCY                  |                                       2.3997    |                                               2.26604   |                                       1.09242   |                                               0.37778   |
| VEL_X                            |                                       0.0894001 |                                               0.113306  |                                       0.060635  |                                               0.0640379 |
| VEL_Y                            |                                       0.0964645 |                                               0.0988428 |                                       0.0523838 |                                               0.056304  |
| SPEED                            |                                       0.141027  |                                               0.173086  |                                       0.103421  |                                               0.076719  |
| AVERAGE_SPEED                    |                                       0.365392  |                                               0.942643  |                                       0.212053  |                                               0.571829  |

### Standardized Kolmogorov-Smirnov Statistic ($D_{\text{KS}}$)
|                                  |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| POSITION_X                       |                                       0.0854594 |                                               0.192728  |                                       0.102122  |                                               0.183379  |
| POSITION_Y                       |                                       0.130766  |                                               0.0581222 |                                       0.0873873 |                                               0.0682581 |
| DX_FROM_PREVIOUS_POINT           |                                       0.0357606 |                                               0.0317625 |                                       0.0277437 |                                               0.015822  |
| DY_FROM_PREVIOUS_POINT           |                                       0.0430707 |                                               0.056776  |                                       0.0163528 |                                               0.0251091 |
| DISPLACEMENT_FROM_PREVIOUS_POINT |                                       0.121616  |                                               0.123756  |                                       0.0531969 |                                               0.0251994 |
| DX_FROM_ORIGIN                   |                                       0.223093  |                                               0.291571  |                                       0.274699  |                                               0.196965  |
| DY_FROM_ORIGIN                   |                                       0.258379  |                                               0.29039   |                                       0.230988  |                                               0.203163  |
| DISPLACEMENT_FROM_ORIGIN         |                                       0.512197  |                                               0.59958   |                                       0.513144  |                                               0.209154  |
| DX_ACC                           |                                       0.199943  |                                               0.13181   |                                       0.103635  |                                               0.0592363 |
| DY_ACC                           |                                       0.229188  |                                               0.0996744 |                                       0.0875567 |                                               0.0763218 |
| DT_ACC                           |                                       0.172034  |                                               0.0133754 |                                       0.046378  |                                               0.0355535 |
| DISTANCE_TRAVELED                |                                       0.222255  |                                               0.114078  |                                       0.095234  |                                               0.0573644 |
| PATH_EFFICIENCY                  |                                       0.718022  |                                               0.702275  |                                       0.535505  |                                               0.227063  |
| VEL_X                            |                                       0.0350895 |                                               0.0321623 |                                       0.0289172 |                                               0.0175904 |
| VEL_Y                            |                                       0.0423725 |                                               0.0564442 |                                       0.0174252 |                                               0.0234114 |
| SPEED                            |                                       0.120055  |                                               0.122905  |                                       0.0560422 |                                               0.0295993 |
| AVERAGE_SPEED                    |                                       0.329074  |                                               0.320198  |                                       0.118365  |                                               0.19831   |

### Standardized Jensen-Shannon Divergence ($\text{JSD}$ in bits)
|                                  |   Experimental Killing Cancer vs Killing Cancer |   Experimental Non-Killing Cancer vs Non-Killing Cancer |   Experimental Killing T-Cell vs Killing T-Cell |   Experimental Non-Killing T-Cell vs Non-Killing T-Cell |
|:---------------------------------|------------------------------------------------:|--------------------------------------------------------:|------------------------------------------------:|--------------------------------------------------------:|
| POSITION_X                       |                                       0.0579931 |                                             0.0612268   |                                      0.0802137  |                                              0.0944693  |
| POSITION_Y                       |                                       0.088762  |                                             0.0563223   |                                      0.0433842  |                                              0.0564958  |
| DX_FROM_PREVIOUS_POINT           |                                       0.0150798 |                                             0.0159685   |                                      0.00669895 |                                              0.00547712 |
| DY_FROM_PREVIOUS_POINT           |                                       0.0157821 |                                             0.0180521   |                                      0.00608416 |                                              0.0049103  |
| DISPLACEMENT_FROM_PREVIOUS_POINT |                                       0.0287999 |                                             0.0365089   |                                      0.00830372 |                                              0.0044011  |
| DX_FROM_ORIGIN                   |                                       0.209269  |                                             0.293894    |                                      0.163195   |                                              0.053927   |
| DY_FROM_ORIGIN                   |                                       0.217745  |                                             0.321698    |                                      0.19887    |                                              0.0649787  |
| DISPLACEMENT_FROM_ORIGIN         |                                       0.298814  |                                             0.381932    |                                      0.246751   |                                              0.0586236  |
| DX_ACC                           |                                       0.057722  |                                             0.0587502   |                                      0.0181498  |                                              0.0153053  |
| DY_ACC                           |                                       0.0637164 |                                             0.0583406   |                                      0.0167105  |                                              0.0183189  |
| DT_ACC                           |                                       0.0296095 |                                             0.000266631 |                                      0.00198132 |                                              0.00805021 |
| DISTANCE_TRAVELED                |                                       0.0647202 |                                             0.0599728   |                                      0.0173303  |                                              0.0168881  |
| PATH_EFFICIENCY                  |                                       0.508264  |                                             0.456703    |                                      0.277975   |                                              0.056499   |
| VEL_X                            |                                       0.0143845 |                                             0.0160768   |                                      0.00657319 |                                              0.00575898 |
| VEL_Y                            |                                       0.0156482 |                                             0.0181517   |                                      0.00549644 |                                              0.00522885 |
| SPEED                            |                                       0.0265006 |                                             0.0364528   |                                      0.00775065 |                                              0.00505454 |
| AVERAGE_SPEED                    |                                       0.208941  |                                             0.288373    |                                      0.076236   |                                              0.159319   |
