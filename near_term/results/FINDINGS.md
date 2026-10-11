# Near-term findings

See the root NEAR_TERM_SUMMARY.txt and near_term/README.md for interpretation and methods.

## including_summaries_sensitivity

| Protocol | Model | Target | Row R2 | Group RMSE (nm) | Group coverage |
|---|---|---|---:|---:|---:|
| grouped_cv | mean | bonded_nm | -0.0133 | 0.1764 | 85.0% |
| grouped_cv | mean | mobile_nm | -0.0341 | 2.5813 | 90.0% |
| grouped_cv | rf_log_interactions | bonded_nm | 0.3837 | 0.1481 | 72.5% |
| grouped_cv | rf_log_interactions | mobile_nm | 0.7503 | 1.2186 | 82.5% |
| grouped_cv | rf_raw | bonded_nm | 0.5274 | 0.1214 | 85.0% |
| grouped_cv | rf_raw | mobile_nm | 0.8913 | 0.8757 | 87.5% |
| grouped_cv | ridge_log_interactions | bonded_nm | 0.4169 | 0.1308 | 77.5% |
| grouped_cv | ridge_log_interactions | mobile_nm | 0.5772 | 1.5545 | 87.5% |
| grouped_cv | ridge_raw | bonded_nm | 0.0539 | 0.1599 | 85.0% |
| grouped_cv | ridge_raw | mobile_nm | 0.0353 | 2.3619 | 92.5% |
| leave_one_solvent_out | mean | bonded_nm | -0.0341 | 0.1798 | 90.0% |
| leave_one_solvent_out | mean | mobile_nm | -0.2342 | 2.7854 | 75.0% |
| leave_one_solvent_out | rf_log_interactions | bonded_nm | -1.1054 | 0.2490 | 62.5% |
| leave_one_solvent_out | rf_log_interactions | mobile_nm | -2.0372 | 3.4836 | 52.5% |
| leave_one_solvent_out | rf_raw | bonded_nm | -0.7835 | 0.2298 | 60.0% |
| leave_one_solvent_out | rf_raw | mobile_nm | -0.5877 | 2.8015 | 50.0% |
| leave_one_solvent_out | ridge_log_interactions | bonded_nm | -1.6312 | 0.2513 | 70.0% |
| leave_one_solvent_out | ridge_log_interactions | mobile_nm | -2.6097 | 3.7924 | 37.5% |
| leave_one_solvent_out | ridge_raw | bonded_nm | -7.1921 | 0.3901 | 57.5% |
| leave_one_solvent_out | ridge_raw | mobile_nm | -14.0336 | 6.3542 | 55.0% |
| ood_high_concentration | mean | bonded_nm | -1.2157 | 0.2813 | 66.7% |
| ood_high_concentration | mean | mobile_nm | -0.3981 | 5.4123 | 33.3% |
| ood_high_concentration | rf_log_interactions | bonded_nm | -0.0049 | 0.1894 | 66.7% |
| ood_high_concentration | rf_log_interactions | mobile_nm | -0.0014 | 4.3382 | 0.0% |
| ood_high_concentration | rf_raw | bonded_nm | 0.1662 | 0.1716 | 66.7% |
| ood_high_concentration | rf_raw | mobile_nm | 0.2827 | 4.0516 | 11.1% |
| ood_high_concentration | ridge_log_interactions | bonded_nm | 0.4777 | 0.1347 | 44.4% |
| ood_high_concentration | ridge_log_interactions | mobile_nm | 0.0128 | 3.8140 | 11.1% |
| ood_high_concentration | ridge_raw | bonded_nm | -54.2255 | 1.0917 | 22.2% |
| ood_high_concentration | ridge_raw | mobile_nm | -25.0285 | 14.0442 | 0.0% |
| ood_low_concentration | mean | bonded_nm | -0.2556 | 0.1309 | 77.8% |
| ood_low_concentration | mean | mobile_nm | -48.2260 | 1.1982 | 100.0% |
| ood_low_concentration | rf_log_interactions | bonded_nm | -0.0476 | 0.1094 | 66.7% |
| ood_low_concentration | rf_log_interactions | mobile_nm | 0.0822 | 0.1504 | 100.0% |
| ood_low_concentration | rf_raw | bonded_nm | -0.3433 | 0.1220 | 66.7% |
| ood_low_concentration | rf_raw | mobile_nm | -0.0115 | 0.1571 | 100.0% |
| ood_low_concentration | ridge_log_interactions | bonded_nm | -0.2423 | 0.1263 | 77.8% |
| ood_low_concentration | ridge_log_interactions | mobile_nm | -29.1161 | 0.8926 | 55.6% |
| ood_low_concentration | ridge_raw | bonded_nm | -0.7561 | 0.1534 | 77.8% |
| ood_low_concentration | ridge_raw | mobile_nm | -115.8320 | 1.7564 | 55.6% |

## primary

| Protocol | Model | Target | Row R2 | Group RMSE (nm) | Group coverage |
|---|---|---|---:|---:|---:|
| grouped_cv | mean | bonded_nm | -0.0727 | 0.1699 | 72.2% |
| grouped_cv | mean | mobile_nm | -0.0193 | 2.6779 | 80.6% |
| grouped_cv | rf_log_interactions | bonded_nm | 0.4115 | 0.1360 | 66.7% |
| grouped_cv | rf_log_interactions | mobile_nm | 0.6127 | 1.4011 | 86.1% |
| grouped_cv | rf_raw | bonded_nm | 0.5678 | 0.1063 | 86.1% |
| grouped_cv | rf_raw | mobile_nm | 0.8790 | 0.9448 | 83.3% |
| grouped_cv | ridge_log_interactions | bonded_nm | 0.3624 | 0.1235 | 75.0% |
| grouped_cv | ridge_log_interactions | mobile_nm | 0.5578 | 1.7156 | 75.0% |
| grouped_cv | ridge_raw | bonded_nm | -0.1865 | 0.1701 | 69.4% |
| grouped_cv | ridge_raw | mobile_nm | -0.1140 | 2.5744 | 80.6% |
| leave_one_solvent_out | mean | bonded_nm | -0.0087 | 0.1665 | 77.8% |
| leave_one_solvent_out | mean | mobile_nm | -0.1981 | 2.8916 | 86.1% |
| leave_one_solvent_out | rf_log_interactions | bonded_nm | -0.8608 | 0.2274 | 80.6% |
| leave_one_solvent_out | rf_log_interactions | mobile_nm | -1.6186 | 3.6332 | 75.0% |
| leave_one_solvent_out | rf_raw | bonded_nm | -0.6054 | 0.2103 | 83.3% |
| leave_one_solvent_out | rf_raw | mobile_nm | -0.4555 | 2.9277 | 66.7% |
| leave_one_solvent_out | ridge_log_interactions | bonded_nm | -1.4049 | 0.2606 | 61.1% |
| leave_one_solvent_out | ridge_log_interactions | mobile_nm | -2.1397 | 3.9278 | 66.7% |
| leave_one_solvent_out | ridge_raw | bonded_nm | -6.4589 | 0.3949 | 63.9% |
| leave_one_solvent_out | ridge_raw | mobile_nm | -11.7563 | 6.7005 | 77.8% |
| ood_high_concentration | mean | bonded_nm | -1.3038 | 0.2761 | 66.7% |
| ood_high_concentration | mean | mobile_nm | -0.4702 | 5.3864 | 44.4% |
| ood_high_concentration | rf_log_interactions | bonded_nm | 0.1476 | 0.1736 | 66.7% |
| ood_high_concentration | rf_log_interactions | mobile_nm | 0.0006 | 4.3424 | 11.1% |
| ood_high_concentration | rf_raw | bonded_nm | 0.2951 | 0.1545 | 77.8% |
| ood_high_concentration | rf_raw | mobile_nm | 0.1575 | 4.1572 | 11.1% |
| ood_high_concentration | ridge_log_interactions | bonded_nm | -3.5930 | 0.3364 | 33.3% |
| ood_high_concentration | ridge_log_interactions | mobile_nm | -0.5510 | 4.4782 | 0.0% |
| ood_high_concentration | ridge_raw | bonded_nm | -83.5145 | 1.3909 | 11.1% |
| ood_high_concentration | ridge_raw | mobile_nm | -25.7207 | 15.2102 | 0.0% |
| ood_low_concentration | mean | bonded_nm | -0.3743 | 0.1413 | 77.8% |
| ood_low_concentration | mean | mobile_nm | -58.6960 | 1.3555 | 100.0% |
| ood_low_concentration | rf_log_interactions | bonded_nm | -0.1324 | 0.1128 | 55.6% |
| ood_low_concentration | rf_log_interactions | mobile_nm | 0.1391 | 0.1456 | 100.0% |
| ood_low_concentration | rf_raw | bonded_nm | -0.4741 | 0.1265 | 66.7% |
| ood_low_concentration | rf_raw | mobile_nm | 0.0515 | 0.1518 | 100.0% |
| ood_low_concentration | ridge_log_interactions | bonded_nm | -0.2187 | 0.1240 | 44.4% |
| ood_low_concentration | ridge_log_interactions | mobile_nm | -26.1586 | 0.8355 | 100.0% |
| ood_low_concentration | ridge_raw | bonded_nm | -0.8176 | 0.1559 | 66.7% |
| ood_low_concentration | ridge_raw | mobile_nm | -118.4027 | 1.7499 | 100.0% |
