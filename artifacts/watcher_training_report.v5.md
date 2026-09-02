# Watcher training report

Generated at: 2026-08-31T23:28:10.842855+00:00
Schema hash: 2ce35fef153f54063cc1a91459eca36e2485c4fe6e30957b1876dc6409d973f4
Rows loaded / kept / rejected: 6400 / 3570 / 2830

| output_id | status | reason | C | cv_mean_bacc | test_bacc | test_ci_lower | lofo_pooled | confirm_bacc | accepted |
|---|---|---|---|---|---|---|---|---|---|
| labor_expansion | rejected | cv_gate_failed | 1.0 | 0.6001633986928104 | 0.4734848484848485 | 0.3038461538461539 | 0.4962121212121212 | 0.55 | False |
| land_expansion | rejected | leave_one_family_out_gate_failed | 0.1 | 0.7319230769230769 | 0.8055555555555556 | 0.670413961038961 | 0.42207792207792205 | 0.7063492063492063 | False |
| product:EGG:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:MELON:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:MILK:increase | rejected | test_gate_failed | 0.01 | 0.6637878787878787 | 0.5357142857142857 | 0.37098484848484853 | 0.48833333333333334 | 0.6666666666666667 | False |
| product:STRAWBERRY:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:WOOL:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:WOOL:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
