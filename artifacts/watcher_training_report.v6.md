# Watcher training report

Generated at: 2026-09-02T06:07:34.132893+00:00
Schema hash: bb4debdb162dbbf1f8a692b523b183770c4e6943916d2eebff11c3e5995b978f
Rows loaded / kept / rejected: 13317 / 5282 / 8035

| output_id | status | reason | C | cv_mean_bacc | test_bacc | test_ci_lower | lofo_pooled | confirm_bacc | accepted |
|---|---|---|---|---|---|---|---|---|---|
| labor_expansion | rejected | cv_gate_failed | 1.0 | 0.6075378787878789 | 0.675 | 0.5472836921850079 | 0.498868778280543 | 0.43939393939393934 | False |
| land_expansion | rejected | leave_one_family_out_gate_failed | 0.01 | 0.7141147741147741 | 0.7613636363636364 | 0.6458333333333333 | 0.3783240568954855 | 0.7833333333333333 | False |
| product:CARROT:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:CARROT:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:EGG:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:EGG:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:MELON:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:MELON:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:MILK:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:MILK:increase | rejected | test_gate_failed | 3.0 | 0.6277652086475616 | 0.6928571428571428 | 0.5587990196078432 | 0.5005450581395349 | 0.7 | False |
| product:STRAWBERRY:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:STRAWBERRY:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:TOMATO:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:WHEAT:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:WHEAT:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:WOOL:avoid | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
| product:WOOL:increase | unavailable | overall_quota_not_met |  |  |  |  |  |  |  |
