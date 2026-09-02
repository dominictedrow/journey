# Phase H / 5.2 -- SVM coefficient audit

Corpus: `data/counterfactual_dataset.v6.jsonl`
Schema hash: `bb4debdb162dbbf1f8a692b523b183770c4e6943916d2eebff11c3e5995b978f`

"Adaptive L1 share" = fraction of the standardized |coef| mass on the shared-market (`price_ratio_*`, `market_displacement_*`) plus opponent-state (`opponent_*`) features. Near zero => the model decides from its own board, not the opponent or the market.

| output | fitted | CV bacc | market share | opponent share | adaptive share |
|---|---|---|---|---|---|
| labor_expansion | yes | 0.608 | 28.2% | 19.8% | 48.0% |
| land_expansion | yes | 0.714 | 17.0% | 18.2% | 35.2% |
| product:CARROT:avoid | no (quota_not_met) | | | | |
| product:CARROT:increase | no (quota_not_met) | | | | |
| product:EGG:avoid | no (quota_not_met) | | | | |
| product:EGG:increase | no (quota_not_met) | | | | |
| product:MELON:avoid | no (quota_not_met) | | | | |
| product:MELON:increase | no (quota_not_met) | | | | |
| product:MILK:avoid | no (quota_not_met) | | | | |
| product:MILK:increase | yes | 0.628 | 23.7% | 29.8% | 53.6% |
| product:STRAWBERRY:avoid | no (quota_not_met) | | | | |
| product:STRAWBERRY:increase | no (quota_not_met) | | | | |
| product:TOMATO:increase | no (quota_not_met) | | | | |
| product:WHEAT:avoid | no (quota_not_met) | | | | |
| product:WHEAT:increase | no (quota_not_met) | | | | |
| product:WOOL:avoid | no (quota_not_met) | | | | |
| product:WOOL:increase | no (quota_not_met) | | | | |

## labor_expansion

C = 1.0, CV mean balanced accuracy = 0.608, fit rows (binary) = 129, signed = {'0': 714, '1': 109, '-1': 78}

Top coefficients by |value|:

| feature | coef | group |
|---|---|---|
| market_displacement_MELON | -1.514 | market |
| own_seed_STRAWBERRY | +1.384 | own |
| own_money | -1.361 | own |
| price_ratio_TOMATO | +1.199 | market |
| own_shed_FERTILIZER | -1.040 | own |
| own_weeds | +0.974 | own |
| opponent_animal_COW | -0.827 | opponent |
| opponent_ready_yield | +0.810 | opponent |
| own_shed_MILK | -0.792 | own |
| price_ratio_WOOL | +0.766 | market |
| market_displacement_MILK | +0.711 | market |
| opponent_crop_CARROT | -0.692 | opponent |
| price_ratio_EGG | +0.659 | market |
| own_shed_WHEAT | -0.645 | own |
| shop_BRUNCH_SPOT | -0.628 | phase_misc |

Shared-market coefficients:

| feature | coef |
|---|---|
| price_ratio_WHEAT | +0.442 |
| market_displacement_WHEAT | -0.106 |
| price_ratio_CARROT | -0.183 |
| market_displacement_CARROT | +0.531 |
| price_ratio_TOMATO | +1.199 |
| market_displacement_TOMATO | -0.163 |
| price_ratio_STRAWBERRY | -0.605 |
| market_displacement_STRAWBERRY | -0.482 |
| price_ratio_MELON | -0.416 |
| market_displacement_MELON | -1.514 |
| price_ratio_EGG | +0.659 |
| market_displacement_EGG | -0.537 |
| price_ratio_MILK | +0.540 |
| market_displacement_MILK | +0.711 |
| price_ratio_WOOL | +0.766 |
| market_displacement_WOOL | -0.253 |

## land_expansion

C = 0.01, CV mean balanced accuracy = 0.714, fit rows (binary) = 115, signed = {'-1': 79, '1': 94, '0': 297}

Top coefficients by |value|:

| feature | coef | group |
|---|---|---|
| own_shed_MELON | +0.139 | own |
| own_empty | -0.132 | own |
| own_quadrants | -0.116 | own |
| own_shed_FERTILIZER | +0.116 | own |
| price_ratio_MILK | +0.106 | market |
| shop_SMOOTHIE_SHOP | -0.101 | phase_misc |
| shop_FARMERS_MARKET | +0.095 | phase_misc |
| price_ratio_EGG | -0.082 | market |
| own_animal_COW | +0.081 | own |
| opponent_ready_yield | -0.071 | opponent |
| own_seed_MELON | -0.067 | own |
| own_shed_WHEAT | +0.066 | own |
| opponent_weeds | +0.063 | opponent |
| own_crop_MELON | -0.060 | own |
| shop_BRUNCH_SPOT | +0.058 | phase_misc |

Shared-market coefficients:

| feature | coef |
|---|---|
| price_ratio_WHEAT | +0.009 |
| market_displacement_WHEAT | +0.034 |
| price_ratio_CARROT | +0.011 |
| market_displacement_CARROT | -0.013 |
| price_ratio_TOMATO | -0.011 |
| market_displacement_TOMATO | +0.020 |
| price_ratio_STRAWBERRY | +0.049 |
| market_displacement_STRAWBERRY | +0.009 |
| price_ratio_MELON | -0.048 |
| market_displacement_MELON | -0.001 |
| price_ratio_EGG | -0.082 |
| market_displacement_EGG | +0.043 |
| price_ratio_MILK | +0.106 |
| market_displacement_MILK | -0.028 |
| price_ratio_WOOL | -0.004 |
| market_displacement_WOOL | -0.043 |

## product:MILK:increase

C = 3.0, CV mean balanced accuracy = 0.628, fit rows (binary) = 150, signed = {'0': 877, '-1': 83, '1': 111}

Top coefficients by |value|:

| feature | coef | group |
|---|---|---|
| opponent_crop_MELON_age_0_2 | -1.657 | opponent |
| market_displacement_WHEAT | -1.406 | market |
| price_ratio_WHEAT | -1.243 | market |
| own_money | +1.242 | own |
| own_seed_WHEAT | +1.227 | own |
| opponent_quadrants | -1.224 | opponent |
| opponent_structure_PASTURE | -1.130 | opponent |
| opponent_animal_COW | -1.080 | opponent |
| price_ratio_STRAWBERRY | +1.002 | market |
| opponent_crop_MELON | -0.982 | opponent |
| own_crop_WHEAT_age_3_7 | -0.982 | own |
| market_displacement_CARROT | +0.975 | market |
| own_crop_MELON_age_8_plus | +0.908 | own |
| opponent_crop_CARROT | -0.906 | opponent |
| own_ready_yield | -0.829 | own |

Shared-market coefficients:

| feature | coef |
|---|---|
| price_ratio_WHEAT | -1.243 |
| market_displacement_WHEAT | -1.406 |
| price_ratio_CARROT | +0.488 |
| market_displacement_CARROT | +0.975 |
| price_ratio_TOMATO | -0.295 |
| market_displacement_TOMATO | -0.381 |
| price_ratio_STRAWBERRY | +1.002 |
| market_displacement_STRAWBERRY | +0.614 |
| price_ratio_MELON | -0.226 |
| market_displacement_MELON | -0.731 |
| price_ratio_EGG | -0.742 |
| market_displacement_EGG | -0.115 |
| price_ratio_MILK | +0.366 |
| market_displacement_MILK | +0.205 |
| price_ratio_WOOL | -0.249 |
| market_displacement_WOOL | -0.077 |

