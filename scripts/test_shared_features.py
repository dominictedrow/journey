import copy
import unittest

import counterfactual_dataset as cd
import shared_features as sf


def observation():
    farm = {"money": 1000, "hands": [], "unlocked_quadrants": [0],
            "tiles": [[None, {"kind": "PLANT", "crop": "WHEAT",
                              "planted_day": 1, "yield_units": 2,
                              "consecutive_unwatered": 0}]]}
    opponent = {"money": 900, "hands": [{}], "unlocked_quadrants": [0, 1],
                "tiles": [[{"kind": "WEED"}, None]]}
    return {"player": 0, "step": 72, "day": 3, "hour": 0,
            "farms": [farm, opponent],
            "private": {"shed": {"WHEAT": 7}, "seeds": {"MELON": 2},
                        "inventories": [{"CARROT": 3}]},
            "market": {"prices": {}, "inventory": {}},
            "town": {"unlocked_shops": ["BAKERY"]}}


class SharedFeatureTests(unittest.TestCase):
    def test_counterfactual_module_reexports_shared_contract(self):
        self.assertIs(cd.FEATURE_ORDER, sf.FEATURE_ORDER)
        self.assertEqual(cd.SCHEMA_HASH, sf.SCHEMA_HASH)
        self.assertIs(cd.extract_features, sf.extract_features)

    def test_schema_contract_is_stable(self):
        self.assertEqual(len(sf.FEATURE_ORDER), 129)
        self.assertEqual(sf.SCHEMA_HASH,
                         "ac1487d70ff7df007d3ba7414e0fa4de4db4a4f31be9296534c570a86f2210c9")

    def test_opponent_private_sentinels_do_not_affect_features(self):
        base = observation()
        altered = copy.deepcopy(base)
        base["opponent_private"] = {"shed": {"MELON": -999999}}
        altered["opponent_private"] = {"shed": {"MELON": 999999}}
        base["farms"][1]["private"] = {"seeds": {"WHEAT": -999999}}
        altered["farms"][1]["private"] = {"seeds": {"WHEAT": 999999}}
        self.assertEqual(sf.extract_features(base), sf.extract_features(altered))

    def test_focal_private_inventory_is_included(self):
        before = observation()
        after = copy.deepcopy(before)
        after["private"]["shed"]["WHEAT"] = 8
        index = sf.FEATURE_ORDER.index("own_shed_WHEAT")
        self.assertEqual(sf.extract_features(after)[index] - sf.extract_features(before)[index], 1)


if __name__ == "__main__":
    unittest.main()
