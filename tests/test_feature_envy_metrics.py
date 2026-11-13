import unittest

from src.feature_envy_metrics import evaluate_analysis


def analysis(fields, properties, unresolved=None):
    return {
        "Target": {"MethodName": "Synthetic"},
        "Fields": fields,
        "Properties": properties,
        "Callees": [{"Symbol": "Ignored.Call()"}],
        "UnresolvedDependencies": unresolved or [],
    }


def reference(scope, declaring_type, member, relationship="FIELD_ACCESS"):
    return {
        "Scope": scope,
        "DeclaringType": declaring_type,
        "Member": member,
        "Relationship": relationship,
        "Symbol": declaring_type + "." + member,
    }


class FeatureEnvyMetricsTests(unittest.TestCase):
    def test_exact_thresholds_do_not_classify(self):
        result = evaluate_analysis(
            analysis(
                [reference("FOREIGN", "Foreign", f"f{i}") for i in range(5)],
                [reference("LOCAL", "Host", f"local{i}") for i in range(5)],
            )
        )
        self.assertEqual(result["Metrics"], {"ATFD": 5, "LAA": 0.5, "FDP": 1})
        self.assertFalse(result["FeatureEnvy"])

    def test_threshold_boundary_classifies_with_and_rule(self):
        result = evaluate_analysis(
            analysis(
                [reference("FOREIGN", "Foreign", f"f{i}") for i in range(6)],
                [reference("LOCAL", "Host", "local")],
            )
        )
        self.assertEqual(result["Metrics"], {"ATFD": 6, "LAA": 1 / 7, "FDP": 1})
        self.assertTrue(result["FeatureEnvy"])

    def test_laa_equal_boundary_does_not_classify(self):
        result = evaluate_analysis(
            analysis(
                [reference("FOREIGN", "Foreign", f"f{i}") for i in range(6)],
                [reference("LOCAL", "Host", f"l{i}") for i in range(3)],
            )
        )
        self.assertEqual(result["Metrics"]["LAA"], 1 / 3)
        self.assertFalse(result["FeatureEnvy"])

    def test_fdp_boundary_is_inclusive(self):
        result = evaluate_analysis(
            analysis(
                [
                    reference("FOREIGN", f"Foreign{i}", "value")
                    for i in range(5)
                ],
                [],
            )
        )
        self.assertEqual(result["Metrics"]["FDP"], 5)
        self.assertFalse(result["FeatureEnvy"])

    def test_calls_are_not_data_and_unresolved_is_uncertain(self):
        result = evaluate_analysis(
            analysis(
                [reference("FOREIGN", "Foreign", "value")],
                [],
                unresolved=[{"Relationship": "MEMBER_ACCESS", "Expression": "unknown"}],
            )
        )
        self.assertEqual(result["Metrics"], {"ATFD": 1, "LAA": 0.0, "FDP": 1})
        self.assertIsNone(result["FeatureEnvy"])
        self.assertEqual(result["Status"], "UNCERTAIN")

    def test_repeated_foreign_attribute_counts_once_for_atfd(self):
        result = evaluate_analysis(
            analysis(
                [
                    reference("FOREIGN", "Foreign", "value"),
                    reference("FOREIGN", "Foreign", "value"),
                    reference("FOREIGN", "Foreign", "other"),
                ],
                [reference("LOCAL", "Host", "local")],
            )
        )
        self.assertEqual(result["Metrics"], {"ATFD": 2, "LAA": 1 / 3, "FDP": 1})
        self.assertEqual(
            result["Evidence"]["ForeignDataAccessAttributes"],
            ["Foreign.other", "Foreign.value"],
        )


if __name__ == "__main__":
    unittest.main()
