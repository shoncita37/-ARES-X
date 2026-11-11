import csv
import json
import unittest

from ares_x_engine import (
    CLASSES,
    FUZZY_RULES,
    MissionEngine,
    fuzzy_control,
)


class FuzzyControllerTests(unittest.TestCase):
    def test_mamdani_rules_produce_explainable_crisp_output(self):
        calm = fuzzy_control(
            battery=90, hazard=0.05, slope=2, gas=0, neural_confidence=0.95
        )
        hazard = fuzzy_control(
            battery=40, hazard=0.95, slope=35, gas=0.8, neural_confidence=0.5
        )
        self.assertGreaterEqual(len(FUZZY_RULES), 8)
        self.assertGreater(calm["speed"], 0)
        self.assertLess(hazard["speed"], calm["speed"])
        self.assertEqual(hazard["defuzzification"], "centroide Mamdani (min–max)")
        self.assertTrue(hazard["rules"])


class MissionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = MissionEngine(seed=37)

    def test_rna_uses_disjoint_map_splits_and_reports_metrics(self):
        metrics = self.engine.perception_metrics
        self.assertTrue(metrics["training_maps_do_not_overlap_validation_or_test"])
        train_maps = set(metrics["split_scenarios"]["train"])
        validation_maps = set(metrics["split_scenarios"]["validation"])
        test_maps = set(metrics["split_scenarios"]["test"])
        self.assertTrue(train_maps.isdisjoint(validation_maps | test_maps))
        self.assertEqual(metrics["architecture"], "MLP 8-32-16-3 · ReLU · softmax")
        self.assertEqual(metrics["confusion_matrix"]["labels"], list(CLASSES))
        self.assertIn("macro_f1", metrics["metrics"]["test"])
        self.assertIn("hazard_recall", metrics["metrics"]["test"])
        self.assertIn("converged", metrics)
        self.assertLessEqual(metrics["training_iterations"], 220)
        self.assertGreaterEqual(len(metrics["loss_curve"]), 2)

    def test_genetic_planner_evolves_and_reports_candidates(self):
        self.assertEqual(len(self.engine.genetic_history), 30)
        self.assertEqual(self.engine.genetic_history[0]["population"], 32)
        self.assertTrue(self.engine.route_candidates)
        self.assertTrue(all(
            candidate.distance_m >= 0 and candidate.feasible
            for candidate in self.engine.route_candidates
        ))
        self.assertIn("torneo", self.engine.genetic_history[0]["operators"])

    def test_experiment_genetic_budget_is_shared_and_reported(self):
        compact = MissionEngine(seed=96, train_model=False)
        compact.model = self.engine.model
        compact.perception_metrics = self.engine.perception_metrics
        compact.genetic_population_size = 8
        compact.genetic_generations = 5
        compact._scan_nearby()
        compact.replan()
        self.assertEqual(len(compact.genetic_history), 5)
        self.assertEqual(compact.genetic_history[0]["population"], 8)

    def test_step_connects_algorithms_and_csv_export(self):
        state = self.engine.step()
        decision = state["last_decision"]
        self.assertEqual(state["steps"], 1)
        self.assertIn("probabilities", decision["rna"])
        self.assertIn("memberships", decision["fuzzy"])
        self.assertTrue(decision["genetic"]["route_candidates"])
        self.assertTrue(decision["supervisor"]["hard_constraints_checked"])
        rows = list(csv.DictReader(self.engine.export_csv().splitlines()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], "1")

    def test_dynamic_obstacle_replans_and_cannot_overlap_rover(self):
        state = self.engine.state()
        rover = state["rover"]
        with self.assertRaises(ValueError):
            self.engine.add_object("rock", rover["row"], rover["col"])
        before = len(self.engine.events)
        self.engine.add_object("crater", 4, 4)
        self.assertGreater(len(self.engine.events), before)
        self.assertEqual(
            self.engine.events[-1]["type"], "replan"
        )

    def test_object_id_rejects_markup_and_oversized_values(self):
        for object_id in ("<img src=x onerror=alert(1)>", "x" * 65):
            with self.subTest(object_id=object_id):
                with self.assertRaises(ValueError):
                    self.engine.add_object("rock", 4, 4, object_id=object_id)

    def test_crater_is_blocked_by_altimeter_with_or_without_rna(self):
        engine = MissionEngine(seed=95, train_model=False)
        engine.model = self.engine.model
        engine.perception_metrics = self.engine.perception_metrics
        engine.move_start(8, 3)
        crater = engine._belief_at((8, 4))
        self.assertIsNotNone(crater)
        self.assertTrue(crater["blocked"])
        self.assertEqual(crater["terrain"], "barranco")

        engine.ablation["rna"] = False
        engine._scan_nearby()
        crater_without_rna = engine._belief_at((8, 4))
        self.assertTrue(crater_without_rna["blocked"])
        self.assertEqual(crater_without_rna["probabilities"]["obstaculo"], 0.2)

    def test_export_json_contains_config_and_trace(self):
        document = json.loads(self.engine.export_json())
        self.assertIn("configuration", document)
        self.assertIn("perception_metrics", document)
        self.assertIn("trace", document)

    def test_fuzzy_ablation_changes_movement_energy(self):
        first = MissionEngine(seed=94, train_model=False)
        first.model = self.engine.model
        first.perception_metrics = self.engine.perception_metrics
        first._scan_nearby()
        first.replan()
        with_fuzzy = first.step()

        second = MissionEngine(seed=94, train_model=False)
        second.model = self.engine.model
        second.perception_metrics = self.engine.perception_metrics
        second._scan_nearby()
        second.replan()
        without_fuzzy = second.step(ablation={"fuzzy": False})
        self.assertNotEqual(
            with_fuzzy["last_decision"]["supervisor"]["action"]["energy_cost"],
            without_fuzzy["last_decision"]["supervisor"]["action"]["energy_cost"],
        )


if __name__ == "__main__":
    unittest.main()
