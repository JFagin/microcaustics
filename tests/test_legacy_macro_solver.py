"""Smoke checks for the frozen internal solver, not the package API."""

import importlib.util
import unittest

import torch
from _legacy_macro_solver import CallableMacroModel, solve_macroimages


class LegacySolverReferenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires scipy")
    def test_frozen_callable_solver_recovers_identity(self) -> None:
        def raytrace(x, y):
            return x, y

        def jacobian(x, y):
            identity = torch.eye(2, dtype=x.dtype, device=x.device)
            return identity.expand(*x.shape, 2, 2)

        def time_delay(x, y):
            return 0.5 * (x.square() + y.square())

        model = CallableMacroModel(raytrace, jacobian, time_delay)
        solutions = solve_macroimages(
            model,
            0.0,
            0.0,
            initial_grid_size=41,
            field_of_view_arcsec=1.0,
        )
        self.assertEqual(len(solutions), 1)
        self.assertAlmostEqual(solutions[0].macro_magnification, 1.0)
        self.assertAlmostEqual(solutions[0].convergence, 0.0)
        self.assertAlmostEqual(solutions[0].shear, 0.0)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires scipy")
    def test_frozen_solver_rejects_iteration_limited_candidates(self) -> None:
        def raytrace(x, y):
            return x, y

        def jacobian(x, y):
            identity = torch.eye(2, dtype=x.dtype, device=x.device)
            return identity.expand(*x.shape, 2, 2)

        def time_delay(x, y):
            return 0.5 * (x.square() + y.square())

        model = CallableMacroModel(raytrace, jacobian, time_delay)
        with self.assertRaisesRegex(RuntimeError, "did not produce a valid image"):
            solve_macroimages(
                model,
                0.123,
                -0.087,
                initial_grid_size=41,
                field_of_view_arcsec=1.0,
                maximum_function_evaluations=1,
            )
