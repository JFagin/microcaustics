"""Shared spectral checks retain the public validation behavior."""

import itertools
import unittest

from microcaustics._system.scheduling import _validate_spectral_options
from microcaustics.batching import batched_system_light_curves
from microcaustics.multi_system import MultiImageSystem
from microcaustics.system import MicrolensingRealization, MicrolensingSystem


class SpectralOptionsTests(unittest.TestCase):
    def test_all_option_combinations_and_error_priority(self):
        for wavelength_batch, band_batch, model, filters, spectrum in itertools.product(
            (False, True), repeat=5
        ):
            options = dict(
                wavelength_batch_size=1 if wavelength_batch else None,
                band_batch_size=1 if band_batch else None,
                spectral_model=object() if model else None,
                bandpasses=object() if filters else None,
                return_spectrum=spectrum,
            )
            expected = None
            if wavelength_batch and band_batch:
                expected = "supply wavelength_batch_size or band_batch_size, not both"
            elif model and not filters:
                expected = "spectral_model requires bandpasses"
            elif spectrum and not filters:
                expected = "return_spectrum requires bandpasses"
            with self.subTest(options=options):
                if expected is None:
                    self.assertIsNone(_validate_spectral_options(**options))
                else:
                    with self.assertRaises(ValueError) as caught:
                        _validate_spectral_options(**options)
                    self.assertEqual(str(caught.exception), expected)

    def test_public_entrypoints_reject_invalid_options_before_setup(self):
        # Invalid spectral options must fail before accessing a system or source.
        calls = (
            (MicrolensingRealization.light_curve, (None, [0.0])),
            (MicrolensingRealization.source_light_curve, (None, [0.0])),
            (MicrolensingRealization.multirate_light_curve, (None, [0.0], [0.0])),
            (MicrolensingSystem.light_curve, (None, [0.0])),
            (MultiImageSystem.light_curves, (None, [0.0])),
            (batched_system_light_curves, ([], [0.0])),
        )
        invalid_options = (
            (
                dict(wavelength_batch_size=1, band_batch_size=1),
                "supply wavelength_batch_size or band_batch_size, not both",
            ),
            (dict(spectral_model=object()), "spectral_model requires bandpasses"),
            (dict(return_spectrum=True), "return_spectrum requires bandpasses"),
            (
                dict(spectral_model=object(), return_spectrum=True),
                "spectral_model requires bandpasses",
            ),
        )
        for function, args in calls:
            for options, expected in invalid_options:
                with self.subTest(function=function.__qualname__, options=options):
                    with self.assertRaises(ValueError) as caught:
                        function(*args, **options)
                    self.assertEqual(str(caught.exception), expected)

    def test_zero_batch_sizes_still_count_as_supplied(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            _validate_spectral_options(
                wavelength_batch_size=0,
                band_batch_size=0,
                spectral_model=None,
                bandpasses=None,
                return_spectrum=False,
            )
