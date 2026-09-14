"""Generic source interfaces and lightweight adapters."""

from .analytic import GaussianSource
from .base import CallableSource, PixelatedSource, SourceGeometry, StaticSource
from .physical import (
    GaussianModel,
    KerrDiskModel,
    PhysicalSourceModel,
    SourceGridConfig,
    ThinDiskModel,
)
from .reprocessing import (
    ThermalReprocessingSource,
    lamppost_irradiation_efficiency,
)
from .supernova import (
    ExpandingPhotosphereSource,
    PhotosphereAppearance,
    PhotosphereEvolution,
    PowerLawExponentialPhotosphere,
    paper_type_ia_supernova_source,
)
from .thin_disk import (
    RadiativeEfficiency,
    ThinDiskSource,
    ViscousFluxProfile,
    thin_disk_flux_radius_rg,
    thin_disk_temperature4,
)
from .transferred_disk import TransferredThinDiskSource
from .variability import (
    BrokenPowerLawPSD,
    CallableDrivingSignal,
    DelayedModulatedSource,
    DrivingSignal,
    ModulatedSource,
    TabulatedDrivingSignal,
    TimeShiftedSource,
    broken_power_law_driving_signal,
    driving_signal_from_psd,
    lognormal_damped_random_walk,
)

__all__ = [
    "BrokenPowerLawPSD",
    "CallableDrivingSignal",
    "CallableSource",
    "DelayedModulatedSource",
    "DrivingSignal",
    "ExpandingPhotosphereSource",
    "GaussianModel",
    "GaussianSource",
    "KerrDiskModel",
    "ModulatedSource",
    "PixelatedSource",
    "PhotosphereAppearance",
    "PhotosphereEvolution",
    "PhysicalSourceModel",
    "PowerLawExponentialPhotosphere",
    "RadiativeEfficiency",
    "SourceGeometry",
    "SourceGridConfig",
    "StaticSource",
    "TabulatedDrivingSignal",
    "ThermalReprocessingSource",
    "TimeShiftedSource",
    "broken_power_law_driving_signal",
    "driving_signal_from_psd",
    "lognormal_damped_random_walk",
    "lamppost_irradiation_efficiency",
    "paper_type_ia_supernova_source",
    "ThinDiskSource",
    "ThinDiskModel",
    "TransferredThinDiskSource",
    "ViscousFluxProfile",
    "thin_disk_flux_radius_rg",
    "thin_disk_temperature4",
]
