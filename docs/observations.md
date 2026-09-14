# Rubin cadences and observations

`microcaustics` can select Rubin visits from a local OpSim SQLite database and
apply them to either one light curve or all resolved macroimages of a lensed
system. The database is indexed once per Python process; subsequent sky
queries and random samples use compact in-memory arrays rather than rescanning
SQLite.

## Build the reusable cadence index

```python
import microcaustics as mc

rubin = mc.RubinOpSimCadenceIndex.from_database("baseline_v4.3.5_10yrs.db")
print(rubin.ddf_fields)
```

Repeated calls to `from_database` for the same unchanged file return the
process-cached index. Keep the `rubin` object and reuse it when producing a
large simulation set.

## Select a cadence

Random WFD sampling is the default:

```python
cadence = rubin.sample(seed=123)
```

Select a random or named deep-drilling field with the same method:

```python
random_ddf = rubin.sample(seed=123, survey="ddf")
cosmos = rubin.sample(seed=123, survey="ddf", field="COSMOS")
cosmos_with_wfd = rubin.sample(
    seed=123,
    survey="ddf",
    field="COSMOS",
    include_wfd=True,
)
```

For a known source position, select all pointings whose centers lie within the
configured field-of-view radius:

```python
cadence = rubin.at_sky_position(
    ra_deg=340.126,
    dec_deg=3.358,
    survey="all",  # "wfd", "ddf", or "all"
)
```

Every cadence uses the first MJD in the full OpSim database as its time origin.
This preserves the relative seasonal phase of different sky positions. The
selection is a circular pointing-center approximation controlled by
`radius_deg`; it does not evaluate the detailed camera footprint.

## Observe one light curve

```python
observed = mc.observe_light_curve(
    truth_curve,
    cadence,
    seed=123,
)
```

`truth_curve` must cover the cadence time range and contain every selected
band. Its flux is interpreted as Jy by the default AB zero point of 3631 Jy.

## Observe a lensed system

```python
observed_images = mc.observe_multi_image_light_curves(
    truth_curves,
    cadence,
    seed=123,
)
```

The same visit times, bands, depths, and systematic-noise prescription are
applied to every macroimage. This is the physical default for resolved images
of one lensed system. Random measurement draws remain independent for each
visit and macroimage.

Advanced users can configure the photometric model in both observation
functions:

```python
observed = mc.observe_light_curve(
    truth_curve,
    cadence,
    seed=123,
    gamma_by_band={"g": 0.040, "r": 0.039},
    systematic_floor_mag=0.003,
    add_noise=True,
)
```

Set `add_noise=False` to retain noiseless magnitudes while still calculating
the expected uncertainty.
