---
title: "BraggCalculator: Fast and differentiable Bragg powder diffraction in Python"
tags:
  - Python
  - crystallography
  - powder diffraction
  - X-ray diffraction
  - neutron diffraction
  - automatic differentiation
authors:
  - name: Frederik Lizak Johansen
    orcid: 0000-0002-8049-8624
    affiliation: "1"
    corresponding: true
affiliations:
  - index: 1
    name: Department of Chemistry, University of Copenhagen, Denmark
    ror: 035b05819
date: 16 July 2026
bibliography: paper.bib
---

# Summary

Powder diffraction connects an atomic structure to an experimentally familiar
fingerprint. Simulating that fingerprint is consequently a routine step in
phase identification, structure screening, refinement, and machine-learning
workflows. `BraggCalculator` [@braggcalculator2026] is a Python package for calculating ideal,
monochromatic powder X-ray and neutron diffraction from periodic crystals. It
accepts CIF files and structures from pymatgen or the Atomic Simulation
Environment (ASE) [@larsen2017], and returns indexed reflections, powder lines,
or broadened profiles in scattering angle or momentum transfer. A NumPy backend
provides a lightweight CPU path [@harris2020], while an optional PyTorch backend
supports automatic differentiation and accelerator execution [@paszke2019].

The package deliberately solves a bounded problem: kinematic diffraction from
an ideal periodic structure. It includes occupancies, isotropic displacement
parameters, neutral-atom X-ray form factors, coherent neutron scattering
lengths, powder Lorentz--polarization corrections, and area-normalized Gaussian
profiles. It does not attempt Rietveld refinement or model background,
preferred orientation, microstrain, absorption, diffuse scattering, or finite
particle shape. Keeping this boundary explicit makes the numerical core small,
testable, and suitable for repeated forward calculations.

# Statement of need

A single powder pattern is inexpensive to calculate, but research applications
increasingly require many evaluations: screening generated structures,
optimizing a structure against an observed pattern, constructing differentiable
losses, or augmenting training data. In these settings, Python-level work per
reflection and repeated symmetry analysis become significant. A useful forward
model must also handle general cells without hand-coded reflection rules,
preserve systematic absences, and expose its physical parameters to automatic
differentiation.

`BraggCalculator` addresses this need for researchers working with periodic
crystals who require a scriptable and verifiable forward model. It separates
the discrete crystallographic topology from the continuous calculation. Space
group analysis and reflection enumeration are performed when a structure is
loaded; repeated calculations then reuse that topology while evaluating
lattice, coordinate, occupancy, and displacement parameters with vectorized
array operations. This is particularly useful when many related patterns are
needed from one reflection set. The package also provides an end-to-end path
when structures are unrelated and caching is not applicable.

# State of the field

pymatgen provides widely used powder X-ray and neutron line patterns
[@ong2013], while GSAS-II covers data reduction, structure solution, and
refinement [@toby2013]. Both remain important general materials tools. The
pymatgen 2026.5.4 benchmark release and BraggCalculator evaluate the same
kinematic equations. Pymatgen rebuilds reciprocal points and flattened site
arrays for every pattern and loops over reflections in Python, although its
site sum within each reflection is NumPy-vectorized. BraggCalculator reduces
the primitive cell and enumerates the complete reflection topology at loading,
then evaluates reflection-by-site chunks and merges coincident lines with
indexed reductions. Cached calls reuse the discrete setup. Symmetric
supercells gain further because numerical work follows the primitive site
count. These changes reorganize exact work: every reflection and the full
structure factor remain present. Oracle tests verify pymatgen's convention.

The Debye scattering equation provides another route from atom coordinates to
scattering and is especially valuable for finite, disordered, or
non-crystalline systems. `DebyeCalculator`, for example, provides GPU-accelerated
powder and total-scattering calculations for such structures [@johansen2024].
Its pair sum scales quadratically with atom count. For an ideal periodic
crystal, reciprocal-space Bragg diffraction instead evaluates the finite set of
reflections in the requested range. The two approaches therefore cover
different physical models: `BraggCalculator` uses translational symmetry for
periodic crystals, while a Debye calculation retains finite-size and
non-periodic information. This narrower architecture is the reason for a
separate package: its physical scope and computational structure differ from
those of general refinement suites and real-space scattering codes.

# Software design

Bragg diffraction is the constructive interference of waves scattered by a
periodic crystal. For parallel lattice planes separated by $d$, rays scattered
from adjacent planes travel an extra distance $2d\sin\theta$, where $\theta$ is
half the angle between the incident and scattered beams. A peak occurs when
this path difference is one wavelength,

$$2d_{\mathbf{h}}\sin\theta_{\mathbf{h}} = \lambda.$$

The Miller index triplet $\mathbf{h}=(h,k,l)$ labels a family of planes. In a
reciprocal-lattice enumeration, higher diffraction orders are represented by
integer multiples of $\mathbf{h}$, so this first-order form covers every peak.
With reciprocal vectors defined without the $2\pi$ convention,
$\mathbf{g}_{\mathbf{h}}$ has length $1/d_{\mathbf{h}}$. The scattering-vector
magnitude is consequently
$Q_{\mathbf{h}}=2\pi|\mathbf{g}_{\mathbf{h}}|=4\pi\sin\theta_{\mathbf{h}}/\lambda$.
Calculating all peaks in a requested angular or $Q$ range is therefore an
integer-point search inside an ellipsoid defined by the reciprocal metric.
`BraggCalculator` performs that complete search after pymatgen and spglib
[@togo2024] have identified a consistent primitive cell.

Peak strength follows from the same interference argument within one unit
cell. In the kinematic approximation, each atomic site scatters the incident
wave once. The far-field waves add as complex amplitudes, and a fractional
coordinate $\mathbf{r}_j$ contributes the phase
$2\pi\mathbf{h}\cdot\mathbf{r}_j$. The unit-cell amplitude is the corresponding
Fourier coefficient of the periodic scattering density,

$$
F_{\mathbf{h}} = \sum_j o_j f_j(s)
\exp(-B_j s^2)
\exp(2\pi i\,\mathbf{h}\cdot\mathbf{r}_j),
\qquad s = \frac{\sin\theta}{\lambda},
$$

where $o_j$ is occupancy and $B_j$ is the isotropic displacement parameter. The
factor $f_j(s)$ describes the interaction strength at
$s=\sin\theta/\lambda=1/(2d)$. For X-rays it is the angle-dependent atomic form
factor of the electron density, evaluated with the Doyle--Turner
parameterization [@doyle1968]. For neutrons it is replaced by the coherent
nuclear scattering length [@sears1992]. Thermal motion attenuates high-angle
coherence through $\exp(-B_js^2)$. The detector records intensity, giving
$|F_{\mathbf{h}}|^2$. Exact cancellation of the phase terms produces systematic
absences directly, so no separate table of space-group extinction rules is
needed.

A powder contains crystallites in many orientations, allowing every eligible
plane family to satisfy Bragg's law. The implementation evaluates every
reciprocal point explicitly and sums points with the same $d$ spacing when it
reports powder lines; this accounts for multiplicity once. The
Lorentz--polarization factor corrects the angular sampling geometry and, for
X-rays, beam polarization. For the convention used here,

$$
I_{\mathbf{h}}=|F_{\mathbf{h}}|^2L(2\theta),\qquad
L_{\mathrm{X}}=\frac{1+\cos^2(2\theta)}{\sin^2\theta\cos\theta},\qquad
L_{\mathrm{N}}=\frac{1}{\sin^2\theta\cos\theta}.
$$

Convolution with an area-normalized Gaussian turns the ideal lines into a
finite-width profile while preserving integrated intensity. Physical
coefficients and named wavelengths are read from the versioned pymatgen tables.

Reflection enumeration and symmetry detection are discrete and cannot be
differentiated. After this preparation, however, the structure-factor,
geometry, correction, and profile operations remain in the selected numerical
backend. PyTorch users can therefore differentiate profiles with respect to
the lattice, fractional coordinates, occupancies, and isotropic displacement
parameters. A calculation must be rebuilt if a change is large enough for a
reflection to enter or leave the prepared range. Phase matrices and profile
kernels are evaluated in bounded chunks, trading a small loop over chunks for
predictable memory use on both CPUs and accelerators.

With `TorchBackend(device="cuda")`, reciprocal geometry, form factors,
structure factors, indexed line sums, and profile convolution execute on the
GPU. CIF parsing, primitive-cell detection, and reflection enumeration remain
CPU preprocessing as discrete topology operations. CUDA is most useful for
large reflection-by-site workloads, broadened profiles, and repeated calls
that amortize transfers and kernel launches. The oracle-grade benchmark uses
double precision, so gains also depend on GPU double-precision throughput.
Batching patterns and tuning chunk sizes can increase utilization further.

# Validation and performance

Correctness is tested at three levels: analytical unit tests, extinction and
occupancy cases, and an oracle suite comparing X-ray and neutron line patterns
with pymatgen. The generated-pattern comparison in \autoref{fig:oracle} covers
NaCl, SrTiO$_3$, and a triclinic SiO$_2$ cell. Across 172 X-ray powder lines,
the largest position difference is $4.3\times10^{-14}$ degrees and the largest
relative-intensity difference is $3.6\times10^{-14}$ percentage points. The
same area-normalized Gaussian was then applied independently to both sets of
lines. At full widths at half maximum of 0.1 and 0.5 degrees, the largest
profile differences are $1.1\times10^{-11}$ and $2.1\times10^{-12}$ percentage
points, respectively. Agreement remains at the same numerical level after the
stronger broadening, confirming that peak overlap does not conceal a
line-pattern discrepancy.

![Oracle comparison before broadening (left) and after applying the same
area-normalized Gaussian to both implementations at 0.1 degrees (centre) and
0.5 degrees (right) full width at half maximum. Orange pymatgen and dashed blue
BraggCalculator profiles are overlaid; the magenta lower trace is their
pointwise difference on the displayed scientific scale.
\label{fig:oracle}](figures/pattern_comparison_xray.pdf){width="100%"}

A frozen corpus test, summarized in \autoref{fig:cif-validation}, broadens this
oracle comparison beyond constructed examples. Seventy CC0 CIF records from
the Crystallography Open Database
[@grazulis2009] were selected before comparison, with ten structures from each
crystal system, 62 declared space groups, 3--992 supplied sites, and 31
disordered structures. Every COD revision and file hash is versioned. Across
57,693 X-ray and 58,819 neutron powder lines, all 140 structure--radiation
comparisons pass. The largest line-position difference is
$7.1\times10^{-14}$ degrees and the largest normalized-intensity difference is
$7.5\times10^{-13}$ percentage points. Both implementations receive the same
pymatgen-parsed structure; parser warnings are retained in the result artifact,
so this establishes implementation agreement rather than experimental
validation of the deposited structures or the kinematic model.

![Frozen COD corpus coverage and pymatgen agreement. (a) Supplied and
primitive-cell sites for 70 structures; color denotes crystal system, squares
denote disorder, and the dashed line is equality. (b) BraggCalculator and
pymatgen powder-line counts for X-ray circles and neutron triangles; all 140
comparisons lie on equality. (c,d) Maximum line-position and normalized-
intensity errors expressed as decimal orders below their respective acceptance
tolerances. Every comparison has at least a 5.1-order margin.
\label{fig:cif-validation}](figures/cif_validation_summary.pdf){width="100%"}

The scaling benchmark validates every case before timing and records raw
samples, software versions, hardware, thread settings, and the exact Git
revision. \autoref{fig:scaling} reports seven interleaved repeats from a single
WSL2 host with an NVIDIA RTX A3000 Laptop GPU. The NumPy CPU, PyTorch CPU, and
float64 PyTorch CUDA records use the same revision, dependencies, cases, thread
limits, and timing protocol.
“Supplied sites” means sites in the structure passed to both calculators,
before primitive-cell reduction.
For P1 cells containing 4--256 irreducible sites, the NumPy cached speedup over
pymatgen is 18.7--49.3 times and its end-to-end speedup is 13.5--20.0 times. The
CUDA run becomes increasingly effective as the reflection-by-site workload
grows. It crosses the matched PyTorch CPU runtime at 64 sites and is 8.7 times
faster cached and 6.5 times faster end-to-end at 256 sites; its corresponding
speedups over host pymatgen are 190 and 132 times.

For NaCl supercells containing 8--512 supplied sites, BraggCalculator reduces
every case to the same two-site primitive cell and 410 reciprocal points. The
CUDA calculation is therefore launch-, transfer-, and synchronization-bound
rather than throughput-bound. Against the matched PyTorch CPU run, CUDA is
5.2--5.7 times slower cached and 1.4--1.5 times slower end-to-end across these
four cases. Its end-to-end speedup over host pymatgen nevertheless rises from
0.81 to 77.1 times because the increasing advantage comes from primitive-cell
reduction rather than additional GPU work. Each plotted speedup uses pymatgen
from the same timing record; the direct device row uses the matched PyTorch CPU
and CUDA records.

![Runtime, speedup over pymatgen, and direct CUDA acceleration for increasing
irreducible P1 cells (left) and symmetry-reducible NaCl supercells (right).
Color identifies the BraggCalculator execution path; circles with solid lines
are cached timings and squares with dashed lines are end-to-end timings.
Runtime lines show medians and speedup lines show ratios of medians; bands show
interquartile ranges over the raw timings or paired speedup ratios. The gray
pymatgen runtime pools the three same-host records, while each speedup uses the
pymatgen repeats in its own timing record. The bottom row is the ratio of the
PyTorch CPU and CUDA median runtimes, so values above one indicate a CUDA
advantage. Cached timings reuse symmetry and reflection topology; end-to-end
timings include all preprocessing.
\label{fig:scaling}](figures/scaling_speedup.pdf){width="100%"}

# Research impact statement

The package's near-term scholarly significance is supported by reproducible
correctness and performance measurements. The repository contains
the structure generators, oracle checks, raw timing samples, environment
metadata, and figure scripts used above. Continuous integration exercises two
Python versions, both diffraction modes, general and disordered structures,
automatic differentiation, and the public result objects. Its small public API,
Apache-2.0 license, typed package marker, pymatgen and optional ASE inputs, and
NumPy/PyTorch backends make the calculation reusable in established Python
materials workflows. The differentiable path additionally makes the same
validated physical model available to gradient-based fitting and
machine-learning pipelines without rewriting diffraction equations in a tensor
framework.

# AI usage disclosure

OpenAI GPT-5 assisted with code review and refactoring, validation and figure
workflow scaffolding, and editorial development of the manuscript.
The author made the scientific and architectural decisions and reviewed,
edited, and validated all retained outputs. Numerical claims are checked by
executable tests against pymatgen and by the versioned benchmark data included
in the repository.

# Acknowledgements

The author thanks the maintainers of pymatgen and spglib for their open
crystallographic software and data resources.

# References
