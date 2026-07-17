# Diffraction Diagnostics and Differentiable Structural Refinement

**Working design and mathematics specification — version 0.2**

This is a research roadmap rather than a promise that every component will ship
in one release or support one publication. The first implementation and paper
should concentrate on lattice-compatible model diagnostics. Commensurate-cell
analysis, arbitrary-polymorph comparison, full experimental refinement, UI and
agent interfaces are later layers built only after the central diagnostics are
validated.

This document consolidates the main ideas developed so far into a concrete software and research specification. The proposed system uses a differentiable Bragg diffraction calculator as its physical forward model and adds:

1. model-to-model structural and diffraction diagnostics;
2. model-to-experimental-data refinement by automatic differentiation;
3. comparison of several candidate models against the same data;
4. quantitative refinement guidance based on sensitivities, parameter correlations and experimental discriminability;
5. an interactive UI and structured API/MCP interface suitable for both scientists and agents.

The working scientific premise is:

> A diffraction fit should not only report whether a model agrees with a pattern. It should explain which structural features control the agreement, where competing models are experimentally distinguishable, where information has been lost, and which refinement operation is justified next.

---

## 1. Non-technical description

### What the software does

A user provides one or more candidate crystal structures and, optionally, an experimental powder diffraction pattern.

The software can then answer questions such as:

- What would the diffraction pattern of this structure look like?
- How are these two structures different?
- Why do these structurally different models produce similar powder patterns?
- Which peaks actually distinguish the models?
- Are those distinguishing peaks resolvable with the current instrument?
- Which atoms, sites, occupancies or lattice parameters are responsible for the pattern differences?
- Which model agrees best with the experimental data?
- Which structural parameters should be refined next?
- Which parameters cannot be independently determined from the current data?
- Would another wavelength, radiation type or higher-resolution measurement distinguish the models better?

### What the software does **not** promise

The software is not initially intended to solve an arbitrary crystal structure from a random configuration using gradient descent alone.

Powder diffraction loses phase and directional information, peaks overlap, and the optimization landscape can contain many incorrect structures with similar calculated patterns. Direct unconstrained optimization is therefore not guaranteed to recover the correct structure.

The intended use is:

- simulation;
- explanation;
- comparison;
- local or moderately non-local refinement from scientifically plausible starting models;
- refinement planning;
- candidate discrimination.

---

# 2. Exact product definition

The system shall support four distinct modes.

## Mode A — Forward simulation and sensitivity analysis

### Input

- one structural model;
- experimental definition:
  - radiation type;
  - wavelength;
  - reciprocal-space or angular range;
  - instrumental peak profile;
  - optional size, strain and preferred-orientation parameters.

### Output

- reflection table;
- calculated structure factors;
- ideal stick pattern;
- broadened powder profile;
- parameter sensitivities;
- predicted changes under user-defined structural perturbations.

### Plain-language purpose

> “Show me what this structure should produce, and show me which parts of the pattern change when I move an atom or alter a parameter.”

No experimental pattern is required.

---

## Mode B — Model-to-model diagnostics

### Input

- two structural models;
- experimental definition;
- optional experimental pattern.

### Output

- structural relationship classification;
- structural difference report;
- diffraction-information-loss report;
- peak-by-peak explanation;
- local regions where the models are easy or difficult to distinguish;
- complex structure-factor mismatch disk when mathematically valid;
- Patterson/PDF or motif explanation where appropriate;
- suggested measurements that would distinguish the models.

### Plain-language purpose

> “Explain why these structures are different, why their diffraction patterns may nevertheless look similar, and where I should look to tell them apart.”

An experimental pattern is optional. Without data, the software reports theoretical distinguishability under an assumed experiment. With data, it also reports whether the observed data support either model.

---

## Mode C — Model-to-data refinement

### Input

- one starting structural model;
- one experimental diffraction pattern;
- experimental metadata;
- a declared set of refinable parameters and restraints.

### Output

- refined structural and profile parameters;
- calculated pattern and residual;
- conventional fit statistics such as \(R_{\mathrm{wp}}\) and \(\chi^2\);
- optimization history;
- gradient and sensitivity diagnostics;
- approximate parameter covariance and correlations;
- identifiability warnings;
- recommended next refinement operations;
- provenance record of every parameter change.

### Plain-language purpose

> “Fit this scientifically plausible model to my data, explain which changes improve the fit, and warn me when the data cannot determine a parameter.”

---

## Mode D — Multiple-model comparison against data

### Input

- two or more starting models;
- one experimental pattern;
- common experimental definition;
- model-specific and shared refinement settings.

### Output

- independent refined result for every model;
- standardized comparison of final fits;
- regions where each model succeeds or fails;
- pairwise model-difference diagnostics;
- uncertainty and robustness analysis;
- warnings when the data do not discriminate the candidates;
- optional model-selection statistics where assumptions permit them.

### Plain-language purpose

> “Refine all plausible models fairly, determine whether the experiment can distinguish them, and explain the evidence rather than merely selecting the lowest \(R_{\mathrm{wp}}\).”

---

# 3. Support for arbitrary structures

The interface should accept arbitrary periodic structures. However, the mathematically valid diagnostics depend on the relationship between the structures.

The system must first classify the pair into one of three regimes.

## Regime I — Equivalent or lattice-compatible structures

Examples:

- identical structures in different origins;
- equivalent crystallographic settings;
- same lattice with different atomic positions;
- same phase with different occupancies or displacement parameters;
- closely related isostructural models.

Available analyses:

- direct reflection correspondence;
- complex structure-factor comparison;
- amplitude–phase mismatch disk;
- atom/site/orbit attribution;
- powder and profile comparison;
- Patterson/PDF comparison.

This is the richest comparison regime.

## Regime II — Commensurate structures

Examples:

- primitive cell versus supercell;
- ordered versus disordered approximant;
- symmetry-lowered distortion;
- doubled or tripled repeat period.

Required preprocessing:

- find a common supercell or reciprocal-lattice representation;
- fold or unfold reflections consistently;
- identify superstructure reflections.

Available analyses:

- most Regime I analyses after transformation;
- explicit analysis of new, split or extinct reflections;
- superstructure sensitivity.

## Regime III — Unrelated or noncommensurate lattices

Examples:

- different polymorphs;
- unrelated space groups and unit cells;
- distinct packings;
- different numbers of formula units with no practical common cell.

There is no unique mapping

\[
h_A \leftrightarrow h_B
\]

and no intrinsic reflection-by-reflection phase difference.

Available analyses:

- calculated powder-profile comparison;
- peak and \(Q\)-window matching;
- Patterson and pair-distribution comparison;
- motif and local-environment matching;
- registered density/map comparison where feasible;
- experimental discriminability;
- counterfactual structural analysis.

Unavailable or explicitly disabled:

- direct \(hkl\)-wise phase-difference disk unless a valid reciprocal mapping has first been established.

The correct product claim is therefore:

> The software accepts arbitrary periodic structural models and automatically selects the strongest mathematically valid diagnostics for their crystallographic relationship.

---

# 4. The diffraction information ladder

The forward process can be written as

\[
\rho(\mathbf r)
\longrightarrow
F(\mathbf G)
\longrightarrow
|F(\mathbf G)|^2
\longrightarrow
I_{\mathrm{stick}}(Q)
\longrightarrow
I_{\mathrm{profile}}(Q).
\]

Each transformation can hide structural information.

1. **Complex structure factors \(F(\mathbf G)\)** retain calculated amplitudes and phases.
2. **Reflection intensities \(|F(\mathbf G)|^2\)** discard phase.
3. **Powder averaging** discards reciprocal-space direction and combines reflections with equal or similar \(Q\).
4. **Instrument and sample broadening** merge nearby powder features.
5. **Noise and background** can make surviving differences experimentally insignificant.

A central diagnostic output should state **where the models become similar**.

Example:

> The models have strongly different complex structure factors but similar reflection intensities. Their apparent similarity is therefore primarily caused by phase loss rather than instrumental broadening.

Or:

> The ideal powder stick patterns are distinguishable, but the supplied peak-width model merges the discriminating reflections. The ambiguity is primarily resolution-limited.

---

# 5. Differentiable forward model

## 5.1 Structural variables

Let the direct lattice matrix be

\[
\mathbf A =
\begin{bmatrix}
\mathbf a & \mathbf b & \mathbf c
\end{bmatrix}.
\]

For Miller index vector

\[
\mathbf h=(h,k,l)^\mathsf T,
\]

the reciprocal vector is

\[
\mathbf G_{\mathbf h}
=
2\pi \mathbf A^{-\mathsf T}\mathbf h,
\]

with magnitude

\[
Q_{\mathbf h}
=
\|\mathbf G_{\mathbf h}\|.
\]

Atom \(j\) has:

- fractional coordinate \(\mathbf x_j\);
- occupancy \(o_j\);
- scattering factor \(f_j(Q,\lambda)\);
- anisotropic displacement tensor \(\mathbf U_j\), or an isotropic equivalent.

## 5.2 Structure factor

A convenient Cartesian-\(\mathbf U\) convention is

\[
F_{\mathbf h}
=
\sum_j
o_j
f_j(Q_{\mathbf h},\lambda)
\exp\left(
-\frac{1}{2}
\mathbf G_{\mathbf h}^{\mathsf T}
\mathbf U_j
\mathbf G_{\mathbf h}
\right)
\exp\left(
2\pi i\,\mathbf h^\mathsf T\mathbf x_j
\right).
\]

Define the atomic contribution

\[
c_{\mathbf h j}
=
o_j
f_j(Q_{\mathbf h},\lambda)
T_{\mathbf h j}
\exp\left(
2\pi i\,\mathbf h^\mathsf T\mathbf x_j
\right),
\]

where

\[
T_{\mathbf h j}
=
\exp\left(
-\frac{1}{2}
\mathbf G_{\mathbf h}^{\mathsf T}
\mathbf U_j
\mathbf G_{\mathbf h}
\right).
\]

Then

\[
F_{\mathbf h}=\sum_j c_{\mathbf h j}.
\]

## 5.3 Integrated reflection intensity

A generic reflection contribution is

\[
A_{\mathbf h}
=
s\,
m_{\mathbf h}\,
L_{\mathbf h}\,
P_{\mathbf h}\,
C_{\mathbf h}\,
|F_{\mathbf h}|^2,
\]

where:

- \(s\) is the scale;
- \(m_{\mathbf h}\) is multiplicity;
- \(L_{\mathbf h}\) is a Lorentz factor;
- \(P_{\mathbf h}\) is polarization or related geometry correction;
- \(C_{\mathbf h}\) collects optional corrections such as preferred orientation.

The implementation should keep these factors modular because they depend on experiment type and geometry.

## 5.4 Powder profile

For observed coordinate \(Q_i\),

\[
I_{\mathrm{calc}}(Q_i)
=
B(Q_i;\boldsymbol\beta)
+
\sum_{\mathbf h}
A_{\mathbf h}
\,
\phi
\left(
Q_i-Q_{\mathbf h};
\boldsymbol\eta_{\mathbf h}
\right),
\]

where:

- \(B\) is the background;
- \(\phi\) is a normalized differentiable profile function;
- \(\boldsymbol\eta_{\mathbf h}\) contains width, shape and asymmetry parameters.

The first implementation can use a Gaussian or pseudo-Voigt profile. More complete instrumental models can be added later.

## 5.5 Important differentiability requirement

The reflection list must not change discontinuously during refinement.

A practical implementation should:

1. enumerate a fixed superset of Miller indices using a margin beyond the requested \(Q_{\max}\);
2. retain that index set throughout the optimization;
3. evaluate reflection positions continuously as the lattice changes;
4. apply a smooth range window near \(Q_{\min}\) and \(Q_{\max}\), rather than abruptly inserting or deleting reflections.

Otherwise, changing a lattice parameter can cause discontinuous reflection-list changes that invalidate or destabilize gradients.

---

# 6. Analytical gradients and automatic differentiation

The software may rely on PyTorch or another automatic-differentiation framework, but the analytical derivatives should be documented and tested.

## 6.1 Atomic coordinates

For fractional coordinate component \(x_{j\alpha}\),

\[
\frac{\partial F_{\mathbf h}}
{\partial x_{j\alpha}}
=
2\pi i\,h_\alpha\,c_{\mathbf h j}.
\]

## 6.2 Occupancy

Writing \(c_{\mathbf h j}=o_j d_{\mathbf h j}\),

\[
\frac{\partial F_{\mathbf h}}
{\partial o_j}
=
d_{\mathbf h j}.
\]

## 6.3 Intensity derivative

For any real parameter \(p\),

\[
\frac{\partial |F_{\mathbf h}|^2}{\partial p}
=
2\operatorname{Re}
\left[
F_{\mathbf h}^*
\frac{\partial F_{\mathbf h}}{\partial p}
\right].
\]

## 6.4 Full-profile derivative

\[
\frac{\partial I_{\mathrm{calc}}(Q_i)}{\partial p}
=
\sum_{\mathbf h}
\left[
\frac{\partial A_{\mathbf h}}{\partial p}
\phi_{i\mathbf h}
+
A_{\mathbf h}
\frac{\partial \phi_{i\mathbf h}}{\partial p}
\right]
+
\frac{\partial B(Q_i)}{\partial p}.
\]

For structural parameters, both reflection amplitude and position may vary. Lattice parameters affect:

- \(Q_{\mathbf h}\);
- the atomic form factor evaluation;
- displacement factors;
- Lorentz/polarization terms;
- peak centre and potentially width.

Automatic differentiation is particularly useful because these dependencies are coupled.

---

# 7. Model-to-model mismatch disk

This diagnostic is available only after valid reciprocal-lattice and origin alignment.

Let

\[
a_{\mathbf h}=|F_A(\mathbf h)|,
\qquad
b_{\mathbf h}=|F'_B(\mathbf h)|,
\]

where \(F'_B\) is model \(B\) transformed into the aligned setting and relative origin.

Define the wrapped phase difference

\[
\Delta\phi_{\mathbf h}
=
\operatorname{wrap}
\left[
\arg F'_B(\mathbf h)-\arg F_A(\mathbf h)
\right].
\]

The disk coordinates are

\[
x_{\mathbf h}
=
\frac{
b_{\mathbf h}-a_{\mathbf h}
}{
a_{\mathbf h}+b_{\mathbf h}+\varepsilon
},
\]

and

\[
y_{\mathbf h}
=
\frac{
2\sqrt{a_{\mathbf h}b_{\mathbf h}}
}{
a_{\mathbf h}+b_{\mathbf h}+\varepsilon
}
\sin
\left(
\frac{\Delta\phi_{\mathbf h}}{2}
\right).
\]

They satisfy

\[
x_{\mathbf h}^2+y_{\mathbf h}^2
=
\frac{
|F'_B(\mathbf h)-F_A(\mathbf h)|^2
}{
\left(
|F_A(\mathbf h)|+|F'_B(\mathbf h)|+\varepsilon
\right)^2
}
\leq 1.
\]

Define the per-reflection radius

\[
r_{\mathbf h}
=
\sqrt{x_{\mathbf h}^2+y_{\mathbf h}^2}.
\]

Interpretation:

- centre: amplitude and phase agreement;
- left/right: candidate intensity too weak/strong;
- vertical direction: signed phase disagreement;
- boundary: extinction mismatch or severe complex disagreement.

## 7.1 Scalar structure-factor dissimilarity

With normalized weights

\[
\sum_{\mathbf h}w_{\mathbf h}=1,
\]

define

\[
D_{\mathrm{SF}}
=
\min_{\mathcal T}
\sqrt{
\sum_{\mathbf h}
w_{\mathbf h}
r_{\mathbf h}^2
},
\]

where \(\mathcal T\) denotes valid crystallographic alignment transformations, including:

- equivalent cell settings;
- symmetry operations;
- relative origin;
- common-supercell mappings when supported.

The score decomposes as

\[
D_{\mathrm{amp}}
=
\sqrt{
\sum_{\mathbf h}w_{\mathbf h}x_{\mathbf h}^2
},
\]

\[
D_{\mathrm{phase}}
=
\sqrt{
\sum_{\mathbf h}w_{\mathbf h}y_{\mathbf h}^2
},
\]

and therefore

\[
D_{\mathrm{SF}}^2
=
D_{\mathrm{amp}}^2
+
D_{\mathrm{phase}}^2.
\]

This decomposition is one of the strongest proposed mathematical features.

## 7.2 Reflection weighting

Equal reflection weights overemphasize large numbers of weak high-\(Q\) reflections. Pure intensity weights can allow a few intense low-\(Q\) reflections to dominate.

A starting proposal is shell-balanced intensity weighting.

For resolution shell \(s\), define

\[
\bar I_{\mathbf h}
=
\frac{
a_{\mathbf h}^2+b_{\mathbf h}^2
}{2},
\]

and

\[
w_{\mathbf h}
=
\frac{1}{N_{\mathrm{shell}}}
\frac{
\bar I_{\mathbf h}
}{
\sum_{\mathbf g\in s(\mathbf h)}
\bar I_{\mathbf g}
}.
\]

This gives equal total influence to every resolution shell while weighting reflections by strength inside the shell.

This weighting is a research choice and must be benchmarked against alternatives.

## 7.3 Interpretation and numerical caveats

Until its metric properties have been proved, \(D_{\mathrm{SF}}\) shall be
called a **dissimilarity**, not a distance metric.

Phase is poorly defined when either amplitude is close to zero. The
implementation must therefore report the amplitude threshold or regularizer,
and must test the sensitivity of the result to this choice. Extinction
mismatches remain meaningful through the disk radius, but their vertical phase
coordinate must not be overinterpreted.

The wrapped phase convention has a sign discontinuity at
\(\Delta\phi=\pm\pi\). The radius and scalar dissimilarities are stable there;
the plotted vertical sign is not. Friedel mates, origin choices and equivalent
settings must also be handled consistently and stated in result metadata.

---

# 8. Alignment and invariance requirements

A model-to-model score is meaningful only if arbitrary representation choices are removed.

The software must test invariance to:

- atom ordering;
- unit-cell origin;
- equivalent symmetry operations;
- equivalent conventional settings;
- primitive/conventional representation;
- allowed supercell transformations;
- rigid permutation of chemically equivalent sites.

For a relative translation \(\mathbf t\),

\[
F'_B(\mathbf h;\mathbf t)
=
F_B(\mathbf h)
\exp\left(
2\pi i\,\mathbf h^\mathsf T\mathbf t
\right).
\]

The optimal relative origin can be found by minimizing \(D_{\mathrm{SF}}\) or maximizing a weighted complex correlation.

For unrelated lattices, the software must not invent an \(hkl\)-phase correspondence. It must switch to powder-, pair- or density-based diagnostics.

---

# 9. Why two models produce similar diffraction

The software should compute a set of similarities at successive information levels:

\[
\mathcal S
=
\left(
S_{\mathrm{geometry}},
S_{\mathrm{complex}},
S_{\mathrm{intensity}},
S_{\mathrm{Patterson}},
S_{\mathrm{powder}},
S_{\mathrm{profile}}
\right).
\]

Exact normalizations require further development, but the transition pattern is diagnostic.

## 9.1 Phase-loss signature

\[
S_{\mathrm{intensity}}\approx1,
\qquad
S_{\mathrm{complex}}\ll1.
\]

Meaning:

- calculated amplitudes are similar;
- calculated phases differ;
- diffraction intensities hide the structural difference.

On the disk, points are approximately vertical:

\[
x_{\mathbf h}\approx0,
\qquad
|y_{\mathbf h}|>0.
\]

This includes homometric or near-homometric behavior.

## 9.2 Powder-averaging signature

\[
S_{\mathrm{powder}}
>
S_{\mathrm{intensity}}.
\]

Meaning:

- the three-dimensional reflection sets differ;
- different reciprocal vectors collapse to similar \(Q\)-positions;
- powder averaging hides directional differences.

## 9.3 Peak-overlap signature

The ideal stick patterns differ, but broadened profiles are similar:

\[
S_{\mathrm{profile}}
>
S_{\mathrm{powder}}.
\]

Meaning:

- distinguishing reflections exist;
- the experiment cannot resolve them under the supplied broadening model.

## 9.4 Common-pair signature

High Patterson or PDF similarity indicates similar scattering-weighted interatomic vectors.

Meaning:

- different global structures may share the same heavy-atom framework;
- common layer or stacking distances dominate;
- common coordination shells generate similar signal;
- different connectivities may nevertheless share similar pair distributions.

---

# 10. Peak-group diagnostics

The software should group reflections according to experimental resolution.

For peak group \(g\),

\[
I_g(Q)
=
\sum_{\mathbf h\in g}
A_{\mathbf h}
\phi(Q-Q_{\mathbf h}).
\]

A group can be defined using a resolution-dependent criterion such as

\[
|Q_{\mathbf h}-Q_g|
<
c\,\mathrm{FWHM}(Q_g).
\]

For each peak group, report:

- contributing reflections;
- multiplicities;
- integrated intensities;
- fraction of total peak intensity from each reflection;
- whether the two models use the same or different reflection sets;
- whether component differences cancel in the summed profile;
- whether a distinguishing reflection is buried under a common strong peak.

Suggested overlap descriptors include:

### Effective number of contributing reflections

With normalized contributions

\[
p_{\mathbf h}
=
\frac{A_{\mathbf h}}
{\sum_{\mathbf g\in g} A_{\mathbf g}},
\]

define

\[
N_{\mathrm{eff}}
=
\exp\left(
-\sum_{\mathbf h\in g}
p_{\mathbf h}\log p_{\mathbf h}
\right).
\]

A large \(N_{\mathrm{eff}}\) indicates a strongly mixed powder feature.

### Peak-group model difference

\[
\Delta I_g
=
\int
\left|
I_{A,g}(Q)-I_{B,g}(Q)
\right|
\,dQ.
\]

This can be standardized by experimental uncertainty.

---

# 11. Experimental discriminability

The theoretical profile difference alone is insufficient. Differences must be scaled by expected measurement uncertainty.

For profile point \(i\),

\[
d_i
=
\frac{
I_A(Q_i)-I_B(Q_i)
}{
\sigma_{\mathrm{eff}}(Q_i)
},
\]

where \(\sigma_{\mathrm{eff}}\) may include:

- counting uncertainty;
- background uncertainty;
- calibration uncertainty;
- modelled instrumental resolution;
- optionally sample-related uncertainty.

Define local discriminating information

\[
\mathcal D_i=d_i^2.
\]

For statistically independent measured bins, the total expected separation is

\[
\mathcal D_{\mathrm{total}}
=
\sum_i d_i^2.
\]

Under a Gaussian observation model, this is closely related to an expected \(\Delta\chi^2\) between the two fixed model profiles.

This sum is meaningful only when \(\sigma_i\) is the uncertainty of the actual
measured bin. It must not change merely because a calculated curve is sampled
on a finer plotting grid. With correlated calibration, background or profile
errors, use the full covariance model

\[
\mathcal D_{\mathrm{total}}
=
\Delta\boldsymbol\mu^\mathsf T
\boldsymbol\Sigma^{-1}
\Delta\boldsymbol\mu
\]

instead of a diagonal pointwise approximation.

Interpretation:

- low \(\mathcal D_i\): the models are effectively indistinguishable there;
- high \(\mathcal D_i\): the region can discriminate them;
- large theoretical difference but low \(\mathcal D_i\): difference exists but is not measurable with the current data quality.

The UI should show aligned plots of:

1. \(I_A\), \(I_B\) and optionally \(I_{\mathrm{obs}}\);
2. \(I_A-I_B\);
3. \(\mathcal D(Q)\);
4. reflection/peak-group contributions.

---

# 12. Refinement objective

## 12.1 Gaussian weighted least squares

For observed values \(y_i\), calculated values \(\mu_i(\boldsymbol\theta)\), and weights \(w_i\),

\[
\chi^2(\boldsymbol\theta)
=
\sum_i
w_i
\left[
y_i-\mu_i(\boldsymbol\theta)
\right]^2.
\]

The conventional weighted-profile residual is

\[
R_{\mathrm{wp}}
=
\sqrt{
\frac{
\sum_i
w_i
\left[
y_i-\mu_i
\right]^2
}{
\sum_i w_i y_i^2
}
}.
\]

The software should report \(R_{\mathrm{wp}}\), but it should not use \(R_{\mathrm{wp}}\) alone as evidence that a structure is correct.

## 12.2 Poisson likelihood

For raw counts, a Poisson negative log-likelihood is often more natural:

\[
\mathcal L_{\mathrm{Poisson}}
=
\sum_i
\left[
\mu_i-y_i\log(\mu_i+\varepsilon)
\right]
+
\mathrm{constant}.
\]

## 12.3 Constrained total objective

A practical objective is

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{data}}
+
\lambda_{\mathrm{geom}}\mathcal L_{\mathrm{geom}}
+
\lambda_{\mathrm{sym}}\mathcal L_{\mathrm{sym}}
+
\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}
+
\lambda_{\mathrm{ADP}}\mathcal L_{\mathrm{ADP}}
+
\lambda_{\mathrm{prior}}\mathcal L_{\mathrm{prior}}.
\]

Possible restraints:

- bond-length and angle restraints;
- minimum interatomic distances;
- composition and site-occupancy constraints;
- shared occupancies;
- symmetry constraints;
- positive-definite displacement tensors;
- parameter priors;
- smooth background;
- physically valid cell geometry.

Every restraint contribution should be reported separately.

---

# 13. Parameterization and constraints

Unconstrained atom-by-atom optimization is unsafe and difficult to interpret. Parameters should be transformed into physically valid domains.

## Lattice parameters

Possible parameterizations:

- logarithms for positive lengths;
- bounded transforms for angles;
- symmetry-restricted independent lattice parameters;
- positive-definite metric tensor representation.

The space-group lattice restrictions must be enforced unless the user explicitly enables symmetry breaking.

## Fractional coordinates

Refine only independent Wyckoff parameters by default. Generate symmetry-equivalent positions in the forward pass.

Fractional parameters can remain unwrapped during optimization because the phase term is periodic; coordinates are mapped into \([0,1)\) only for display and export.

## Occupancies

Use a sigmoid or constrained simplex.

For competing species on one site:

\[
\mathbf o_j
=
\operatorname{softmax}(\mathbf z_j),
\]

so occupancies are non-negative and sum to one.

## Isotropic displacement parameters

Use a positive transform such as

\[
B_j
=
\operatorname{softplus}(b_j)+B_{\min}.
\]

## Anisotropic displacement tensors

Parameterize through a Cholesky factor:

\[
\mathbf U_j
=
\mathbf L_j\mathbf L_j^\mathsf T,
\]

ensuring positive semidefiniteness.

## Scale and phase fractions

Use positive or simplex transforms:

\[
s=\operatorname{softplus}(z_s),
\]

and for phase fractions,

\[
\boldsymbol\pi
=
\operatorname{softmax}(\mathbf z_\pi).
\]

---

# 14. Differentiable refinement strategy

Recent work has shown that direct powder-pattern gradient descent can encounter highly non-convex landscapes and spurious peak-overlap minima. The software should therefore use structured, symmetry-aware and staged optimization rather than unrestricted simultaneous refinement.

## Recommended staged workflow

### Stage 0 — Data and profile preparation

Refine or estimate:

- scale;
- background;
- zero shift;
- wavelength/calibration corrections;
- peak-width and shape parameters.

Keep the structure fixed.

### Stage 1 — Lattice refinement

Refine:

- symmetry-allowed cell parameters;
- zero shift jointly if necessary.

Keep atomic coordinates fixed.

### Stage 2 — Coarse structural refinement

Refine:

- selected positional groups;
- rigid bodies;
- motif translations or rotations;
- high-sensitivity Wyckoff parameters.

Use strong restraints.

### Stage 3 — Occupancy and displacement refinement

Refine only parameters supported by sensitivity and identifiability diagnostics.

Do not refine highly correlated occupancy and displacement parameters together unless the data support both.

### Stage 4 — Limited joint refinement

Jointly optimize the supported parameter subset.

### Stage 5 — Validation

Evaluate:

- stability under perturbed starting points;
- held-out \(Q\)-ranges;
- parameter uncertainty;
- alternative candidate models;
- structural plausibility;
- residual structure.

## Coarse-to-fine continuation

One promising stabilization method is to begin with artificially broader peaks or a smoothed profile and gradually approach the true experimental resolution.

Let \(\gamma_0>\gamma_1>\cdots>\gamma_T\) be a sequence of peak widths. Optimize

\[
\mathcal L_{\gamma_0}
\rightarrow
\mathcal L_{\gamma_1}
\rightarrow
\cdots
\rightarrow
\mathcal L_{\gamma_T}.
\]

This may reduce early sensitivity to peak-assignment discontinuities and local overlap minima. It must be tested, not assumed effective.

## Optimizers

The framework should support:

- Adam for robust early movement;
- L-BFGS or Gauss–Newton-like methods for local convergence;
- multistart refinement;
- trust-region or damped least-squares updates;
- optional global or discrete search for cell/setting/model choices.

Backpropagation supplies derivatives; it does not imply that plain gradient descent is always the correct optimizer.

---

# 15. Mathematical refinement guidance

The differentiable forward model provides a Jacobian

\[
J_{ij}
=
\frac{
\partial \mu_i
}{
\partial p_j
},
\]

where \(p_j\) is a refinement parameter.

Let

\[
\mathbf W
=
\operatorname{diag}(w_i),
\qquad
\mathbf r
=
\mathbf y-\boldsymbol\mu.
\]

## 15.1 Parameter sensitivity

Define

\[
s_j
=
\sqrt{
\mathbf J_j^\mathsf T
\mathbf W
\mathbf J_j
}.
\]

A low \(s_j\) means that changing parameter \(p_j\) barely changes the calculated pattern under the current experiment.

The numerical value of \(s_j\) depends on the units and parameterization of
\(p_j\). Sensitivities may be ranked across unlike parameters only after the
parameters are made dimensionless using a declared characteristic step, bound,
prior scale or transformed-coordinate convention.

## 15.2 Residual support

Define the signed residual projection

\[
e_j
=
\frac{
\mathbf J_j^\mathsf T
\mathbf W
\mathbf r
}{
\sqrt{
\mathbf J_j^\mathsf T
\mathbf W
\mathbf J_j
}+\varepsilon
}.
\]

Interpretation:

- large \(|e_j|\): the current residual has a component consistent with changing \(p_j\);
- \(e_j\approx0\): the residual does not support that parameter direction;
- the sign indicates the locally preferred direction under the chosen parameterization.

This is local guidance, not proof that the parameter should be changed globally.

## 15.3 Approximate normal/Fisher matrix

\[
\mathbf H
=
\mathbf J^\mathsf T
\mathbf W
\mathbf J.
\]

Under local linear and Gaussian assumptions,

\[
\operatorname{Cov}(\hat{\mathbf p})
\approx
\mathbf H^{-1}.
\]

The parameter correlation matrix is

\[
\rho_{jk}
=
\frac{
\operatorname{Cov}_{jk}
}{
\sqrt{
\operatorname{Cov}_{jj}
\operatorname{Cov}_{kk}
}
}.
\]

The UI should identify:

- insensitive parameters;
- strongly correlated parameter pairs;
- near-null parameter combinations;
- unstable covariance estimates;
- high condition numbers.

## 15.4 Where a parameter is measured

Define local information for parameter \(p_j\):

\[
\mathcal I_{ij}
=
w_i J_{ij}^2.
\]

Plotting \(\mathcal I_{ij}\) over \(Q_i\) shows which pattern regions constrain that parameter.

For a group of parameters \(G\),

\[
\mathcal I_{i,G}
=
w_i
\sum_{j\in G}J_{ij}^2
\]

is a simple first visualization, though correlated-group information should use the full matrix.

## 15.5 Which parameters explain the difference between two models?

Let

\[
\Delta\boldsymbol\mu_{AB}
=
\boldsymbol\mu_B-\boldsymbol\mu_A.
\]

For one parameter \(p_j\), define the alignment score

\[
C_j
=
\frac{
\left(
\mathbf J_j^\mathsf T
\mathbf W
\Delta\boldsymbol\mu_{AB}
\right)^2
}{
\left(
\mathbf J_j^\mathsf T
\mathbf W
\mathbf J_j
\right)
\left(
\Delta\boldsymbol\mu_{AB}^\mathsf T
\mathbf W
\Delta\boldsymbol\mu_{AB}
\right)
+\varepsilon
}.
\]

Then

\[
0\leq C_j\leq1.
\]

Interpretation:

- high \(C_j\): varying \(p_j\) changes the pattern in a direction similar to the difference between models \(A\) and \(B\);
- low \(C_j\): \(p_j\) does not explain that model difference.

For a parameter group \(G\), define the weighted projection

\[
\mathbf P_G
=
\mathbf J_G
\left(
\mathbf J_G^\mathsf T
\mathbf W
\mathbf J_G
\right)^{+}
\mathbf J_G^\mathsf T
\mathbf W,
\]

where \(+\) denotes the pseudoinverse.

The fraction of the model difference locally explainable by group \(G\) is

\[
C_G
=
\frac{
\Delta\boldsymbol\mu_{AB}^\mathsf T
\mathbf W
\mathbf P_G
\Delta\boldsymbol\mu_{AB}
}{
\Delta\boldsymbol\mu_{AB}^\mathsf T
\mathbf W
\Delta\boldsymbol\mu_{AB}
+\varepsilon
}.
\]

This allows statements such as:

> Seventy percent of the local diffraction difference between the models lies in the sensitivity subspace of the oxygen coordinates, while only five percent is aligned with the lattice parameters.

The percentage must be described as a **local linear explainability score**,
not a causal truth. Scores from overlapping parameter groups are not generally
additive and must not be presented as parts summing to 100%. An additive
decomposition requires an explicitly ordered orthogonalization or a more
expensive attribution method such as Shapley averaging.

## 15.6 Guidance logic

The system can classify a parameter as:

### Recommended to refine

- appreciable sensitivity \(s_j\);
- strong residual support \(|e_j|\);
- acceptable correlation with active parameters;
- physically valid refinement direction.

### Potentially discriminative but unsupported by current residual

- high \(C_j\);
- low \(|e_j|\).

Meaning:

> This parameter distinguishes the candidate models in theory, but the current experiment does not favor moving the present model in that direction.

### Poorly identifiable

- low sensitivity;
- high variance;
- strong correlations;
- instability under small changes in data range or starting point.

### Unsafe to release jointly

Two parameters have nearly collinear Jacobian columns:

\[
\frac{
\mathbf J_j^\mathsf T\mathbf W\mathbf J_k
}{
s_js_k
}
\approx \pm1.
\]

Example:

> Occupancy and \(B_{\mathrm{iso}}\) affect the available reflections almost identically and should not be refined simultaneously from this dataset.

---

# 16. Structural attribution

The complex structure-factor difference is

\[
\Delta F_{\mathbf h}
=
F_{B,\mathbf h}-F_{A,\mathbf h}
=
\sum_j
\Delta c_{\mathbf h j}.
\]

Where atom or site mappings exist, contributions can be grouped by:

- atom;
- element;
- crystallographic site;
- Wyckoff orbit;
- molecule or rigid body;
- layer;
- coordination polyhedron;
- user-defined motif.

The tool should provide both:

1. exact complex contribution differences;
2. sensitivity-based approximations.

## Counterfactual analysis

Construct hybrid or intervened models to ask:

- What if model \(A\) had the lattice of \(B\)?
- What if only one site position were replaced?
- What if only occupancies changed?
- What if the displacement parameters were exchanged?
- What if one motif were translated or rotated?

Define a counterfactual change

\[
\Delta I^{(g)}
=
I(\mathcal M_{A\leftarrow B}^{(g)})
-
I(\mathcal M_A),
\]

where group \(g\) is replaced by its value from \(B\).

Because contribution estimates can depend on replacement order, a later version may use Shapley-style averaging across intervention orders. This is computationally expensive but conceptually attractive for motif-level explanations.

---

# 17. Model selection and several starting models

When several models are refined against the same data, the software must not rank them solely by final \(R_{\mathrm{wp}}\).

It should report:

- \(R_{\mathrm{wp}}\);
- \(\chi^2\) or likelihood;
- number of effective free parameters;
- restraint penalties;
- structural plausibility;
- convergence stability across restarts;
- parameter uncertainty;
- residual autocorrelation or structured misfit;
- performance on held-out \(Q\)-ranges;
- pairwise experimental discriminability;
- whether differences are below the information content of the data.

Where likelihood assumptions and effective parameter counts are credible, AIC or BIC may be reported. They should not be treated as universally valid for strongly restrained, correlated or non-linear crystallographic models.

A legitimate conclusion may be:

> The experiment does not discriminate models A and B under the supplied resolution and uncertainty model.

This is preferable to forcing a winner.

---

# 18. Measurement guidance

The differentiable model can also guide data collection.

For two models, local discrimination is

\[
\mathcal D_i
=
\frac{
\left(
\mu_{A,i}-\mu_{B,i}
\right)^2
}{
\sigma_i^2
}.
\]

For a parameter \(p_j\), local information is

\[
\mathcal I_{ij}
=
\frac{
1
}{
\sigma_i^2
}
\left(
\frac{\partial\mu_i}{\partial p_j}
\right)^2.
\]

By recomputing these quantities under alternative experimental settings, the tool can compare:

- X-ray versus neutron contrast;
- different wavelengths;
- anomalous-edge measurements;
- wider \(Q\)-range;
- improved resolution;
- increased counting statistics;
- selected angular windows.

Possible output:

> A resolution improvement from 0.12° to 0.05° FWHM would separate the two principal discriminating reflections near \(Q=3.1\ \text{Å}^{-1}\).

Or:

> Neutron diffraction provides substantially higher information for the light-atom coordinate than Cu Kα X-ray diffraction.

These are predictions under the supplied forward and uncertainty models.

---

# 19. UI specification

The UI should expose the mathematics rather than hide it.

## Main workspace

### Structure panel

- synchronized 3D views of model A, model B and refined model;
- selectable atom/site/orbit/motif;
- difference vectors;
- cell and symmetry information;
- structural mapping confidence.

### Pattern panel

- observed and calculated profiles;
- model A and model B profiles;
- residual;
- local discriminability;
- selected parameter sensitivity;
- reflection tick marks;
- peak-group decomposition.

### Complex mismatch panel

Enabled only for compatible structures:

- amplitude–phase disk;
- colour by \(Q\), element sensitivity or reflection family;
- point size by weight or intensity;
- filters by resolution shell;
- click point to highlight corresponding profile region and atoms.

### Refinement panel

- active/fixed parameter groups;
- physical bounds and restraints;
- optimizer controls;
- staged recipe;
- live loss history;
- \(R_{\mathrm{wp}}\), likelihood and restraint components;
- gradient norms;
- warnings.

### Identifiability panel

- parameter sensitivity ranking;
- uncertainty;
- correlation heatmap;
- near-null parameter combinations;
- \(Q\)-regions supporting each parameter;
- recommended next refinement step.

### Explanation panel

Generate structured explanations with explicit evidence:

> The largest remaining residual is associated with reflections sensitive to the \(c\)-axis. The \(c\)-axis derivative aligns strongly with the residual and is weakly correlated with the active profile parameters. Refining \(c\) is therefore supported.

The explanation must link to numerical quantities and selected regions.

---

# 20. Programmatic API and MCP specification

The physics engine, diagnostics and UI should be separated. BraggCalculator can act as the differentiable forward backend, while a diagnostics/refinement layer exposes stable high-level operations.

## Core callable operations

### `simulate_pattern`

Input:

- structure;
- experiment;
- profile parameters;
- requested outputs.

Output:

- reflection table;
- complex structure factors;
- integrated intensities;
- calculated profile;
- metadata and warnings.

### `compare_models`

Input:

- structure A;
- structure B;
- experiment;
- comparison options.

Output:

- structural relationship regime;
- alignment transformation;
- structural differences;
- profile differences;
- disk metrics when valid;
- Patterson/PDF metrics;
- peak-group diagnostics;
- discriminating regions.

### `refine_model`

Input:

- starting structure;
- observed data;
- experiment;
- parameter specification;
- restraints;
- optimization schedule.

Output:

- refined structure;
- refined nuisance/profile parameters;
- optimization trace;
- fit statistics;
- uncertainties;
- diagnostics;
- provenance.

### `refine_models`

Input:

- list of candidate structures;
- shared observed data and experiment;
- refinement policy.

Output:

- independent refinement result for every candidate;
- standardized comparison;
- pairwise diagnostics;
- non-discrimination warning where appropriate.

### `analyze_sensitivity`

Input:

- model;
- experiment;
- selected parameters;
- optional observed residual.

Output:

- Jacobian summaries;
- sensitivity ranking;
- local information;
- correlations;
- residual support;
- recommended active set.

### `diagnose_difference`

Input:

- two structures or two refined models;
- optional observed data.

Output:

- explanation of where structural differences appear or disappear;
- atom/site/motif attributions;
- experimentally accessible discriminators.

### `suggest_measurement`

Input:

- competing models or target parameters;
- candidate experiment configurations.

Output:

- expected discrimination/information for every configuration;
- ranked recommendation;
- assumptions.

## MCP tool behavior

MCP-facing tools should return structured JSON, not only prose.

Every result should include:

- numerical outputs;
- units;
- formula or metric name;
- assumptions;
- warnings;
- provenance;
- artifact references for plots and tables;
- machine-readable parameter paths.

An agent must be able to ask:

- “Which parameter should be refined next?”
- “Why are these two models not distinguishable?”
- “Show the three most informative \(Q\)-regions.”
- “Refine only parameters supported by the current data.”
- “Compare X-ray and neutron experiments for discriminating these models.”

The agent should not be permitted to silently release all structural parameters without an explicit policy.

---

# 21. Suggested software architecture

```text
braggcalculator/
├── core/
│   ├── lattice.py
│   ├── symmetry.py
│   ├── structure_factor.py
│   ├── reflection_set.py
│   ├── corrections.py
│   └── profile.py
├── experiment/
│   ├── xray.py
│   ├── neutron.py
│   ├── geometry.py
│   └── uncertainty.py
├── compare/
│   ├── relationship.py
│   ├── alignment.py
│   ├── mismatch_disk.py
│   ├── powder_similarity.py
│   ├── peak_groups.py
│   ├── patterson.py
│   └── attribution.py
├── refine/
│   ├── parameters.py
│   ├── constraints.py
│   ├── losses.py
│   ├── schedule.py
│   ├── optimize.py
│   ├── covariance.py
│   └── guidance.py
├── api/
│   ├── schemas.py
│   ├── service.py
│   └── mcp.py
├── ui/
├── tests/
├── MATH.md
└── notebooks/
    ├── 00_single_reflection_gradients.ipynb
    ├── 01_synthetic_coordinate_refinement.ipynb
    ├── 02_lattice_refinement.ipynb
    ├── 03_mismatch_disk.ipynb
    ├── 04_parameter_identifiability.ipynb
    └── 05_two_model_discrimination.ipynb
```

The exact repository layout can differ, but the separation of concerns is important.

For the current BraggCalculator repository, new packages should not be created
prematurely. The first implementation should extend the existing
`structure_factor.py`, `results.py` and `core.py`, adding a small diagnostics
module. Dedicated `compare/`, `refine/`, `experiment/` and `api/` packages
should appear only when their public interfaces and internal responsibilities
have become clear.

## Canonical mathematics location

The repository should contain a version-controlled `MATH.md` that defines:

- conventions;
- symbols;
- forward equations;
- normalization;
- losses;
- gradients;
- diagnostics;
- invariances;
- assumptions.

All implementation functions should link to equation identifiers in `MATH.md`.

The notebooks should be disposable experiments. `MATH.md` should be the canonical specification.

---

# 22. Numerical and scientific tests

## 22.1 Gradient tests

For every refinable parameter:

- compare autograd derivative to central finite differences;
- test real and imaginary structure-factor derivatives;
- test full-profile derivatives;
- test CPU/GPU consistency;
- test single and double precision.

A typical criterion might be

\[
\frac{
\|g_{\mathrm{AD}}-g_{\mathrm{FD}}\|
}{
\|g_{\mathrm{FD}}\|+\varepsilon
}
<\tau,
\]

with parameter-specific tolerances.

## 22.2 Invariance tests

The comparison score must be unchanged by:

- atom permutation;
- origin translation;
- equivalent crystallographic setting;
- symmetry-equivalent coordinate representation;
- primitive/conventional cell conversion;
- supported common-supercell representation.

## 22.3 Synthetic recovery tests

Generate exact synthetic data and perturb:

- lattice;
- coordinates;
- occupancies;
- \(B\)-factors;
- scale;
- background;
- peak width;
- zero shift.

Test:

- recovery radius;
- convergence rate;
- uncertainty calibration;
- parameter correlations;
- failure modes.

## 22.4 Ambiguity tests

Use:

- homometric pairs;
- near-homometric pairs;
- polymorphs with similar powder patterns;
- structures differing only in weak scatterers;
- superstructure variants;
- resolution-induced ambiguity.

The tool should correctly identify **why** patterns are similar.

## 22.5 Real-data validation

Compare against established refinement software and manually curated expert analyses.

Validate:

- profile calculation;
- refined parameter values;
- conventional residuals;
- uncertainty;
- diagnostic usefulness;
- reproducibility.

---

# 23. Minimum viable implementation

## Existing foundation — Differentiable mathematical kernel

BraggCalculator already provides:

- one phase;
- arbitrary periodic input reduced through crystallographic preprocessing;
- X-ray and neutron scattering;
- fixed reflection list;
- Gaussian peaks;
- differentiable lattice, expanded atomic coordinates, occupancies and
  isotropic displacement parameters;
- NumPy/PyTorch parity and an initial coordinate-gradient test.

The remaining foundation work is:

- expose complex structure factors without duplicating kernel work;
- make profile and nuisance parameters refinable;
- add comprehensive analytical and finite-difference gradient tests;
- distinguish expanded-atom tensors from symmetry-preserving refinement
  parameters.

## Prototype 1 — Lattice-compatible model diagnostics

Add:

- cell/origin alignment for equivalent structures;
- mismatch disk;
- \(D_{\mathrm{SF}}\), \(D_{\mathrm{amp}}\), \(D_{\mathrm{phase}}\);
- profile discriminability;
- peak-group decomposition;
- atom/site attribution.

## Prototype 2 — Jacobian diagnostics and symmetry-aware local refinement

Add:

- normalized parameter sensitivities;
- residual support, covariance and identifiability;
- independent Wyckoff/orbit parameters;
- symmetry-preserving lattice refinement;
- constrained occupancies and isotropic displacement parameters;
- differentiable profile, scale and background parameters;
- staged optimization on synthetic data.

## Prototype 3 — Experimental refinement and guidance

Add:

- experimental import and bin-level uncertainty models;
- pseudo-Voigt and instrument/profile parameters;
- background, scale, zero-shift and calibration refinement;
- multi-model refinement;
- refinement guidance;
- validation against reference software and real data.

## Prototype 4 — Arbitrary-structure comparison

Add:

- commensurate-cell detection;
- supercell comparison;
- unrelated-lattice powder comparison;
- Patterson/PDF diagnostics;
- motif analysis.

## Prototype 5 — UI, service and MCP

Add:

- interactive views;
- REST/Python API;
- structured explanation objects;
- MCP tools;
- audit and provenance.

---

# 24. Initial Codex coding tasks

A practical first coding sequence should extend the validated kernel through
small, executable vertical slices.

## Task 1 — Expose differentiable complex structure factors

Implement:

```python
def structure_factors(
    hkl: Tensor,              # [n_reflections, 3], integer
    frac_coords: Tensor,      # [n_atoms, 3]
    occupancies: Tensor,      # [n_atoms]
    form_factors: Tensor,     # [n_reflections, n_atoms]
    debye_waller: Tensor,     # [n_reflections, n_atoms]
) -> Tensor:                  # complex [n_reflections]
    ...
```

Validate

\[
\frac{\partial F_{\mathbf h}}
{\partial x_{j\alpha}}
=
2\pi i h_\alpha c_{\mathbf h j}.
\]

## Task 2 — Differentiable Gaussian powder profile

Implement a fixed reflection list and

```python
def powder_profile(
    q_grid: Tensor,
    q_reflections: Tensor,
    integrated_intensities: Tensor,
    sigma: Tensor,
    background: Tensor,
) -> Tensor:
    ...
```

## Task 3 — Synthetic coordinate refinement

- generate a small P1 structure;
- simulate a noiseless pattern;
- perturb coordinates slightly;
- refine coordinates with strong bounds/restraints;
- compare Adam and L-BFGS;
- record structural and profile errors.

## Task 4 — Jacobian diagnostics

Calculate

\[
J_{ij}=\partial I_i/\partial p_j
\]

using automatic differentiation and report:

- \(s_j\);
- parameter correlations;
- local information \(\mathcal I_{ij}\);
- condition number.

## Task 5 — Mismatch disk

For two same-cell models:

- optimize relative origin;
- calculate \(x_h,y_h,r_h\);
- verify the unit-disk identity;
- calculate amplitude and phase components;
- test invariance under origin shifts.

These five tasks should reveal whether the central mathematics behaves as expected before implementing a large UI.

---

# 25. Publication framing

The strongest research contribution is not simply:

- a complex-plane plot;
- another powder similarity score;
- automatic differentiation by itself;
- another automated refinement recipe.

The stronger contribution is:

> A differentiable and interpretable diffraction-analysis framework connecting structural parameters, complex structure factors, powder peak overlap, experimental information and refinement decisions.

Potential claims, subject to validation:

1. a bounded per-reflection amplitude–phase diagnostic;
2. decomposition of reciprocal-space disagreement into amplitude and phase components;
3. automated identification of the stage at which structural differences become hidden;
4. peak-group attribution explaining powder-pattern degeneracy;
5. parameter-level refinement guidance from differentiable sensitivities and identifiability;
6. unified human and agent interface with explicit mathematical provenance.

A potential paper title is:

> **Differentiable Diffraction Diagnostics: Explaining Structural Ambiguity and Guiding Powder Refinement**

The first methods paper should prioritize the bounded mismatch diagnostic,
alignment invariance, the diffraction information ladder, peak/profile
discriminability and structural attribution for lattice-compatible models.
Full experimental refinement may be included as a limited demonstration, but
should become a separate contribution if validating it would dilute the central
diagnostic claims.

---

# 26. Prior art and novelty boundaries

The following areas already exist and must be acknowledged.

## Established areas

- Rietveld whole-profile refinement;
- Patterson functions and pair-vector analysis;
- Argand/complex-plane structure-factor illustrations;
- atom-level structure-factor contribution diagrams for selected reflections;
- powder-pattern similarity metrics;
- arbitrary crystal-structure geometry matching;
- automated refinement recipes;
- machine-learning-guided refinement;
- map comparison;
- differentiable or gradient-based diffraction optimization.

## Closest visual precedent

Owen and Sherrell showed atom-by-atom and grouped structure-factor contributions in Argand diagrams to explain effects such as radiation damage and derivatization.

The proposed mismatch disk differs by aiming to provide:

- a bounded normalized point for every corresponding reflection;
- a global decomposable score;
- links from disk points to powder peaks and structural contributors;
- comparison across the complete diffraction information ladder.

## Important optimization warning

Segal and co-workers found that powder-XRD structure optimization can have a highly non-convex and ill-posed loss landscape, with spurious peak overlaps allowing structurally incorrect states to give similar patterns.

Therefore, this software should emphasize:

- plausible starting models;
- symmetry-aware parameterizations;
- staged refinement;
- restraints;
- multistart or hybrid optimization;
- explicit ambiguity reporting.

It should not market backpropagation as a complete solution to the powder inverse problem.

---

# 27. Open mathematical questions

The following require experimentation or proof.

1. Is \(D_{\mathrm{SF}}\) a true metric after minimization over crystallographic equivalences, including the triangle inequality?
2. What reflection weighting gives the most useful agreement with structural and expert judgments?
3. What is the best relative-origin optimization algorithm?
4. How should commensurate supercell reflection weights be normalized?
5. Which powder similarity should serve as the default for unrelated cells?
6. How should model, background and instrument uncertainty be combined in \(\sigma_{\mathrm{eff}}\)?
7. Does coarse-to-fine peak broadening improve the optimization landscape?
8. Which parameterization best preserves symmetry while remaining numerically stable?
9. How should restraint strengths be selected and reported?
10. How reliable is the local covariance approximation for strongly non-linear refinements?
11. How should parameter-group contributions be attributed when Jacobian columns are strongly correlated?
12. Can counterfactual or Shapley-style attribution produce stable and useful explanations?
13. What experimental design metric best predicts successful model discrimination?
14. How should multiple phases and phase fractions be incorporated without introducing severe non-identifiability?
15. Can complex structure-factor diagnostics improve candidate selection even though experimental powder data do not supply phases?

---

# 28. Core references

1. Patterson, A. L. “A Fourier Series Method for the Determination of the Components of Interatomic Distances in Crystals.” *Physical Review* **46**, 372–376 (1934). DOI: [10.1103/PhysRev.46.372](https://doi.org/10.1103/PhysRev.46.372).

2. Rietveld, H. M. “A Profile Refinement Method for Nuclear and Magnetic Structures.” *Journal of Applied Crystallography* **2**, 65–71 (1969). DOI: [10.1107/S0021889869006558](https://doi.org/10.1107/S0021889869006558).

3. Read, R. J. “Structure-Factor Probabilities for Related Structures.” *Acta Crystallographica A* **46**, 900–912 (1990). DOI: [10.1107/S0108767390005529](https://doi.org/10.1107/S0108767390005529).

4. de Gelder, R., Wehrens, R. & Hageman, J. A. “A Generalized Expression for the Similarity of Spectra: Application to Powder Diffraction Pattern Classification.” *Journal of Computational Chemistry* **22**, 273–289 (2001). DOI: [10.1002/1096-987X(200102)22:3<273::AID-JCC1001>3.0.CO;2-0](https://doi.org/10.1002/1096-987X(200102)22:3%3C273::AID-JCC1001%3E3.0.CO;2-0).

5. Chisholm, J. A. & Motherwell, S. “COMPACK: A Program for Identifying Crystal Structure Similarity Using Distances.” *Journal of Applied Crystallography* **38**, 228–231 (2005). DOI: [10.1107/S0021889804027074](https://doi.org/10.1107/S0021889804027074).

6. McCoy, A. J. et al. “Phaser Crystallographic Software.” *Journal of Applied Crystallography* **40**, 658–674 (2007). DOI: [10.1107/S0021889807021206](https://doi.org/10.1107/S0021889807021206).

7. Baake, M. & Grimm, U. “Kinematic Diffraction Is Insufficient to Distinguish Order from Disorder.” *Physical Review B* **79**, 020203 (2009). DOI: [10.1103/PhysRevB.79.020203](https://doi.org/10.1103/PhysRevB.79.020203).

8. Tian, P. et al. “SrRietveld: A Program for Automating Rietveld Refinements for High-Throughput Powder Diffraction Studies.” *Journal of Applied Crystallography* **46**, 255–258 (2013). DOI: [10.1107/S0021889812045967](https://doi.org/10.1107/S0021889812045967).

9. Urzhumtsev, A. et al. “Metrics for Comparison of Crystallographic Maps.” *Acta Crystallographica D* **70**, 2593–2606 (2014). DOI: [10.1107/S1399004714016289](https://doi.org/10.1107/S1399004714016289).

10. Owen, R. L. & Sherrell, D. A. “Radiation Damage and Derivatization in Macromolecular Crystallography: A Structure Factor’s Perspective.” *Acta Crystallographica D* **72**, 388–394 (2016). DOI: [10.1107/S2059798315021555](https://doi.org/10.1107/S2059798315021555).

11. Habermehl, S., Schlesinger, C. & Prill, D. “Comparison and Evaluation of Pair Distribution Functions, Using a Similarity Measure Based on Cross-Correlation Functions.” *Journal of Applied Crystallography* **54**, 612–623 (2021). DOI: [10.1107/S1600576721001722](https://doi.org/10.1107/S1600576721001722).

12. Hicks, D. et al. “AFLOW-XtalFinder: A Reliable Choice to Identify Crystalline Prototypes.” *npj Computational Materials* **7**, 30 (2021). DOI: [10.1038/s41524-020-00483-4](https://doi.org/10.1038/s41524-020-00483-4).

13. Mayo, R. A., Otero-de-la-Roza, A. & Johnson, E. R. “Development and Assessment of an Improved Powder-Diffraction-Based Method for Molecular Crystal Structure Similarity.” *CrystEngComm* **24**, 8326–8338 (2022). DOI: [10.1039/D2CE01080A](https://doi.org/10.1039/D2CE01080A).

14. Schlesinger, C. et al. “Ambiguous Structure Determination from Powder Data: Four Different Structural Models of 4,11-Difluoroquinacridone with Similar X-Ray Powder Patterns, Fit to the PDF, SSNMR and DFT-D.” *IUCrJ* **9**, 406–424 (2022). DOI: [10.1107/S2052252522004237](https://doi.org/10.1107/S2052252522004237).

15. Mayo, R. A., Marczenko, K. M. & Johnson, E. R. “Quantitative Matching of Crystal Structures to Experimental Powder Diffractograms.” *Chemical Science* **14**, 4777–4785 (2023). DOI: [10.1039/D3SC00168G](https://doi.org/10.1039/D3SC00168G).

16. Toby, B. H. “A Simple Solution to the Rietveld Refinement Recipe Problem.” *Journal of Applied Crystallography* **57**, 175–180 (2024). DOI: [10.1107/S1600576723011032](https://doi.org/10.1107/S1600576723011032).

17. Zhang, Z., Shen, Y. & Sun, J. “Understanding Extended Homometry Based on Complementary Crystallographic Orbit Sets.” *Acta Crystallographica A* **80**, 151–160 (2024). DOI: [10.1107/S205327332400007X](https://doi.org/10.1107/S205327332400007X).

18. Brookner, D. E. & Hekstra, D. R. “MatchMaps: Non-Isomorphous Difference Maps for X-Ray Crystallography.” *Journal of Applied Crystallography* **57**, 885–895 (2024). DOI: [10.1107/S1600576724003510](https://doi.org/10.1107/S1600576724003510).

19. Otero-de-la-Roza, A. “Powder-Diffraction-Based Structural Comparison for Crystal Structure Prediction without Prior Indexing.” *Journal of Applied Crystallography* **57**, 1415–1425 (2024). DOI: [10.1107/S1600576724006721](https://doi.org/10.1107/S1600576724006721).

20. Segal, N., Subramanian, A., Li, M., Miller, B. K. & Gómez-Bombarelli, R. “The Loss Landscape of Powder X-Ray Diffraction-Based Structure Optimization Is Too Rough for Gradient Descent.” *Digital Discovery* **5**, 1590–1599 (2026). DOI: [10.1039/D6DD00017G](https://doi.org/10.1039/D6DD00017G).

21. Mun, S. J., Nam, Y. & Choi, S. “Automation of Rietveld Refinement through Machine Learning.” *Journal of Applied Crystallography* **59**, 564–577 (2026). DOI: [10.1107/S1600576726001494](https://doi.org/10.1107/S1600576726001494).

---

# 29. Immediate implementation sequence

Build and demonstrate the work in this order:

1. expose complex \(F_{\mathbf h}\) from the existing differentiable kernel;
2. define matched-reflection and crystallographic-alignment result objects;
3. implement the mismatch disk with identity, origin and permutation tests;
4. add bin-aware profile discrimination and visible model-difference examples;
5. add parameter-scaled Jacobian, residual-support and identifiability diagnostics;
6. introduce symmetry-aware constrained refinement parameters that generate
   expanded atomic tensors for the forward pass;
7. add experimental nuisance/profile parameters and staged optimization.

Every step must include a small executable example, numerical assertions and a
human-readable interpretation. Only after these seven steps work should the
design expand to commensurate/unrelated cells, rich attribution, web UI and MCP
access.

## 29.1 Current implementation status

- Steps 1–3 are implemented with complex structure factors, matched-reflection
  results, relative-origin alignment and the mismatch disk.
- Step 4 is implemented for expected measured-bin values with either diagonal
  variances or a full covariance matrix. A calculator convenience function
  converts profile density into bin counts under an explicit synthetic count
  model.
- Step 5 is implemented for declared characteristic parameter scales, diagonal
  or correlated observation uncertainty, residual support, local information,
  rank, condition number, column correlations and generalized covariance.
- Step 6 is implemented for local symmetry-compatible coordinates: site
  stabilizers define the allowed independent displacement subspaces, and fixed
  orbit operations expand them to all scattering contributions. Symmetry-aware
  lattice, occupancy and displacement-factor parameterizations remain future
  extensions of this layer.
- Step 7 is implemented for differentiable scale, constant background, zero
  shift and Gaussian FWHM. Positive quantities use constrained transforms, and
  a declared-stage Adam runner can release structural and nuisance groups in a
  recorded order. More realistic backgrounds, pseudo-Voigt/asymmetric profiles
  and instrument-specific corrections remain required for experimental work.

---

# 30. Experimental characterization implementation

The first end-to-end materials-characterization slice now implements roadmap
layers 8–14 within a narrow scope:

- immutable XY/XYE datasets with uncertainty, masks, metadata and SHA-256
  provenance;
- wavelength components, pseudo-Voigt peaks, positive Caglioti widths and
  polynomial background;
- cautious/quick release policies, coordinate restraints, held-out bins and
  deterministic multistart refinement;
- observed/calculated/residual diagnostics, informative regions, candidate
  ranking and explicit non-discrimination conclusions;
- Python API, command-line entry point and self-contained HTML report;
- synthetic candidate-selection validation and a public NIST SRM 660c LaB6
  real-data regression.

The NIST regression is a pipeline validation, not a certification-quality
refinement. Its residual warning is expected because the implemented model
does not reproduce the full NIST fundamental-parameters instrument model.
