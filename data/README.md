# Experimental validation data

`nist_srm660c_100a_full.xye` contains all 5332 measured rows from the NIST SRM
660c LaB6 certification dataset, scan `100a`, spanning corrected 2-theta values
from 20.300 through 150.908 degrees. The source scan uses separated angular
windows and adaptive step sizes, which are retained exactly.

`nist_srm660c_100a_20-50.xye` is the original small pipeline-validation excerpt.
It is retained so historical results remain reproducible, but the full scan is
used for lattice validation because high-angle reflections are essential for
separating lattice scale from angular offsets and specimen displacement.

The third column is derived from the published least-squares weight using

\[
\sigma = 1 / \sqrt{w}.
\]

Source archive:
<https://data.nist.gov/od/ds/mds2-2315/srm_660c_cifs_20201029_081700.zip>

Original archive member:
`660_cert_cif_mosaic_consensus_100a.cif`

NIST dataset identifier: `ark:/88434/mds2-2315`  
License: <https://www.nist.gov/open/license>  
Certification DOI: <https://doi.org/10.1017/S0885715620000068>

The certification reports a lattice parameter of 4.156826 Å at 22.5 °C with a
95% expanded uncertainty of 0.000080 Å. The structure CIF here is transcribed
from the structural block supplied in the NIST pdCIF; its starting cell is
4.156780 Å.

The vendored measured pattern is intended for executable ingestion,
provenance, and real-data regression tests. The NIST archive remains the
authoritative source, and BraggCalculator's compact profile model is not a
replacement for NIST's full fundamental-parameters refinement.
