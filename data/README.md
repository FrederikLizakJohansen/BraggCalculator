# Experimental validation data

`nist_srm660c_100a_20-50.xye` contains exact measured rows from the NIST SRM
660c LaB6 certification dataset, scan `100a`, restricted to corrected 2-theta
values from 20.300 through 50.000 degrees. The source scan uses separated
angular windows, which are retained exactly.

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

The excerpt is intended for executable ingestion, provenance, and limited
real-data regression tests. It is not a replacement for the complete NIST
dataset or a full fundamental-parameters refinement.
