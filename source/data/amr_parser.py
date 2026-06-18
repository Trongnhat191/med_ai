"""Parse pre-aligned AMR gene files from data/amr_genes_aligned/.

Each file (named <PATRIC_ID>) is a fixed-length (96 KB) string of DNA:
  - Genes separated by long N-runs (alignment padding)
  - Each gene slot: real sequence (ATCG) or all-N (gene absent in this genome)

Returns only the *present* gene slots (non-N regions) as cleaned DNA strings,
ready to be tokenised and forwarded through NTv3.
"""
import re
from pathlib import Path
from typing import List


# Minimum number of consecutive Ns to count as a separator between slots.
# Using 500 so that short internal N-runs within a gene are not split.
_N_SEP_PATTERN = re.compile(r"N{500,}")


def read_amr_aligned(path: Path) -> str:
    """Read the raw sequence string from an AMR-aligned file (no FASTA header)."""
    with open(path, "r") as f:
        return f.read().strip().upper()


def get_amr_gene_slots(
    path: Path,
    min_len: int = 100,
) -> List[str]:
    """Return the present AMR gene slots from an aligned genome file.

    Each slot is a cleaned DNA string (non-N residues remain; short internal
    N-runs from the alignment itself are kept as-is since NTv3 handles N).

    Args:
        path:    Path to the amr_genes_aligned file (no extension, named by PATRIC ID).
        min_len: Minimum nucleotide length of a slot to be considered *present*.
                 Slots shorter than this are assumed absent / mostly padding.

    Returns:
        List of DNA strings, one per present AMR gene slot.  May be empty if the
        genome has no called AMR genes (all-N file).
    """
    seq = read_amr_aligned(path)

    # Split on long N-runs to get individual gene slots
    slots = _N_SEP_PATTERN.split(seq)

    # Filter: keep slots that are long enough to represent a real gene
    present = [s for s in slots if len(s) >= min_len]

    # Trim any residual leading/trailing Ns within each slot (short padding artefacts)
    present = [s.strip("N") for s in present]
    present = [s for s in present if len(s) >= min_len]

    return present


def amr_path_for_genome(patric_id: str, amr_dir: Path) -> Path:
    """Resolve the amr_genes_aligned path for a given PATRIC ID.

    The files are named exactly by PATRIC ID (e.g. '573.12862'), no extension.
    """
    return amr_dir / patric_id
