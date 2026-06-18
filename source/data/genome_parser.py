"""Parse .fna FASTA files — return top-N largest contigs truncated to max_len."""
import re
from pathlib import Path
from typing import List


def read_fna(fna_path: Path) -> List[str]:
    """Return list of raw contig sequences from a .fna FASTA file."""
    sequences, current = [], []
    with open(fna_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line.upper())
    if current:
        sequences.append("".join(current))
    return sequences


def clean_sequence(seq: str) -> str:
    """Replace any non-ATCGN character with N."""
    return re.sub(r"[^ATCGN]", "N", seq)


def get_top_contigs(
    fna_path: Path,
    top_n: int = 3,
    max_len: int = 131072,
) -> List[str]:
    """
    Parse a .fna file, select the top_n largest contigs, truncate each to
    max_len nucleotides (rounded down to nearest multiple of 128 as required
    by NTv3). Returns a list of cleaned DNA strings.
    """
    seqs = read_fna(fna_path)
    # Clean and filter out contigs that are too short for NTv3 (< 128 nt)
    seqs = [clean_sequence(s) for s in seqs if len(s) >= 128]
    # Sort descending by length, take top_n
    seqs = sorted(seqs, key=len, reverse=True)[:top_n]
    result = []
    for s in seqs:
        trunc_len = min(len(s), max_len)
        trunc_len = (trunc_len // 128) * 128  # must be multiple of 128
        if trunc_len >= 128:
            result.append(s[:trunc_len])
    return result
