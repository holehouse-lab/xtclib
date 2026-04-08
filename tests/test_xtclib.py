"""
Tests for xtclib – read/write/round-trip of XTC trajectory files.
"""

import os
import tempfile
import numpy as np
import pytest

from xtclib import read_xtc, write_xtc, XTCReader

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "example")
SYNUCLEIN_XTC = os.path.join(EXAMPLE_DIR, "synuclein_STARLING.xtc")
SYNUCLEIN_PDB = os.path.join(EXAMPLE_DIR, "synuclein_STARLING.pdb")
LYSOZYME_XTC = os.path.join(EXAMPLE_DIR, "lysozyme_md.xtc")


# ── helpers ──────────────────────────────────────────────────────────

def _round_trip(xtc_path, **write_kwargs):
    """Read an XTC, write it to a temp file, read it back."""
    orig = read_xtc(xtc_path)
    with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
        tmp = f.name
    try:
        write_xtc(tmp, orig["coords"], time=orig["time"],
                  step=orig["step"], box=orig["box"], **write_kwargs)
        rt = read_xtc(tmp)
    finally:
        os.unlink(tmp)
    return orig, rt


# ── synuclein read tests ─────────────────────────────────────────────

class TestSynucleinRead:

    def test_shape(self):
        data = read_xtc(SYNUCLEIN_XTC)
        assert data["coords"].shape == (400, 140, 3)
        assert data["natoms"] == 140

    def test_time_and_step(self):
        data = read_xtc(SYNUCLEIN_XTC)
        assert data["time"][0] == pytest.approx(0.0)
        assert data["time"][-1] == pytest.approx(399.0)
        assert data["step"][0] == 0
        assert data["step"][-1] == 399

    def test_box_shape(self):
        data = read_xtc(SYNUCLEIN_XTC)
        assert data["box"].shape == (400, 3, 3)

    def test_iterator(self):
        frames = []
        with XTCReader(SYNUCLEIN_XTC) as reader:
            for frame in reader:
                frames.append(frame)
        assert len(frames) == 400
        assert frames[0].natoms == 140
        assert frames[0].coords.shape == (140, 3)


# ── synuclein round-trip tests ───────────────────────────────────────

class TestSynucleinRoundTrip:

    def test_coords_exact(self):
        orig, rt = _round_trip(SYNUCLEIN_XTC)
        np.testing.assert_array_equal(orig["coords"], rt["coords"])

    def test_time_exact(self):
        orig, rt = _round_trip(SYNUCLEIN_XTC)
        np.testing.assert_allclose(orig["time"], rt["time"])

    def test_step_exact(self):
        orig, rt = _round_trip(SYNUCLEIN_XTC)
        np.testing.assert_array_equal(orig["step"], rt["step"])

    def test_box_exact(self):
        orig, rt = _round_trip(SYNUCLEIN_XTC)
        np.testing.assert_allclose(orig["box"], rt["box"])

    def test_file_size_matches(self):
        orig_size = os.path.getsize(SYNUCLEIN_XTC)
        orig = read_xtc(SYNUCLEIN_XTC)
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
            tmp = f.name
        try:
            write_xtc(tmp, orig["coords"], time=orig["time"],
                      step=orig["step"], box=orig["box"])
            rt_size = os.path.getsize(tmp)
        finally:
            os.unlink(tmp)
        assert rt_size == orig_size


# ── synuclein PDB cross-validation ───────────────────────────────────

class TestSynucleinPDB:

    @pytest.fixture(autouse=True)
    def _load(self):
        """Load PDB and XTC for comparison."""
        try:
            import mdtraj
        except ImportError:
            pytest.skip("mdtraj not installed")
        self.traj = mdtraj.load(SYNUCLEIN_PDB)
        self.xtc_data = read_xtc(SYNUCLEIN_XTC)

    def test_natoms_match_pdb(self):
        assert self.xtc_data["natoms"] == self.traj.n_atoms

    def test_first_frame_matches_pdb(self):
        # PDB should match the first frame of the XTC
        np.testing.assert_allclose(
            self.traj.xyz[0], self.xtc_data["coords"][0],
            atol=1e-3,
        )


# ── lysozyme read tests ──────────────────────────────────────────────

class TestLysozymeRead:

    def test_shape(self):
        data = read_xtc(LYSOZYME_XTC)
        assert data["coords"].shape == (101, 50949, 3)
        assert data["natoms"] == 50949

    def test_time_and_step(self):
        data = read_xtc(LYSOZYME_XTC)
        assert len(data["time"]) == 101
        assert len(data["step"]) == 101

    def test_box_nonzero(self):
        data = read_xtc(LYSOZYME_XTC)
        # Lysozyme should have a solvation box
        assert np.any(data["box"] != 0)


# ── lysozyme round-trip tests ────────────────────────────────────────

class TestLysozymeRoundTrip:

    def test_coords_exact(self):
        orig, rt = _round_trip(LYSOZYME_XTC)
        np.testing.assert_array_equal(orig["coords"], rt["coords"])

    def test_metadata_exact(self):
        orig, rt = _round_trip(LYSOZYME_XTC)
        np.testing.assert_allclose(orig["time"], rt["time"])
        np.testing.assert_array_equal(orig["step"], rt["step"])
        np.testing.assert_allclose(orig["box"], rt["box"])

    def test_file_size_matches(self):
        orig_size = os.path.getsize(LYSOZYME_XTC)
        orig = read_xtc(LYSOZYME_XTC)
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
            tmp = f.name
        try:
            write_xtc(tmp, orig["coords"], time=orig["time"],
                      step=orig["step"], box=orig["box"])
            rt_size = os.path.getsize(tmp)
        finally:
            os.unlink(tmp)
        assert rt_size == orig_size


# ── mdtraj cross-validation ──────────────────────────────────────────

class TestMdtrajCompat:
    """Verify our output is readable by mdtraj and matches."""

    @pytest.fixture(autouse=True)
    def _check_mdtraj(self):
        try:
            import mdtraj
            self.mdtraj = mdtraj
        except ImportError:
            pytest.skip("mdtraj not installed")

    def _make_top(self, natoms):
        top = self.mdtraj.Topology()
        chain = top.add_chain()
        for i in range(natoms):
            res = top.add_residue("UNK", chain, resSeq=i + 1)
            top.add_atom("CA", self.mdtraj.element.carbon, res)
        return top

    def test_synuclein_mdtraj_reads_our_file(self):
        orig = read_xtc(SYNUCLEIN_XTC)
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
            tmp = f.name
        try:
            write_xtc(tmp, orig["coords"], time=orig["time"],
                      step=orig["step"], box=orig["box"])
            top = self._make_top(140)
            ref = self.mdtraj.load_xtc(SYNUCLEIN_XTC, top=top)
            rt = self.mdtraj.load_xtc(tmp, top=top)
            np.testing.assert_array_equal(ref.xyz, rt.xyz)
        finally:
            os.unlink(tmp)

    def test_lysozyme_mdtraj_reads_our_file(self):
        orig = read_xtc(LYSOZYME_XTC)
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
            tmp = f.name
        try:
            write_xtc(tmp, orig["coords"], time=orig["time"],
                      step=orig["step"], box=orig["box"])
            top = self._make_top(50949)
            ref = self.mdtraj.load_xtc(LYSOZYME_XTC, top=top)
            rt = self.mdtraj.load_xtc(tmp, top=top)
            np.testing.assert_array_equal(ref.xyz, rt.xyz)
        finally:
            os.unlink(tmp)


# ── compression from scratch ─────────────────────────────────────────

class TestCompressFromScratch:
    """Write synthetic coords, read back, verify lossy precision."""

    def test_random_coords(self):
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((10, 50, 3)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
            tmp = f.name
        try:
            write_xtc(tmp, coords, precision=1000.0)
            rt = read_xtc(tmp)
            # XTC is lossy at 1/precision resolution
            np.testing.assert_allclose(
                coords, rt["coords"], atol=1.0 / 1000.0 + 1e-6
            )
        finally:
            os.unlink(tmp)

    def test_single_frame(self):
        coords = np.zeros((1, 20, 3), dtype=np.float32)
        coords[0, :, 0] = np.linspace(-5, 5, 20)
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
            tmp = f.name
        try:
            write_xtc(tmp, coords)
            rt = read_xtc(tmp)
            assert rt["coords"].shape == (1, 20, 3)
            np.testing.assert_allclose(
                coords, rt["coords"], atol=1.0 / 1000.0 + 1e-6
            )
        finally:
            os.unlink(tmp)
