"""Relaxation properties."""
from __future__ import annotations

import contextlib
import io
import pickle
from inspect import isclass
from typing import TYPE_CHECKING

from ase import optimize
from ase.constraints import ExpCellFilter
from ase.optimize.optimize import Optimizer
from pymatgen.io.ase import AseAtomsAdaptor

from .base import PropCalc

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress(include_ideal_gas=True))
        # Stress tensor should include the contribution from the momenta, otherwise
        # during MD simulattion the stress tensor ignores the effect of kinetic part,
        # leanding to the discrepancy between applied pressure and the stress tensor.
        # For more details, see: https://gitlab.com/ase/ase/-/merge_requests/1500
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)


class RelaxCalc(PropCalc):
    """Relaxes and computes the relaxed parameters of a structure."""

    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        fmax: float = 0.1,
        relax_cell: bool = True,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator to use.
            optimizer (str | ase Optimizer): The optimization algorithm. Defaults to "FIRE".
            steps (int): Max number of steps for relaxation. Defaults to 500.
            traj_file (str | None): File to save the trajectory to. Defaults to None.
            interval (int): The step interval for saving the trajectories. Defaults to 1.
            fmax (float): Total force tolerance for relaxation convergence.
                fmax is a sum of force and stress forces. Defaults to 0.1 (eV/A).
            relax_cell (bool): Whether to relax the cell (or just atoms).

        Raises:
            ValueError: If the optimizer is not a valid ASE optimizer.
        """
        self.calculator = calculator

        # check str is valid optimizer key
        def is_ase_optimizer(key):
            return isclass(obj := getattr(optimize, key)) and issubclass(obj, Optimizer)

        valid_keys = [key for key in dir(optimize) if is_ase_optimizer(key)]
        if isinstance(optimizer, str) and optimizer not in valid_keys:
            raise ValueError(f"Unknown {optimizer=}, must be one of {valid_keys}")

        self.optimizer: Optimizer = getattr(optimize, optimizer) if isinstance(optimizer, str) else optimizer
        self.fmax = fmax
        self.interval = interval
        self.steps = steps
        self.traj_file = traj_file
        self.relax_cell = relax_cell

    def calc(self, structure: Structure) -> dict:
        """
        Perform relaxation to obtain properties.

        Args:
            structure: Pymatgen structure.

        Returns: {
            final_structure: final_structure,
            energy: trajectory observer final energy in eV,
            volume: lattice.volume in A^3,
            a: lattice.a in A,
            b: lattice.b in A,
            c: lattice.c in A,
            alpha: lattice.alpha in degrees,
            beta: lattice.beta in degrees,
            gamma: lattice.gamma in degrees,
        }
        """
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.calculator
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.optimizer(atoms)
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)
            if self.traj_file is not None:
                obs()
                obs.save(self.traj_file)
        if self.relax_cell:
            atoms = atoms.atoms

        final_structure = AseAtomsAdaptor.get_structure(atoms)
        lattice = final_structure.lattice

        return {
            "final_structure": final_structure,
            "energy": obs.energies[-1],
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
            "volume": lattice.volume,
        }
