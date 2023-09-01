"""Calculators for density related properties."""
from __future__ import annotations

import contextlib
import io
from inspect import isclass
from typing import TYPE_CHECKING

import numpy as np
from ase import optimize, units
from ase.constraints import ExpCellFilter
from ase.geometry.cell import Cell
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize.optimize import Optimizer

from .base import PropCalc
from .relaxation import TrajectoryObserver

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


class DensityCalc(PropCalc):
    """Relaxes and run NPT simulations to compuate the density of structures."""
    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        steps: int = 500,
        interval: int = 1,
        fmax: float = 0.1,
        mask: list | np.ndarray | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        out_stem: str | None = None,
    ):
        """
        Initialize the Density Calculator.

        Args:
            calculator (Calculator): Calculator to use.
            optimizer (Optimizer | str): Optimizer to use. Defaults to "FIRE".
            steps (int, optional): Number of steps to run the relaxation. Defaults to 500.
            interval (int, optional): Interval to save the trajectory. Defaults to 1.
            fmax (float, optional): Maximum force to stop the relaxation. Defaults to 0.1.
            mask (list | np.ndarray | None, optional): Mask allowing cell parameter relaxation. Defaults to None.
            rtol (float, optional): Relative tolerance for the NPT simulation, in the unit of eV/A^3. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance for the NPT simulation. Defaults to 1e-5.
            out_stem (str | None, optional): Filename to save the trajectory. Defaults to None.
        """
        self.calculator = calculator

        # check str is valid optimizer key
        def is_ase_optimizer(key):
            return isclass(obj := getattr(optimize, key)) and issubclass(obj, Optimizer)

        valid_keys = [key for key in dir(optimize) if is_ase_optimizer(key)]
        if isinstance(optimizer, str) and optimizer not in valid_keys:
            raise ValueError(f"Unknown {optimizer=}, must be one of {valid_keys}")

        self.optimizer = optimizer
        self.steps = steps
        self.interval = interval
        self.fmax = fmax
        self.mask = mask
        self.rtol = rtol
        self.atol = atol
        self.out_stem = out_stem

    def calc(
        self,
        atoms: Atoms,
        temperature: float,
        externalstress: float | np.ndarray,
        timestep: float = 2.0,
        pfactor: float | None = None,
    ) -> dict:
        """Relax the structure and run NPT simulations to compute the density.

        Args:
            atoms (Atoms): Structure to relax.
            temperature (float): Temperature of the simulation in Kelvin.
            externalstress: External pressure of the simulation in eV/A^3.
            timestep (float, optional): Timestep of the simulation in fs. Defaults to 1.0.
            pfactor (float | None, optional): Pressure factor of the simulation. Defaults to None.

        Returns:
            Atoms: Relaxed structure.
        """
        # relax the structure
        atoms.calc = self.calculator
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.mask is not None:
                atoms = ExpCellFilter(atoms, mask=self.mask)
            optimizer = self.optimizer(atoms)
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)
            if self.out_stem is not None:
                obs()
                obs.save(f"{self.out_stem}-relax.pkl")
            del obs
        if self.mask is not None:
            atoms = atoms.atoms

        # run NPT simulation
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms, preserve_temperature=True)

        if not pfactor:
            B = 1 * units.GPa  # bulk modulus
            ptime = 75 * units.fs  # suggested by ase
            pfactor = ptime**2 * B

        dyn = NPT(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            externalstress=externalstress,
            pfactor=pfactor,
            mask=self.mask,
        )

        converged = False
        restart = 0
        while not converged:
            if self.out_stem is not None:
                traj = Trajectory(f"{self.out_stem}-npt-{restart}.traj", "w", atoms)
                dyn.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            dyn.attach(obs, interval=self.interval)
            dyn.run(steps=self.steps)

            stress = np.mean(np.stack(obs.stresses, axis=0), axis=0)
            converged = np.allclose(dyn.externalstress, stress, atol=self.atol, rtol=self.rtol)
            # TODO: check if the energy is converged as well

            if self.out_stem is not None:
                traj.close()
                obs()
                obs.save(f"{self.out_stem}-npt-{restart}.pkl")

            if not converged:
                print(
                    f"Pressure not converged, restarting simulation. Current pressure: {stress} eV/A^3. "
                    f"Target pressure: {dyn.externalstress} eV/A^3."
                    )
                dyn.observers.clear()
                del obs
                restart += 1

        volumes = [Cell(matrix).volume for matrix in obs.cells]
        vol_avg, vol_std = np.mean(volumes), np.std(volumes)
        erg_avg, erg_std = np.mean(obs.energies), np.std(obs.energies)

        return {
            "volume_avg": vol_avg,
            "volume_std": vol_std,
            "atomic_density": atoms.get_global_number_of_atoms() / vol_avg,
            "mass_density": atoms.get_masses().sum() / vol_avg,
            "energy_avg": erg_avg,
            "energy_std": erg_std,
        }
