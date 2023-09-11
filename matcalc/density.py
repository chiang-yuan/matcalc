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
        ttime: float | None = 25.0,
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

        # step 1: run NVT simulation

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms, preserve_temperature=True)

        nvt = NPT(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            externalstress=0,
            ttime=ttime * units.fs if ttime else None,
            pfactor=None, # disable barostat
        )

        converged = False
        restart = 0
        last_erg_avg, first_erg_avg = None, None
        while not converged:
            if self.out_stem is not None:
                traj = Trajectory(f"{self.out_stem}-nvt-{restart}.traj", "w", atoms)
                nvt.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            nvt.attach(obs, interval=self.interval)
            nvt.run(steps=self.steps)

            erg_avg, erg_std = np.mean(obs.energies), np.std(obs.energies)

            if last_erg_avg is None or first_erg_avg is None:
                last_erg_avg = erg_avg
                first_erg_avg = erg_avg
                converged = False
            else:
                converged = (
                    abs(erg_avg - last_erg_avg)/last_erg_avg  < self.rtol
                    and np.sign(erg_avg - first_erg_avg) * (erg_avg - last_erg_avg) < 0
                    )

            if self.out_stem is not None:
                traj.close()
                obs()
                obs.save(f"{self.out_stem}-nvt-{restart}.pkl")

            if not converged:
                print(
                    f"Energy not converged, restarting simulation.\n"
                    f"Current relative deviation: {(erg_avg - last_erg_avg)/last_erg_avg*100} %. \n" if last_erg_avg != erg_avg else "\n"  # noqa: E501
                    f"Target relative deviation: {self.rtol*100} %."
                    )
                nvt.observers.clear()
                del obs
                last_erg_avg = erg_avg
                restart += 1

        # step 2: relax using exponential cell matrix method

        with contextlib.redirect_stdout(stream):
            optimizer = self.optimizer(atoms)

            if self.mask is not None:
                ecf = ExpCellFilter(atoms, mask=self.mask)

            if self.out_stem is not None:
                traj = Trajectory(f"{self.out_stem}-relax-{restart}.traj", "w", atoms)
                optimizer.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)
            if self.out_stem is not None:
                traj.close()
                obs()
                obs.save(f"{self.out_stem}-relax.pkl")

        if self.mask is not None:
            atoms = ecf.atoms

        # step 3: run NPT simulation

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms, preserve_temperature=True)

        if not pfactor:
            B = 1 * units.GPa  # bulk modulus
            ptime = 75 * units.fs  # suggested by ase
            pfactor = ptime**2 * B

        npt = NPT(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            externalstress=externalstress,
            ttime=ttime * units.fs if ttime else None,
            pfactor=pfactor,
            mask=self.mask,
        )

        converged, erg_converged, str_converged = False, False, False
        restart = 0
        last_erg_avg, first_erg_avg = None, None
        while not converged:
            if self.out_stem is not None:
                traj = Trajectory(f"{self.out_stem}-npt-{restart}.traj", "w", atoms)
                npt.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            npt.attach(obs, interval=self.interval)
            npt.run(steps=self.steps)

            erg_avg, erg_std = np.mean(obs.energies), np.std(obs.energies)

            if last_erg_avg is None or first_erg_avg is None:
                last_erg_avg = erg_avg
                first_erg_avg = erg_avg
                erg_converged = False
            else:
                erg_converged = (
                    abs(erg_avg - last_erg_avg)/last_erg_avg  < self.rtol
                    and np.sign(erg_avg - first_erg_avg) * (erg_avg - last_erg_avg) < 0
                    )

            stress = np.mean(np.stack(obs.stresses, axis=0), axis=0)
            str_converged = np.allclose(npt.externalstress, stress, atol=self.atol, rtol=self.rtol)

            converged = erg_converged and str_converged

            if self.out_stem is not None:
                traj.close()
                obs()
                obs.save(f"{self.out_stem}-npt-{restart}.pkl")

            if not converged:
                print(
                    f"Energy or stress not converged, restarting simulation.\n"
                    f"Current relative energy deviation: {(erg_avg - last_erg_avg)/last_erg_avg*100} %.\n" if last_erg_avg != erg_avg else "\n"  # noqa: E501
                    f"Target relative energy deviation: {self.rtol*100} %.\n"
                    f"Current pressure: {stress} eV/A^3.\n"
                    f"Target pressure: {npt.externalstress} eV/A^3."
                    )
                npt.observers.clear()
                del obs
                last_erg_avg = erg_avg
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
