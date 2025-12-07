"""
Boundary Conditions for Seismic Wave Propagation

This module implements various boundary conditions used in
seismic modeling:
- Absorbing boundaries (to prevent reflections)
- Free surface (air-rock interface)
- Perfectly Matched Layers (PML)

Proper boundary treatment is critical for realistic wave propagation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import math


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.

    Boundary conditions determine how waves behave at the edges
    of the computational domain.
    """

    @abstractmethod
    def apply(
        self,
        coords: Tensor,
        field: Tensor,
        derivatives: Dict[str, Tensor],
        domain_bounds: Tuple[Tuple[float, float], ...],
    ) -> Tensor:
        """
        Apply boundary condition and return residual.

        Args:
            coords: Coordinates of boundary points
            field: Field values at boundary
            derivatives: Field derivatives at boundary
            domain_bounds: ((x_min, x_max), (z_min, z_max), ...)

        Returns:
            Boundary condition residual (should be zero)
        """
        pass

    @abstractmethod
    def get_boundary_mask(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
        thickness: float = 0.01,
    ) -> Tensor:
        """
        Get mask identifying boundary points.

        Args:
            coords: All coordinates
            domain_bounds: Domain extent
            thickness: Boundary layer thickness

        Returns:
            Boolean mask (True for boundary points)
        """
        pass


class AbsorbingBoundary(BoundaryCondition):
    """
    First-Order Absorbing Boundary Condition (Clayton-Engquist).

    The simplest absorbing boundary assumes waves exit normally:

        ∂p/∂t + v * ∂p/∂n = 0

    where n is the outward normal direction. This works well for
    normally-incident waves but causes reflections at oblique angles.

    For PINN: We enforce this as a soft constraint in the loss function.

    Args:
        absorption_coefficient: Strength of absorption (0-1)
    """

    def __init__(self, absorption_coefficient: float = 1.0) -> None:
        self.absorption_coefficient = absorption_coefficient

    def apply(
        self,
        coords: Tensor,
        field: Tensor,
        derivatives: Dict[str, Tensor],
        velocity: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
    ) -> Dict[str, Tensor]:
        """
        Apply absorbing boundary condition.

        Returns residual for each boundary (should be zero).
        """
        x = coords[..., 0]
        z = coords[..., 1]

        x_min, x_max = domain_bounds[0]
        z_min, z_max = domain_bounds[1]

        p_t = derivatives.get("u_t", derivatives.get("p_t"))
        p_x = derivatives.get("u_x", derivatives.get("p_x"))
        p_z = derivatives.get("u_z", derivatives.get("p_z"))

        residuals = {}

        # Left boundary (x = x_min): ∂p/∂t - v * ∂p/∂x = 0
        left_mask = x < x_min + 0.01 * (x_max - x_min)
        if left_mask.any():
            residuals["left"] = self.absorption_coefficient * (
                p_t[left_mask] - velocity[left_mask] * p_x[left_mask]
            )

        # Right boundary (x = x_max): ∂p/∂t + v * ∂p/∂x = 0
        right_mask = x > x_max - 0.01 * (x_max - x_min)
        if right_mask.any():
            residuals["right"] = self.absorption_coefficient * (
                p_t[right_mask] + velocity[right_mask] * p_x[right_mask]
            )

        # Top boundary (z = z_min): ∂p/∂t - v * ∂p/∂z = 0
        top_mask = z < z_min + 0.01 * (z_max - z_min)
        if top_mask.any():
            residuals["top"] = self.absorption_coefficient * (
                p_t[top_mask] - velocity[top_mask] * p_z[top_mask]
            )

        # Bottom boundary (z = z_max): ∂p/∂t + v * ∂p/∂z = 0
        bottom_mask = z > z_max - 0.01 * (z_max - z_min)
        if bottom_mask.any():
            residuals["bottom"] = self.absorption_coefficient * (
                p_t[bottom_mask] + velocity[bottom_mask] * p_z[bottom_mask]
            )

        return residuals

    def get_boundary_mask(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
        thickness: float = 0.01,
    ) -> Tensor:
        """Get mask for boundary points."""
        x = coords[..., 0]
        z = coords[..., 1]

        x_min, x_max = domain_bounds[0]
        z_min, z_max = domain_bounds[1]

        dx = thickness * (x_max - x_min)
        dz = thickness * (z_max - z_min)

        mask = (
            (x < x_min + dx) |
            (x > x_max - dx) |
            (z < z_min + dz) |
            (z > z_max - dz)
        )

        return mask


class FreeSurfaceBoundary(BoundaryCondition):
    """
    Free Surface Boundary Condition.

    At the Earth's surface (air-rock interface), we have:

    Acoustic: p = 0 (pressure is zero)
    Elastic: σ·n = 0 (normal traction is zero)

    This creates reflections with polarity reversal for P-waves.

    Args:
        surface_location: z-coordinate of the free surface
    """

    def __init__(self, surface_location: float = 0.0) -> None:
        self.surface_location = surface_location

    def apply(
        self,
        coords: Tensor,
        field: Tensor,
        derivatives: Dict[str, Tensor],
        domain_bounds: Tuple[Tuple[float, float], ...],
        field_type: str = "acoustic",
    ) -> Dict[str, Tensor]:
        """
        Apply free surface boundary condition.

        For acoustic: p = 0 at surface
        For elastic: σ_zz = 0, σ_xz = 0 at surface
        """
        z = coords[..., 1]
        z_min = domain_bounds[1][0]

        # Surface is at top (z_min)
        surface_mask = torch.abs(z - z_min) < 0.01 * (
            domain_bounds[1][1] - domain_bounds[1][0]
        )

        residuals = {}

        if field_type == "acoustic":
            # Pressure should be zero at free surface
            residuals["pressure"] = field[surface_mask] if surface_mask.any() else torch.tensor(0.0)

        elif field_type == "elastic":
            # Normal stress should be zero
            # σ_zz = λ(∇·u) + 2μ ∂u_z/∂z = 0
            # σ_xz = μ(∂u_x/∂z + ∂u_z/∂x) = 0
            if surface_mask.any():
                residuals["stress_zz"] = derivatives.get("sigma_zz", torch.zeros(1))[surface_mask]
                residuals["stress_xz"] = derivatives.get("sigma_xz", torch.zeros(1))[surface_mask]

        return residuals

    def get_boundary_mask(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
        thickness: float = 0.01,
    ) -> Tensor:
        """Get mask for free surface points."""
        z = coords[..., 1]
        z_min = domain_bounds[1][0]
        dz = thickness * (domain_bounds[1][1] - domain_bounds[1][0])

        return z < z_min + dz


class PMLBoundary(BoundaryCondition):
    """
    Perfectly Matched Layer (PML) Boundary Condition.

    PML is the gold standard for absorbing boundaries. It introduces
    a complex-valued damping that attenuates waves without reflection.

    The wave equation in PML becomes:

        ∂²p/∂t² + (d_x + d_z) ∂p/∂t + d_x*d_z*p = v² ∇²p

    where d_x, d_z are damping profiles that increase toward the boundary.

    For PINNs, we modify the loss function to include PML damping.

    Args:
        pml_thickness: Thickness of PML layer (normalized)
        max_damping: Maximum damping coefficient
        damping_profile: Profile type ('linear', 'quadratic', 'cubic')
    """

    def __init__(
        self,
        pml_thickness: float = 0.1,
        max_damping: float = 100.0,
        damping_profile: str = "quadratic",
    ) -> None:
        self.pml_thickness = pml_thickness
        self.max_damping = max_damping
        self.damping_profile = damping_profile

    def _compute_damping_profile(self, distance: Tensor, pml_width: float) -> Tensor:
        """
        Compute damping coefficient based on distance into PML.

        Args:
            distance: Distance into PML (0 at interface, pml_width at edge)
            pml_width: Width of PML region

        Returns:
            Damping coefficient
        """
        # Normalize distance to [0, 1]
        normalized = torch.clamp(distance / pml_width, 0.0, 1.0)

        if self.damping_profile == "linear":
            return self.max_damping * normalized
        elif self.damping_profile == "quadratic":
            return self.max_damping * normalized ** 2
        elif self.damping_profile == "cubic":
            return self.max_damping * normalized ** 3
        else:
            return self.max_damping * normalized ** 2

    def get_damping_coefficients(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute PML damping coefficients at each point.

        Args:
            coords: Coordinates
            domain_bounds: Domain extent

        Returns:
            Tuple of (d_x, d_z) damping coefficients
        """
        x = coords[..., 0]
        z = coords[..., 1]

        x_min, x_max = domain_bounds[0]
        z_min, z_max = domain_bounds[1]

        domain_width_x = x_max - x_min
        domain_width_z = z_max - z_min

        pml_width_x = self.pml_thickness * domain_width_x
        pml_width_z = self.pml_thickness * domain_width_z

        # X-direction damping
        d_x = torch.zeros_like(x)

        # Left PML
        left_dist = x_min + pml_width_x - x
        left_mask = left_dist > 0
        d_x[left_mask] = self._compute_damping_profile(left_dist[left_mask], pml_width_x)

        # Right PML
        right_dist = x - (x_max - pml_width_x)
        right_mask = right_dist > 0
        d_x[right_mask] = self._compute_damping_profile(right_dist[right_mask], pml_width_x)

        # Z-direction damping
        d_z = torch.zeros_like(z)

        # Top PML
        top_dist = z_min + pml_width_z - z
        top_mask = top_dist > 0
        d_z[top_mask] = self._compute_damping_profile(top_dist[top_mask], pml_width_z)

        # Bottom PML
        bottom_dist = z - (z_max - pml_width_z)
        bottom_mask = bottom_dist > 0
        d_z[bottom_mask] = self._compute_damping_profile(bottom_dist[bottom_mask], pml_width_z)

        return d_x, d_z

    def apply(
        self,
        coords: Tensor,
        field: Tensor,
        derivatives: Dict[str, Tensor],
        velocity: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
    ) -> Tensor:
        """
        Compute PML-modified wave equation residual.

        The modified equation in PML is:
        ∂²p/∂t² + (d_x + d_z) ∂p/∂t + d_x*d_z*p = v² ∇²p
        """
        d_x, d_z = self.get_damping_coefficients(coords, domain_bounds)

        p = field
        p_t = derivatives.get("u_t", derivatives.get("p_t"))
        p_tt = derivatives.get("u_tt", derivatives.get("p_tt"))
        laplacian = derivatives.get("laplacian")

        if p.dim() > 1:
            d_x = d_x.unsqueeze(-1)
            d_z = d_z.unsqueeze(-1)

        v_squared = velocity.pow(2)
        if v_squared.dim() < p_tt.dim():
            v_squared = v_squared.unsqueeze(-1)

        # PML-modified residual
        residual = (
            p_tt +
            (d_x + d_z) * p_t +
            d_x * d_z * p -
            v_squared * laplacian
        )

        return residual

    def get_boundary_mask(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
        thickness: Optional[float] = None,
    ) -> Tensor:
        """Get mask for PML region."""
        if thickness is None:
            thickness = self.pml_thickness

        x = coords[..., 0]
        z = coords[..., 1]

        x_min, x_max = domain_bounds[0]
        z_min, z_max = domain_bounds[1]

        dx = thickness * (x_max - x_min)
        dz = thickness * (z_max - z_min)

        mask = (
            (x < x_min + dx) |
            (x > x_max - dx) |
            (z < z_min + dz) |
            (z > z_max - dz)
        )

        return mask


class SpongeLayer(BoundaryCondition):
    """
    Sponge Layer (Relaxation) Boundary.

    A simpler alternative to PML that exponentially damps the solution
    toward a reference state (usually zero) in the boundary region:

        p_damped = p * exp(-α * d)

    where d is distance into the sponge layer.

    Args:
        thickness: Sponge layer thickness (normalized)
        damping_rate: Damping rate coefficient
    """

    def __init__(
        self,
        thickness: float = 0.1,
        damping_rate: float = 5.0,
    ) -> None:
        self.thickness = thickness
        self.damping_rate = damping_rate

    def get_damping_factor(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
    ) -> Tensor:
        """
        Compute damping factor (0 at interior, 1 at edge).

        Returns value to multiply field by: exp(-α * factor)
        """
        x = coords[..., 0]
        z = coords[..., 1]

        x_min, x_max = domain_bounds[0]
        z_min, z_max = domain_bounds[1]

        dx = self.thickness * (x_max - x_min)
        dz = self.thickness * (z_max - z_min)

        # Distance into sponge layer (normalized to [0, 1])
        factor_x = torch.zeros_like(x)
        factor_z = torch.zeros_like(z)

        # Left boundary
        left_mask = x < x_min + dx
        factor_x[left_mask] = (x_min + dx - x[left_mask]) / dx

        # Right boundary
        right_mask = x > x_max - dx
        factor_x[right_mask] = (x[right_mask] - (x_max - dx)) / dx

        # Top boundary
        top_mask = z < z_min + dz
        factor_z[top_mask] = (z_min + dz - z[top_mask]) / dz

        # Bottom boundary
        bottom_mask = z > z_max - dz
        factor_z[bottom_mask] = (z[bottom_mask] - (z_max - dz)) / dz

        # Combined factor (maximum of x and z)
        factor = torch.max(factor_x, factor_z)

        return torch.exp(-self.damping_rate * factor)

    def apply(
        self,
        coords: Tensor,
        field: Tensor,
        derivatives: Dict[str, Tensor],
        domain_bounds: Tuple[Tuple[float, float], ...],
    ) -> Tensor:
        """
        Apply sponge layer by returning target field.

        The loss will be: ||field - 0||² in sponge region
        """
        mask = self.get_boundary_mask(coords, domain_bounds)

        if mask.any():
            return field[mask]  # Should be zero in sponge
        return torch.tensor(0.0, device=field.device)

    def get_boundary_mask(
        self,
        coords: Tensor,
        domain_bounds: Tuple[Tuple[float, float], ...],
        thickness: Optional[float] = None,
    ) -> Tensor:
        """Get mask for sponge region."""
        if thickness is None:
            thickness = self.thickness

        x = coords[..., 0]
        z = coords[..., 1]

        x_min, x_max = domain_bounds[0]
        z_min, z_max = domain_bounds[1]

        dx = thickness * (x_max - x_min)
        dz = thickness * (z_max - z_min)

        mask = (
            (x < x_min + dx) |
            (x > x_max - dx) |
            (z < z_min + dz) |
            (z > z_max - dz)
        )

        return mask
