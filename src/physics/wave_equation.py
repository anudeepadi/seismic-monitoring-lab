"""
Wave Equation Implementations for Physics-Informed Learning

This module provides implementations of various wave equations used
as physics constraints in PINNs. These equations describe how seismic
waves propagate through the Earth's subsurface.

Supported equations:
- Acoustic wave equation (pressure waves in fluids/simple solids)
- Elastic wave equation (P and S waves in solids)
- Viscoacoustic wave equation (with attenuation)
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class WaveEquation(ABC):
    """
    Abstract base class for wave equations.

    Wave equations define the physical constraints that the neural network
    solution must satisfy. Each implementation computes the residual of
    the PDE, which should be zero for a valid solution.
    """

    @abstractmethod
    def compute_residual(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
        source_term: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute the PDE residual.

        Args:
            network: Neural network that predicts the wave field
            coords: Space-time coordinates (batch, dim)
            velocity: Velocity field at each point
            source_term: Optional source forcing term

        Returns:
            Dictionary containing residual tensors
        """
        pass

    @abstractmethod
    def compute_energy(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """
        Compute the wave energy (for conservation checks).

        Args:
            network: Neural network
            coords: Coordinates
            velocity: Velocity field

        Returns:
            Total energy at each point
        """
        pass


class AcousticWaveEquation(WaveEquation):
    """
    2D Acoustic Wave Equation.

    The acoustic wave equation describes pressure wave propagation:

        ∂²p/∂t² = v²(x,z) * (∂²p/∂x² + ∂²p/∂z²) + f(x,z,t)

    where:
    - p(x, z, t) is the pressure field
    - v(x, z) is the acoustic velocity
    - f(x, z, t) is the source term

    This is the fundamental equation for seismic exploration, where
    we model P-wave propagation through rock formations.

    Args:
        use_density: Include density variations (variable density acoustic)
        dimension: Spatial dimension (2 or 3)
    """

    def __init__(
        self,
        use_density: bool = False,
        dimension: int = 2,
    ) -> None:
        self.use_density = use_density
        self.dimension = dimension

    def compute_residual(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
        source_term: Optional[Tensor] = None,
        density: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute the acoustic wave equation residual.

        The residual R = ∂²p/∂t² - v² * ∇²p - f should be zero.
        """
        coords = coords.requires_grad_(True)

        # Get pressure field
        p = network(coords)
        if p.dim() > 1 and p.shape[-1] > 1:
            p = p[..., 0:1]

        # First derivatives
        grad_p = torch.autograd.grad(
            outputs=p.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Spatial second derivatives (Laplacian components)
        laplacian = torch.zeros_like(p)
        for i in range(self.dimension):
            grad_component = grad_p[..., i]
            grad2 = torch.autograd.grad(
                outputs=grad_component.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
            )[0]
            laplacian = laplacian + grad2[..., i : i + 1]

        # Second time derivative
        time_idx = self.dimension
        grad_t = grad_p[..., time_idx]
        grad2_t = torch.autograd.grad(
            outputs=grad_t.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        p_tt = grad2_t[..., time_idx : time_idx + 1]

        # Wave equation residual
        v_squared = velocity.pow(2)
        if v_squared.dim() < p_tt.dim():
            v_squared = v_squared.unsqueeze(-1)

        residual = p_tt - v_squared * laplacian

        if source_term is not None:
            residual = residual - source_term

        # Variable density formulation (optional)
        if self.use_density and density is not None:
            # ∇·(1/ρ ∇p) - 1/(ρv²) ∂²p/∂t² = f
            # This is more complex and requires density gradients
            pass

        return {
            "pde_residual": residual,
            "pressure": p,
            "laplacian": laplacian,
            "p_tt": p_tt,
        }

    def compute_energy(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """
        Compute acoustic wave energy.

        Total energy E = (1/2) * ∫ [p_t²/v² + |∇p|²] dV
        """
        coords = coords.requires_grad_(True)
        p = network(coords)

        grad_p = torch.autograd.grad(
            outputs=p.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Kinetic energy: p_t² / (2v²)
        p_t = grad_p[..., -1:]
        kinetic = 0.5 * p_t.pow(2) / velocity.pow(2)

        # Potential energy: |∇p|² / 2
        spatial_grad = grad_p[..., : self.dimension]
        potential = 0.5 * spatial_grad.pow(2).sum(dim=-1, keepdim=True)

        return kinetic + potential


class ElasticWaveEquation(WaveEquation):
    """
    2D Elastic Wave Equation.

    The elastic wave equation describes coupled P-wave and S-wave propagation:

        ρ * ∂²u/∂t² = ∇·σ + f

    where σ is the stress tensor related to strain through Hooke's law:

        σ = λ(∇·u)I + μ(∇u + ∇uᵀ)

    In 2D isotropic media, this gives us two coupled equations for
    horizontal (ux) and vertical (uz) displacement.

    The P-wave velocity vp = √((λ + 2μ)/ρ)
    The S-wave velocity vs = √(μ/ρ)

    Args:
        dimension: Spatial dimension (2 or 3)
    """

    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension

    def compute_residual(
        self,
        network: nn.Module,
        coords: Tensor,
        vp: Tensor,
        vs: Tensor,
        density: Tensor,
        source_term: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute elastic wave equation residuals.

        Returns residuals for both momentum equations.
        """
        coords = coords.requires_grad_(True)

        # Get displacement field (ux, uz)
        u = network(coords)  # (batch, 2)

        # Compute Lamé parameters from velocities
        mu = density * vs.pow(2)  # Shear modulus
        lam = density * vp.pow(2) - 2 * mu  # First Lamé parameter

        # First derivatives of displacement
        grad_ux = torch.autograd.grad(
            outputs=u[:, 0].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_uz = torch.autograd.grad(
            outputs=u[:, 1].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Strain components
        ux_x = grad_ux[:, 0:1]  # ∂ux/∂x
        ux_z = grad_ux[:, 1:2]  # ∂ux/∂z
        uz_x = grad_uz[:, 0:1]  # ∂uz/∂x
        uz_z = grad_uz[:, 1:2]  # ∂uz/∂z

        # Divergence
        div_u = ux_x + uz_z

        # Second derivatives for Laplacian
        grad2_ux_x = torch.autograd.grad(
            outputs=ux_x.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad2_ux_z = torch.autograd.grad(
            outputs=ux_z.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad2_uz_x = torch.autograd.grad(
            outputs=uz_x.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad2_uz_z = torch.autograd.grad(
            outputs=uz_z.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Laplacian of u
        laplacian_ux = grad2_ux_x[:, 0:1] + grad2_ux_z[:, 1:2]
        laplacian_uz = grad2_uz_x[:, 0:1] + grad2_uz_z[:, 1:2]

        # Gradient of divergence
        div_grad_x = grad2_ux_x[:, 0:1] + grad2_uz_x[:, 1:2]
        div_grad_z = grad2_ux_z[:, 0:1] + grad2_uz_z[:, 1:2]

        # Time derivatives
        ux_tt = torch.autograd.grad(
            outputs=grad_ux[:, 2].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        uz_tt = torch.autograd.grad(
            outputs=grad_uz[:, 2].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0][:, 2:3]

        # Momentum equations
        # ρ * ∂²ux/∂t² = (λ + μ) * ∂(∇·u)/∂x + μ * ∇²ux
        # ρ * ∂²uz/∂t² = (λ + μ) * ∂(∇·u)/∂z + μ * ∇²uz

        lam_plus_mu = lam + mu

        residual_x = density * ux_tt - lam_plus_mu * div_grad_x - mu * laplacian_ux
        residual_z = density * uz_tt - lam_plus_mu * div_grad_z - mu * laplacian_uz

        if source_term is not None:
            residual_x = residual_x - source_term[:, 0:1]
            residual_z = residual_z - source_term[:, 1:2]

        return {
            "residual_x": residual_x,
            "residual_z": residual_z,
            "displacement": u,
            "strain_xx": ux_x,
            "strain_zz": uz_z,
            "strain_xz": 0.5 * (ux_z + uz_x),
        }

    def compute_energy(
        self,
        network: nn.Module,
        coords: Tensor,
        vp: Tensor,
        vs: Tensor,
        density: Tensor,
    ) -> Tensor:
        """
        Compute elastic wave energy (kinetic + strain).
        """
        coords = coords.requires_grad_(True)
        u = network(coords)

        # Velocity (time derivatives)
        grad_ux = torch.autograd.grad(
            u[:, 0].sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_uz = torch.autograd.grad(
            u[:, 1].sum(), coords, create_graph=True, retain_graph=True
        )[0]

        ux_t = grad_ux[:, 2:3]
        uz_t = grad_uz[:, 2:3]

        # Kinetic energy: (1/2) ρ (ux_t² + uz_t²)
        kinetic = 0.5 * density * (ux_t.pow(2) + uz_t.pow(2))

        # Lamé parameters
        mu = density * vs.pow(2)
        lam = density * vp.pow(2) - 2 * mu

        # Strain energy (simplified)
        ux_x = grad_ux[:, 0:1]
        uz_z = grad_uz[:, 1:2]
        ux_z = grad_ux[:, 1:2]
        uz_x = grad_uz[:, 0:1]

        div_u = ux_x + uz_z
        strain_xz = 0.5 * (ux_z + uz_x)

        strain_energy = 0.5 * lam * div_u.pow(2) + mu * (
            ux_x.pow(2) + uz_z.pow(2) + 2 * strain_xz.pow(2)
        )

        return kinetic + strain_energy


class ViscoacousticWaveEquation(WaveEquation):
    """
    Viscoacoustic Wave Equation with Attenuation.

    In real earth materials, seismic waves lose energy due to
    intrinsic attenuation. This is modeled using the quality factor Q:

        ∂²p/∂t² + (ω₀/Q) ∂p/∂t = v² ∇²p + f

    where ω₀ is a reference frequency and Q is the quality factor
    (higher Q = less attenuation).

    For frequency-independent Q (constant Q model), we use a
    memory variable formulation.

    Args:
        reference_frequency: Reference frequency for Q model (Hz)
        num_relaxation_mechanisms: Number of relaxation mechanisms for constant Q
        dimension: Spatial dimension
    """

    def __init__(
        self,
        reference_frequency: float = 20.0,
        num_relaxation_mechanisms: int = 3,
        dimension: int = 2,
    ) -> None:
        self.reference_frequency = reference_frequency
        self.omega_0 = 2.0 * torch.pi * reference_frequency
        self.num_mechanisms = num_relaxation_mechanisms
        self.dimension = dimension

        # Relaxation times for constant Q approximation (Cole-Cole model)
        self.tau_sigma, self.tau_epsilon = self._compute_relaxation_times()

    def _compute_relaxation_times(self) -> Tuple[Tensor, Tensor]:
        """
        Compute relaxation times for constant Q approximation.

        Uses logarithmically spaced relaxation mechanisms to achieve
        approximately constant Q over a broad frequency band.
        """
        # Frequency range for constant Q
        f_min = self.reference_frequency / 10
        f_max = self.reference_frequency * 10

        # Relaxation times
        tau_sigma = torch.logspace(
            -torch.log10(torch.tensor(2 * torch.pi * f_max)),
            -torch.log10(torch.tensor(2 * torch.pi * f_min)),
            self.num_mechanisms,
        )

        tau_epsilon = tau_sigma  # For constant Q, tau_epsilon ≈ tau_sigma

        return tau_sigma, tau_epsilon

    def compute_residual(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
        Q: Tensor,
        source_term: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute viscoacoustic wave equation residual.

        Uses a simple viscous damping term for attenuation.
        """
        coords = coords.requires_grad_(True)

        # Get pressure field
        p = network(coords)
        if p.dim() > 1 and p.shape[-1] > 1:
            p = p[..., 0:1]

        # First derivatives
        grad_p = torch.autograd.grad(
            outputs=p.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Spatial Laplacian
        laplacian = torch.zeros_like(p)
        for i in range(self.dimension):
            grad_component = grad_p[..., i]
            grad2 = torch.autograd.grad(
                outputs=grad_component.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
            )[0]
            laplacian = laplacian + grad2[..., i : i + 1]

        # Time derivatives
        time_idx = self.dimension
        p_t = grad_p[..., time_idx : time_idx + 1]

        grad2_t = torch.autograd.grad(
            outputs=grad_p[..., time_idx].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        p_tt = grad2_t[..., time_idx : time_idx + 1]

        # Viscoacoustic residual with damping
        # ∂²p/∂t² + (ω₀/Q) * ∂p/∂t = v² * ∇²p + f
        v_squared = velocity.pow(2)
        if v_squared.dim() < p_tt.dim():
            v_squared = v_squared.unsqueeze(-1)

        damping_coeff = self.omega_0 / Q
        if damping_coeff.dim() < p_t.dim():
            damping_coeff = damping_coeff.unsqueeze(-1)

        residual = p_tt + damping_coeff * p_t - v_squared * laplacian

        if source_term is not None:
            residual = residual - source_term

        return {
            "pde_residual": residual,
            "pressure": p,
            "p_t": p_t,
            "p_tt": p_tt,
            "laplacian": laplacian,
            "damping_term": damping_coeff * p_t,
        }

    def compute_energy(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
        Q: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute viscoacoustic wave energy.

        Note: With attenuation, energy is not conserved but decays over time.
        """
        coords = coords.requires_grad_(True)
        p = network(coords)

        grad_p = torch.autograd.grad(
            outputs=p.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Kinetic energy
        p_t = grad_p[..., -1:]
        kinetic = 0.5 * p_t.pow(2) / velocity.pow(2)

        # Potential energy
        spatial_grad = grad_p[..., : self.dimension]
        potential = 0.5 * spatial_grad.pow(2).sum(dim=-1, keepdim=True)

        return kinetic + potential


class EikonalEquation:
    """
    Eikonal Equation for Travel Time Computation.

    The eikonal equation describes the propagation of wavefronts:

        |∇T|² = 1/v²

    where T(x, z) is the travel time from a source point and v is velocity.

    This is useful for:
    - First-arrival travel time modeling
    - Initializing full waveform inversion
    - Ray tracing applications

    Args:
        dimension: Spatial dimension
    """

    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension

    def compute_residual(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute eikonal equation residual.

        Residual: |∇T|² - 1/v² should be zero.
        """
        coords = coords.requires_grad_(True)

        # Get travel time
        T = network(coords)
        if T.dim() > 1 and T.shape[-1] > 1:
            T = T[..., 0:1]

        # Gradient of travel time
        grad_T = torch.autograd.grad(
            outputs=T.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Only spatial dimensions
        spatial_grad = grad_T[..., : self.dimension]

        # Eikonal residual
        grad_magnitude_sq = spatial_grad.pow(2).sum(dim=-1, keepdim=True)
        slowness_sq = 1.0 / velocity.pow(2)

        if slowness_sq.dim() < grad_magnitude_sq.dim():
            slowness_sq = slowness_sq.unsqueeze(-1)

        residual = grad_magnitude_sq - slowness_sq

        return {
            "eikonal_residual": residual,
            "travel_time": T,
            "gradient": spatial_grad,
            "gradient_magnitude": torch.sqrt(grad_magnitude_sq),
        }

    def factored_eikonal_residual(
        self,
        network: nn.Module,
        coords: Tensor,
        velocity: Tensor,
        source_coords: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute factored eikonal residual for better source handling.

        Uses the factored form: T = T₀ * τ
        where T₀ is the travel time in a reference medium and τ is a
        correction factor. This helps with the singularity at the source.
        """
        coords = coords.requires_grad_(True)

        # Reference travel time (homogeneous medium)
        ref_velocity = velocity.mean()
        distance = torch.sqrt(
            ((coords[..., : self.dimension] - source_coords[..., : self.dimension]) ** 2).sum(
                dim=-1, keepdim=True
            )
        )
        T0 = distance / ref_velocity

        # Network predicts correction factor τ
        tau = network(coords)
        if tau.dim() > 1 and tau.shape[-1] > 1:
            tau = tau[..., 0:1]

        # Ensure τ > 0 for stability
        tau = torch.nn.functional.softplus(tau)

        # Total travel time
        T = T0 * tau

        # Gradient computation
        grad_T = torch.autograd.grad(
            outputs=T.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        spatial_grad = grad_T[..., : self.dimension]
        grad_magnitude_sq = spatial_grad.pow(2).sum(dim=-1, keepdim=True)
        slowness_sq = 1.0 / velocity.pow(2)

        if slowness_sq.dim() < grad_magnitude_sq.dim():
            slowness_sq = slowness_sq.unsqueeze(-1)

        residual = grad_magnitude_sq - slowness_sq

        return {
            "eikonal_residual": residual,
            "travel_time": T,
            "correction_factor": tau,
            "reference_time": T0,
        }
