import functools
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np


class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class SmallImageCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        if len(x.shape) == 4:
            num_dims = 3
        else:
            num_dims = len(x.shape) - 2  # b x num_envs
        # 10x10 2 dimensions
        if num_dims == 2 and x.shape[-2] == x.shape[-1] and x.shape[-2] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=5, strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=4, strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=3, strides=1, padding=0)(out2)

        # 5x5
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 5:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=1)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        # 3x3
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 3:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out1)

        # 10x10
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(5, 5), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out2)

        elif x.shape[-2] == 7 and x.shape[-3] == 4:
            out1 = nn.Conv(features=64, kernel_size=(2, 4), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=128, kernel_size=(2, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)
        elif x.shape[-2] == 5 and x.shape[-3] == 3:
            out1 = nn.Conv(features=64, kernel_size=(2, 3), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=128, kernel_size=(2, 2), strides=1, padding=0)(out1)
            # out2 = nn.relu(out2)
            # conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        elif x.shape[-2] == 3 and x.shape[-3] == 2:
            out1 = nn.Conv(features=64, kernel_size=(1, 1), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=128, kernel_size=(2, 2), strides=1, padding=0)(out1)

        elif x.shape[-2] >= 14:
            out1 = nn.Conv(features=64, kernel_size=(6, 6), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=64, kernel_size=(5, 5), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)

            final_out = out2
            # if x.shape[-2] >= 20:
            #     out3 = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding=0)(out2)
            #     out3 = nn.relu(out3)
            #     final_out = out3
            conv_out = nn.Conv(features=64, kernel_size=(2, 2), strides=1, padding=0)(final_out)

        else:
            raise NotImplementedError

        conv_out = nn.relu(conv_out)
        # Convolutions "flatten" the last num_dims dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-num_dims], -1))  # Flatten
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out


class SimpleNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out


class FullImageCNN(nn.Module):
    hidden_size: int
    num_channels: int = 32

    @nn.compact
    def __call__(self, x):
        if len(x.shape) == 4:
            num_dims = 3
        else:
            num_dims = len(x.shape) - 2  # b x num_envs
        out1 = nn.Conv(features=self.num_channels, kernel_size=(7, 7), strides=4)(x)
        out1 = nn.relu(out1)
        out2 = nn.Conv(features=self.num_channels, kernel_size=(5, 5), strides=2)(out1)
        out2 = nn.relu(out2)
        out3 = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=2)(out2)
        out3 = nn.relu(out3)
        out4 = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=2)(out3)
        flat_out = out4.reshape((*out4.shape[:-num_dims], -1))  # Flatten
        flat_out = nn.relu(flat_out)

        dense_out = nn.Dense(features=self.hidden_size)(flat_out)
        dense_out = nn.relu(dense_out)

        final_out = nn.Dense(features=self.hidden_size)(dense_out)
        return final_out


# ---------------------------------------------------------------------------
# uRNN cell: W = D3 R2 F^-1 D2 Pi R1 F D1 (Arjovsky et al., 2016) over a
# complex64 hidden state, ModReLU activation, optional dense input feed.
# ---------------------------------------------------------------------------


def initial_urnn_carry(batch_size: int, hidden_size: int,
                       norm_scale: float = 1.0) -> jnp.ndarray:
    """Equal-superposition complex hidden state.

    Each entry is sqrt(norm_scale / (2H)) * (1 + 1j).
    """
    v = jnp.sqrt(jnp.asarray(norm_scale / (2.0 * hidden_size),
                             dtype=jnp.float32))
    scalar = (v + 1j * v).astype(jnp.complex64)
    return jnp.broadcast_to(scalar, (batch_size, hidden_size))


def complex_unit_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize a complex vector to unit L2 norm along the last axis.

    Put ``eps`` **inside** the sqrt so the VJP is finite at ``x=0``. The
    naive ``x / (jnp.linalg.norm(x) + eps)`` form has NaN gradient at the
    origin because the norm's derivative is ``x* / |x|`` which is
    ``0/0`` there — no ``eps`` in the denominator can fix that. Adding
    ``eps`` under the sqrt smooths the zero. This matters for any
    sparse-observation env where ``rot_embed(0·obs) = 0`` at init
    (e.g. battleship at episode reset, where the action-concat prev
    action is all-zeros).
    """
    norm2 = jnp.sum((x.conj() * x).real, axis=-1, keepdims=True)
    return x / jnp.sqrt(norm2 + eps)


def householder_matrix(v: jnp.ndarray) -> jnp.ndarray:
    """Batched Householder reflection I - 2 v v^H.

    v: (..., H) complex, already unit-norm.
    Returns: (..., H, H) complex unitary.

    No longer called from the uRNN cells (they use
    ``_apply_factored_householder`` below). Retained as the materialized
    reference in ``tests/test_householder_factored.py``.
    """
    v_col = v[..., :, None]                   # (..., H, 1)
    v_dag = jnp.conj(v)[..., None, :]         # (..., 1, H)
    H = v.shape[-1]
    eye = jnp.eye(H, dtype=v.dtype)
    return eye - 2.0 * (v_col @ v_dag)


def _apply_factored_householder(v: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Apply (I - 2 v v^H) h without materializing the (H, H) matrix.

    By associativity, (I - 2 v v^H) h = h - 2 v (v^H h). The factored form
    costs O(H) FLOPs and O(H) activation tape per call, vs O(H^2) for the
    materialized path. Forward + backward both drop by the same factor —
    see ``tests/test_householder_factored.py`` for equivalence and gradient
    checks.

    v must be unit-norm along the last axis (use ``complex_unit_norm`` on
    the raw rotation vector first). Shapes broadcast normally, so a shared
    ``(H,)`` v applied to ``(batch, H)`` h works via ``v[None, :]``.
    """
    vh_h = jnp.sum(jnp.conj(v) * h, axis=-1, keepdims=True)
    return h - 2.0 * v * vh_h


class ModReLU(nn.Module):
    """Complex activation: z * relu(|z| + beta) / (|z| + eps).

    Algebraically equivalent to max(0, |z|+beta) * exp(i * angle(z)) but
    avoids jnp.angle's singularity at z=0 under autodiff.
    """

    hidden_size: int
    eps: float = 1e-8

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        beta = self.param('beta', nn.initializers.zeros, (self.hidden_size,),
                          jnp.float32)
        mag = jnp.abs(z)
        scale = nn.relu(mag + beta) / (mag + self.eps)
        return z * scale.astype(z.dtype)


def _glorot_complex():
    """Glorot-uniform init over complex64, per-element variance matches
    torch.nn.init.xavier_uniform_ applied to a complex tensor
    (= 4 / (fan_in + fan_out))."""
    return nn.initializers.glorot_uniform(dtype=jnp.complex64)


class URNNCell(nn.Module):
    """Single step of the input-dependent uRNN (ICuRNN: U = U(x_t)).

    D1/D2/D3 and R1/R2 are produced by Dense projections of the real input.
    ``diag_embed`` stays real-dtype (its output is multiplied by 1j before
    exponentiating). ``rot_embed`` and the optional ``input_embed`` are
    complex-weighted (``dtype=complex64``).
    """

    hidden_size: int
    add_input_dense: bool = True

    @nn.compact
    def __call__(self, h: jnp.ndarray, ins: jnp.ndarray,
                 perm: jnp.ndarray) -> jnp.ndarray:
        ins_c = ins.astype(jnp.complex64)

        diag_params = nn.Dense(features=3 * self.hidden_size,
                               param_dtype=jnp.float32,
                               name='diag_embed')(ins)
        d = jnp.exp(1j * diag_params.astype(jnp.complex64))
        d1, d2, d3 = jnp.split(d, 3, axis=-1)

        rot_params = nn.Dense(features=2 * self.hidden_size,
                              param_dtype=jnp.complex64,
                              kernel_init=_glorot_complex(),
                              name='rot_embed')(ins_c)
        r1, r2 = jnp.split(rot_params, 2, axis=-1)
        v1 = complex_unit_norm(r1)
        v2 = complex_unit_norm(r2)

        # W = D3 R2 F^-1 D2 Pi R1 F D1
        h = jnp.fft.fft(d1 * h, axis=-1)
        # h = jnp.einsum('bij,bj->bi', R1, h)
        h = _apply_factored_householder(v1, h)
        h = jnp.fft.ifft(d2 * h[:, perm], axis=-1)
        # h = d3 * jnp.einsum('bij,bj->bi', R2, h)
        h = d3 * _apply_factored_householder(v2, h)

        if self.add_input_dense:
            h = h + nn.Dense(features=self.hidden_size, use_bias=False,
                             param_dtype=jnp.complex64,
                             kernel_init=_glorot_complex(),
                             name='input_embed')(ins_c)

        return ModReLU(hidden_size=self.hidden_size, name='activation')(h)


class LegacyURNNCell(nn.Module):
    """Single step of the fixed-learnable uRNN (constant U, not input-conditioned).

    D1/D2/D3 and the rotation vectors are module parameters rather than
    input-dependent projections. ``input_embed`` is always on.
    """

    hidden_size: int

    def _init_diag(self, key, shape, dtype):
        angles = jax.random.uniform(key, shape, jnp.float32,
                                    -jnp.pi, jnp.pi)
        return jnp.exp(1j * angles).astype(dtype)

    def _init_rotation(self, key, shape, dtype):
        k1, k2 = jax.random.split(key)
        re = jax.random.uniform(k1, shape, jnp.float32, -1.0, 1.0)
        im = jax.random.uniform(k2, shape, jnp.float32, -1.0, 1.0)
        return (re + 1j * im).astype(dtype)

    @nn.compact
    def __call__(self, h: jnp.ndarray, ins: jnp.ndarray,
                 perm: jnp.ndarray) -> jnp.ndarray:
        diag = self.param('diag', self._init_diag,
                          (3 * self.hidden_size,), jnp.complex64)
        rotation = self.param('rotation', self._init_rotation,
                              (2 * self.hidden_size,), jnp.complex64)

        d1, d2, d3 = jnp.split(diag, 3, axis=-1)
        r1, r2 = jnp.split(rotation, 2, axis=-1)
        v1 = complex_unit_norm(r1)
        v2 = complex_unit_norm(r2)

        h = jnp.fft.fft(d1[None, :] * h, axis=-1)
        # h = h @ R1
        h = _apply_factored_householder(v1[None, :], h)
        h = jnp.fft.ifft(d2[None, :] * h[:, perm], axis=-1)
        # h = d3[None, :] * (h @ R2)
        h = d3[None, :] * _apply_factored_householder(v2[None, :], h)

        h = h + nn.Dense(features=self.hidden_size, use_bias=False,
                         param_dtype=jnp.complex64,
                         kernel_init=_glorot_complex(),
                         name='input_embed')(ins.astype(jnp.complex64))

        return ModReLU(hidden_size=self.hidden_size, name='activation')(h)


class EUNNCell(nn.Module):
    """Single step of the tunable EUNN (Jing et al. 2017, arXiv:1612.05231).

    W = D * F_L * ... * F_1, where each F_k is a block-diagonal matrix of
    H/2 independent 2x2 unitary mixing units. L=H gives full U(H)
    coverage; smaller L gives a progressively restricted subspace. Layer
    composition uses jax.lax.scan (O(1) trace size in L).

    Parameters (all real-valued angles; complex action on the complex carry):
      - angles: (H, L) real. Even rows along axis 0 are phi (per-pair
        phase), odd rows are theta (per-pair rotation angle).
      - diag:   (H,) real. Final diagonal phase D applied as
        h * exp(1j * D) — same parameterization as d1/d2/d3 in
        LegacyURNNCell.
      - input_embed: complex Dense (matches LegacyURNNCell convention).
    """

    hidden_size: int
    capacity: int

    def _init_angles(self, key, shape, dtype):
        return jax.random.uniform(key, shape, jnp.float32, 0.0,
                                  2 * jnp.pi).astype(dtype)

    @staticmethod
    def _apply_unitary(angles: jnp.ndarray, diag: jnp.ndarray,
                       h: jnp.ndarray) -> jnp.ndarray:
        """Apply W = D * F_L * ... * F_1 to h.

        angles: (H, L) real; diag: (H,) real; h: (batch, H) complex.
        Returns (batch, H) complex. Exposed as staticmethod so unit tests
        can materialize U by feeding in the identity without module init.
        """
        H, L = angles.shape
        angles_LH = angles.T                                           # (L, H)
        roll_shifts = (2 * (jnp.arange(L) % 2) - 1).astype(jnp.int32)  # (L,) ±1

        def body(h_carry, layer_input):
            ang_L, shift = layer_input
            phi = ang_L[0::2]                                          # (H/2,)
            theta = ang_L[1::2]                                        # (H/2,)

            h_even = h_carry[:, 0::2]                                  # (batch, H/2)
            h_odd = h_carry[:, 1::2]
            e_iphi = jnp.exp(1j * phi).astype(h_carry.dtype)
            cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)

            new_even = e_iphi * (cos_t * h_even - sin_t * h_odd)
            new_odd = sin_t * h_even + cos_t * h_odd

            # Re-interleave [even[0], odd[0], even[1], odd[1], ...]
            h_new = jnp.stack([new_even, new_odd], axis=-1).reshape(-1, H)
            # Alternate pair structure between layers via cyclic ±1 roll.
            h_new = jnp.roll(h_new, shift=shift, axis=-1)
            return h_new, None

        h, _ = jax.lax.scan(body, h, (angles_LH, roll_shifts))
        return h * jnp.exp(1j * diag).astype(h.dtype)

    @nn.compact
    def __call__(self, h: jnp.ndarray, ins: jnp.ndarray) -> jnp.ndarray:
        H, L = self.hidden_size, self.capacity
        if H % 2 != 0:
            raise ValueError(f'EUNN requires even hidden_size, got {H}')

        angles = self.param('angles', self._init_angles, (H, L), jnp.float32)
        diag = self.param('diag', self._init_angles, (H,), jnp.float32)

        h = self._apply_unitary(angles, diag, h)

        h = h + nn.Dense(features=H, use_bias=False,
                         param_dtype=jnp.complex64,
                         kernel_init=_glorot_complex(),
                         name='input_embed')(ins.astype(jnp.complex64))

        return ModReLU(hidden_size=H, name='activation')(h)


class ScannedURNN(nn.Module):
    """Time-scanned uRNN with episode-boundary resets.

    Carry is complex64 ``(batch, hidden_size)``. The scan output y is the
    carry itself (complex); the real-concat adapter for downstream heads
    lives in ``ActorCritic.__call__``.

    The fixed permutation is stored in the ``constants`` variable
    collection so it rides along with the checkpoint and is broadcast
    across the time axis by ``nn.scan``.
    """

    hidden_size: int
    variant: Literal['standard', 'legacy', 'eunn'] = 'standard'
    add_input_dense: bool = True
    norm_scale: float = 1.0
    perm_seed: int = 0
    capacity: int = 2                     # [EUNN] L; ignored by other variants.

    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        # perm is derived deterministically from two static attrs; XLA
        # constant-folds the jnp.asarray on the first trace.
        perm = jnp.asarray(
            np.random.default_rng(self.perm_seed).permutation(self.hidden_size),
            dtype=jnp.int32,
        )

        initial_h = initial_urnn_carry(ins.shape[0], self.hidden_size,
                                       self.norm_scale)
        rnn_state = jnp.where(resets[:, None], initial_h, carry)

        if self.variant == 'legacy':
            new_h = LegacyURNNCell(hidden_size=self.hidden_size,
                                   name='cell')(rnn_state, ins, perm)
        elif self.variant == 'eunn':
            new_h = EUNNCell(hidden_size=self.hidden_size,
                             capacity=self.capacity,
                             name='cell')(rnn_state, ins)
        else:
            new_h = URNNCell(hidden_size=self.hidden_size,
                             add_input_dense=self.add_input_dense,
                             name='cell')(rnn_state, ins, perm)
        return new_h, new_h

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int,
                         norm_scale: float = 1.0) -> jnp.ndarray:
        return initial_urnn_carry(batch_size, hidden_size, norm_scale)


def get_memory_initial_carry(memory_type: str, batch_size: int,
                             hidden_size: int,
                             norm_scale: float = 1.0) -> jnp.ndarray:
    """Dispatch initial carry so ppo.py doesn't branch at the call site."""
    if memory_type in ('urnn', 'eunn'):
        return initial_urnn_carry(batch_size, hidden_size, norm_scale)
    return ScannedRNN.initialize_carry(batch_size, hidden_size)