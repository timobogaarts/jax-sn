import jax.numpy as jnp
from abc import abstractmethod
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator, MomentStrategy, DiscreteStrategy, BatchedMoment, BatchedDiscrete
from jax_sn.operators.mass_operator import MassOperator, MassStrategy, VmapMass
from jax_sn.operators.scatter_operator import ScatterOperator
from jax_sn.operators.transport_operator import TransportOperatorMultipleAngles, TransportSpec, SolutionSetup, AngularSetup
import lineax as lx

import equinox as eqx

class SingleGroupOperatorBase(AbstractLinearOperator_sn):
    r'''
    Single energy group linear operator base class.

    This implements the total, single-group linear operator:

    .. math::
        \mathbf{A}_g = \mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1} \mathbf{H} \mathbf{D}\mathbf{S}_{gg}

    where :math:`\mathbf{T}` is the (multi-angle) transport operator (see :class:`TransportOperatorMultipleAngles`), :math:`\mathbf{H}` the batched (over angles) mass operator (see :class:`MassOperator`), :math:`\mathbf{M}` the batched (over element and basis function)
    moment operator (see :class:`MomentOperator`),  :math:`\mathbf{S}_{gg}` the inscattering operator (see :class:`ScatterOperator`).

    Data ownership
    ---------------------------
    This operator follows the ``operator = setup + spec + strategy`` decomposition. It owns a
    single canonical :attr:`setup` (a :class:`~jax_sn.operators.transport_operator.SolutionSetup`,
    bundling the operator-defining :class:`~jax_sn.solution_domain.SolutionDomainData` with the
    operator-agnostic sweep execution plan) and a small data-free
    :attr:`transport_spec` (:class:`~jax_sn.operators.transport_operator.TransportSpec`), and
    exposes :attr:`transport_operator`, :attr:`mass_operator` and :attr:`inscatter_operator` as
    derived ``@property`` views built from those. The shared spatial/cross-section data therefore
    lives in exactly **one** pytree location (:attr:`setup`), so a single :func:`equinox.tree_at`
    on it updates every sub-operator at once -- there is no hand-synced duplication to keep
    consistent. :attr:`moment_operator` and :attr:`discrete_operator` remain stored fields (they
    carry the angular harmonics).

    Direction handling
    ---------------------------
    Whether this operator applies the forward action or its transpose is carried by the
    boolean field :attr:`transpose_bool`, *not* by a separate subclass. This keeps the
    transpose axis orthogonal to any future implementation axis: a concrete implementation
    subclass only needs to provide :meth:`_forward_mv` and :meth:`_transpose_mv`, while this
    base class handles dispatch, transposition (a cheap flag flip via :func:`equinox.tree_at`,
    no reconstruction) and the in/out shaping. Since this operator is square
    (:math:`\mathcal{M} \otimes \mathcal{F} \rightarrow \mathcal{M} \otimes \mathcal{F}`), the
    in/out structure is identical in both directions.

    Mathematical description
    ---------------------------

    This operator maps:

    .. math::
        \mathbf{A}_g : \mathcal{M} \otimes \mathcal{F} \rightarrow \mathcal{M} \otimes \mathcal{F}


    Transposition
    ---------------------------

    The transpose follows from transposing the individual operators:

    .. math::
        \mathbf{A}_g^T = \mathbf{I} - \mathbf{S}_{gg}^T \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_g^{-1})^T \mathbf{M}^T

    where the individual transposes are defined in their respective classes.

    Solution
    ---------------------------
    Since this is a iteratively defined linear operator, the standard way to solve systems involving this operator is to use an iterative solver (e.g., GMRES, CG, etc.) with the mv() method defined here. This can
    be accomplished using e.g. lineax solvers. This allows for the propagation of the derivatives through the solver as well.

    '''

    setup              : SolutionSetup
    r'''Canonical shared data: operator-defining spatial + angular data + operator-agnostic sweep plan. The single home every sub-operator derives from.'''

    transport_spec     : TransportSpec
    r'''Data-free transport configuration (strategies + energy group); consumes :attr:`setup` via :meth:`TransportSpec.build`.'''

    moment_strategy    : MomentStrategy = eqx.field(default_factory=BatchedMoment)
    r'''Contraction strategy for the derived :attr:`moment_operator`.'''

    discrete_strategy  : DiscreteStrategy = eqx.field(default_factory=BatchedDiscrete)
    r'''Contraction strategy for the derived :attr:`discrete_operator`.'''

    mass_strategy      : MassStrategy = eqx.field(default_factory=VmapMass)
    r'''Contraction strategy for the derived :attr:`mass_operator`.'''

    transpose_bool     : bool = False
    r'''Whether this operator applies the transposed action instead of the forward action.'''

    @property
    def transport_operator(self) -> TransportOperatorMultipleAngles:
        r'''The multi-angle transport operator, derived from :attr:`setup` + :attr:`transport_spec`.'''
        return self.transport_spec.build(self.setup)

    @property
    def moment_operator(self) -> MomentOperator:
        r'''The moment operator, derived from :attr:`setup`'s angular harmonics.'''
        return MomentOperator(self.setup.angular_setup.wk_ylm, self.setup.elements_basis_shape, self.moment_strategy)

    @property
    def discrete_operator(self) -> DiscreteOperator:
        r'''The discrete (moment-to-discrete) operator, derived from :attr:`setup`'s angular harmonics.'''
        return DiscreteOperator(self.setup.angular_setup.ylm, self.setup.elements_basis_shape, self.discrete_strategy)

    @property
    def mass_operator(self) -> MassOperator:
        r'''The mass operator, derived from :attr:`setup` (shares its geometric data + FEM matrices).'''
        return MassOperator.from_transport_operator(self.transport_operator, self.mass_strategy)

    @property
    def inscatter_operator(self) -> ScatterOperator:
        r'''The within-group scatter operator, derived from :attr:`setup`'s cross sections and the energy group.'''
        eg = self.transport_spec.energy_group
        return ScatterOperator(self.setup.domain.cross_sections_indexed, self.moment_operator.out_structure(), eg, eg)

    def __check_init__(self):
        assert isinstance(self.setup, SolutionSetup), f"setup must be of type SolutionSetup but is {type(self.setup)}"
        assert isinstance(self.transport_spec, TransportSpec), f"transport_spec must be of type TransportSpec but is {type(self.transport_spec)}"
        assert isinstance(self.moment_strategy, MomentStrategy), f"moment_strategy must be of type MomentStrategy but is {type(self.moment_strategy)}"
        assert isinstance(self.discrete_strategy, DiscreteStrategy), f"discrete_strategy must be of type DiscreteStrategy but is {type(self.discrete_strategy)}"
        assert isinstance(self.mass_strategy, MassStrategy), f"mass_strategy must be of type MassStrategy but is {type(self.mass_strategy)}"
        assert self.mass_operator.out_structure() == self.transport_operator.out_structure(), f"mass_operator out_structure {self.mass_operator.out_structure()} must match transport_operator [note it is inverted] out_structure {self.transport_operator.out_structure()}"
        assert self.inscatter_operator.in_structure() == self.moment_operator.out_structure(), f"inscatter_operator in_structure {self.inscatter_operator.in_structure()} must match moment_operator out_structure {self.moment_operator.out_structure()}"
        assert self.transport_operator.in_structure() == self.moment_operator.in_structure(), f"transport_operator in_structure [note it is inverted] {self.transport_operator.in_structure()} must match moment_operator in_structure {self.moment_operator.in_structure()}"
        assert self.discrete_operator.in_structure() == self.inscatter_operator.out_structure(), f"discrete_operator in_structure {self.discrete_operator.in_structure()} must match inscatter_operator out_structure {self.inscatter_operator.out_structure()}"
        # TODO add more checks? shapes, etc.

    @abstractmethod
    def _forward_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        '''Apply the forward action. Implemented by concrete subclasses.'''
        ...

    @abstractmethod
    def _transpose_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        '''Apply the transposed action. Implemented by concrete subclasses.'''
        ...

    def mv(self, x):
        if self.transpose_bool:
            return self._transpose_mv(x)
        return self._forward_mv(x)

    def transpose(self) -> "SingleGroupOperatorBase":
        '''Returns the transposed operator by flipping :attr:`transpose_bool`. There is no setup or reconstruction, so this operation is essentially free.'''
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)

    def to_energy_group(self, energy_group) -> "SingleGroupOperatorBase":
        '''Returns this operator for a different energy group, preserving the implementation type and transpose direction.

        Only :attr:`transport_spec`'s energy group is swapped; :attr:`inscatter_operator` follows
        automatically since it is derived from it.'''
        return eqx.tree_at(lambda op: op.transport_spec.energy_group, self, energy_group)

    @classmethod
    def from_operators(cls, moment_operator : MomentOperator, transport_operator : TransportOperatorMultipleAngles,
                       mass_operator : MassOperator, discrete_operator : DiscreteOperator,
                       quadrature_set, transpose_bool : bool = False) -> "SingleGroupOperator":
        r'''
        Build a single-group operator from already-constructed sub-operators.

        The canonical :attr:`setup` bundles the transport operator's spatial data + sweep plan with the
        angular data (``quadrature_set`` + the harmonics read off the moment/discrete operators). The
        strategies are extracted from the supplied operators. The within-group scatter operator and the
        moment/discrete operators are *not* stored: they are derived from :attr:`setup`, which is exactly
        the duplication this design removes.
        '''
        setup = SolutionSetup(
            transport_operator.solution_domain_data,
            AngularSetup(quadrature_set, moment_operator.wk_yklm, discrete_operator.yklm),
            transport_operator.sweep_data,
        )
        return cls(
            setup,
            TransportSpec.from_transport_operator(transport_operator),
            moment_operator.strategy,
            discrete_operator.strategy,
            mass_operator.strategy,
            transpose_bool,
        )

    def in_structure(self):
        return self.moment_operator.out_structure()

    def out_structure(self):
        return self.moment_operator.out_structure()

    @property
    def energy_group(self):
        return self.transport_spec.energy_group


@lx.is_symmetric.register(SingleGroupOperatorBase)
def _(operator : SingleGroupOperatorBase):
    return False

@lx.linearise.register(SingleGroupOperatorBase)
def _(operator : SingleGroupOperatorBase):
    return operator


class SingleGroupOperator(SingleGroupOperatorBase):
    r'''
    Single energy group linear operator (see :class:`SingleGroupOperatorBase`).

    The transposed action does not pre-apply the transposes of its constituent operators, but
    applies them on the fly (the individual transposes are free flag flips / data reversals).
    '''

    def _forward_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _mv_single_group_operator(self.moment_operator,
                                        self.transport_operator,
                                        self.mass_operator,
                                        self.discrete_operator,
                                        self.inscatter_operator,
                                        x)

    def _transpose_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _mv_single_group_operator_transpose(self.moment_operator,
                                                   self.transport_operator,
                                                   self.mass_operator,
                                                   self.discrete_operator,
                                                   self.inscatter_operator,
                                                   x)


@eqx.filter_jit
def _mv_single_group_operator(moment_operator, transport_operator, mass_operator, discrete_operator, inscatter_operator, x):
    r'''
    Multiplies the single group operator with a vector x.

    Parameters
    ----------
    moment_operator : MomentOperator
        The moment operator :math:`\mathbf{M}`.
    transport_operator : TransportOperatorMultipleAngles
        The transport operator :math:`\mathbf{T}^{-1}`.
    mass_operator : MassOperator
        The mass operator :math:`\mathbf{H}`.
    discrete_operator : DiscreteOperator
        The discrete operator :math:`\mathbf{D}`.
    inscatter_operator : ScatterOperator
        The scattering operator :math:`\mathbf{S}`.
    x : jnp.ndarray
        The input vector.

    Returns
    -------
    jnp.ndarray
        The result of the multiplication.
    '''

    # Memory analysis:
    # mass_operator(x) [no temporary ]

    # mass_operator(inscatter(x)) [temporary size x due to not being able to re-use x for the output, this means the inscatter op needs to scale to a temporary, and we use this temporary as input for the mass operator (which we can't reuse as output for the mass operator)]


    # moment_op(disc_op(mass_op(inscatter(x)))) [temporary size discrete x, because we need that buffer as temporary anyway. the temporary of above is apparently re-used]

    # moment_operator.mv(mass_operator.mv(discrete_operator.mv(inscatter_operator.mv(x)))) [temporary size 2 x discrete x because we need it as input for the mass operator and as output for the mass operator ] (same for x - this)

    # x - moment_operator.mv(transport_operator.inv(mass_operator.mv(discrete_operator.mv(inscatter_operator.mv(x))))) [temporary size 2x discrete x, apparently re-used for transport operator (doesn't require much because batch size 1)]

    # switching out mass & discrete does nothing: then you need 2 temporaries for the input output of the transport operator, so you gain absolutely nothing.
    # This is the absolute best you can do (and it's the same actually as in the old stell-blanket.): the transport operator inverse is non local, so you can't merge or fuse the kernels any further.

    return x - moment_operator.mv(transport_operator.inv(discrete_operator.mv(mass_operator.mv(inscatter_operator.mv(x)))))




@eqx.filter_jit
def _mv_single_group_operator_transpose(moment_operator, transport_operator, mass_operator, discrete_operator, inscatter_operator, x):
    r'''
    Multiplies the transpose of the single group operator with a vector x.

    Since the transposes are trivial (we only point to a different function), this does not add any overhead.

    Parameters
    ----------
    moment_operator : MomentOperator
        The moment operator :math:`\mathbf{M}`.
    transport_operator : TransportOperatorMultipleAngles
        The transport operator :math:`\mathbf{T}^{-1}`.
    mass_operator : MassOperator
        The mass operator :math:`\mathbf{H}`.
    discrete_operator : DiscreteOperator
        The discrete operator :math:`\mathbf{D}`.
    inscatter_operator : ScatterOperator
        The scattering operator :math:`\mathbf{S}`.
    x : jnp.ndarray
        The input vector.

    Returns
    -------
    jnp.ndarray
        The result of the multiplication.
    '''
    return x -inscatter_operator.transpose().mv( discrete_operator.transpose().mv(mass_operator.transpose().mv(transport_operator.transpose().inv(moment_operator.transpose().mv(x)))))


# For the solve, we have as memory analysis:
# mv = 2 x discrete temporary + in out moment
# solve  with bicgstab:
#    in  = moment array + setup
#    out = moment_array
#    temp = 2 * discrete temporary + 5 * moment_array from bicstab + 2 * moment array that i cannot place (seems from xla reshuffle in the specific case i tested? only init carry return already gives this...)
