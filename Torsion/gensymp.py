# gensymp.py
#
# Core classes extracted from the notebook:
#   Ysystem/4_1 torsion PGSp4 non-geom.ipynb
#
# These classes are intended to be used inside a SageMath environment.

from sage.all import (
    Matrix,
    zero_matrix,
    identity_matrix,
    block_matrix,
    matrix,
    diagonal_matrix,
    elementary_matrix,
)


class GenSympGroup:
    r"""
    K-valued points of GSp_{2n}:
        GSp_{2n}(K) = { (g, l) in GL_{2n}(K) x K^* | g^T J g = l J }.

    The projective relation for PGSp_{2n}(K) = GSp_{2n}(K) / G_m(K) is:
        (g,l) ~ (t g, t^2 l)  for t in K^*.
    """

    # --------------------------
    # Constructors / matrices
    # --------------------------

    @staticmethod
    def _alternating_antidiagonal_matrix(K, n):
        r"""
        I_n as in the notebook:
          (I_n)_{i, n+1-i} = (-1)^{i-1}, i=1..n
        i.e. anti-diagonal with alternating signs starting with +1 at top-right.
        """
        M = zero_matrix(K, n, n)
        for i in range(n):  # i=0..n-1 corresponds to row i+1
            j = n - 1 - i
            M[i, j] = K((-1) ** i)
        return M

    @classmethod
    def standard_J(cls, K, n):
        r"""
        J = [[0, I_n],
             [-I_n^T, 0]]
        """
        In = cls._alternating_antidiagonal_matrix(K, n)
        Z = zero_matrix(K, n, n)
        return block_matrix(K, [[Z, In], [-In.transpose(), Z]])

    @staticmethod
    def _similitude_factor_from_g(g, J):
        r"""Given g in GL_{2n}(K), find l such that g^T J g = l J."""
        M = g.transpose() * J * g
        rows, cols = J.nrows(), J.ncols()

        for i in range(rows):
            for j in range(cols):
                if J[i, j] != 0:
                    l = M[i, j] / J[i, j]
                    if M != l * J:
                        raise ValueError("No scalar l satisfies g^T J g = l J for this g.")
                    return l

        raise ValueError("J is the zero matrix (unexpected).")

    @classmethod
    def scalar_center_element(cls, K, n, t):
        r"""Central G_m(K)-element embedded in GSp: t |-> (t I_{2n}, t^2)."""
        t = K(t)
        if t == 0:
            raise ValueError("t must be nonzero.")
        g = t * identity_matrix(K, 2 * n)
        l = t**2
        return cls(g, l, J=cls.standard_J(K, n), check=True)

    # --------------------------
    # Core object
    # --------------------------

    def __init__(self, g, l=None, *, J=None, check=True):
        # NOTE: Matrix is not a Python type in Sage; use Matrix_generic if available.
        try:
            from sage.matrix.matrix_generic import Matrix_generic

            if not isinstance(g, Matrix_generic):
                raise TypeError("g must be a Sage matrix (Matrix_generic).")
        except Exception:
            required = ("nrows", "ncols", "base_ring", "det", "transpose")
            if not all(hasattr(g, a) for a in required):
                raise TypeError("g must behave like a Sage matrix (missing required methods).")

        if g.nrows() != g.ncols():
            raise ValueError("g must be square.")
        if g.nrows() % 2 != 0:
            raise ValueError("g must have even size 2n x 2n.")

        self.K = g.base_ring()
        self.dim = g.nrows()
        self.n = self.dim // 2

        if J is None:
            J = self.standard_J(self.K, self.n)
        if J.nrows() != self.dim or J.ncols() != self.dim:
            raise ValueError("J has incompatible size with g.")
        self.J = J

        if l is None:
            l = self._similitude_factor_from_g(g, self.J)

        self._matrix = Matrix(self.K, g)
        self._similitude = self.K(l)

        if check:
            if self._matrix.det() == 0:
                raise ValueError("g must be invertible (det != 0).")
            if self._similitude == 0:
                raise ValueError("l must be nonzero (unit in the field).")
            if self._matrix.transpose() * self.J * self._matrix != self._similitude * self.J:
                raise ValueError("Condition g^T J g = l J is not satisfied.")

    def to_matrix(self):
        return self._matrix

    def similitude(self):
        """Return the similitude factor l (often denoted λ)."""
        return self._similitude

    def __repr__(self):
        return f"GSp_{2 * self.n} element over {self.K} with similitude l"

    def _latex_(self):
        from sage.misc.latex import latex

        return r"\left(%s,\\ %s\right)" % (latex(self._matrix), latex(self._similitude))

    # --------------------------
    # Group law on GSp
    # --------------------------

    def __eq__(self, other):
        return (
            isinstance(other, GenSympGroup)
            and self.K == other.K
            and self.n == other.n
            and self.J == other.J
            and self._matrix == other._matrix
            and self._similitude == other._similitude
        )

    def __mul__(self, other):
        if not isinstance(other, GenSympGroup):
            return NotImplemented
        if self.K != other.K or self.n != other.n or self.J != other.J:
            raise ValueError("Incompatible elements (different base ring, n, or J).")
        return GenSympGroup(
            self._matrix * other._matrix,
            self._similitude * other._similitude,
            J=self.J,
            check=True,
        )

    def inverse(self):
        return GenSympGroup(self._matrix.inverse(), self._similitude ** (-1), J=self.J, check=True)

    def transpose(self):
        return GenSympGroup(self._matrix.transpose(), self._similitude, J=self.J, check=True)

    # --------------------------
    # Scalar action (central G_m)
    # --------------------------

    def scalar_action(self, t):
        r"""(g,l) -> (t g, t^2 l)."""
        t = self.K(t)
        if t == 0:
            raise ValueError("t must be nonzero.")
        return GenSympGroup(t * self._matrix, (t**2) * self._similitude, J=self.J, check=True)

    # --------------------------
    # Scalar equivalence (PGSp relation)
    # --------------------------

    def _scalar_witness_from_g(self, other):
        r"""If other.g = t * self.g return t, else None."""
        if self.dim != other.dim:
            return None

        for i in range(self.dim):
            for j in range(self.dim):
                a = self._matrix[i, j]
                if a != 0:
                    t = other._matrix[i, j] / a
                    if other._matrix == t * self._matrix:
                        return t
                    return None

        return None

    def scalar_equiv_witness(self, other):
        r"""Return t in K^* such that other = t · self (PGSp), else None."""
        if not isinstance(other, GenSympGroup):
            return None
        if self.K != other.K or self.n != other.n or self.J != other.J:
            return None

        t = self._scalar_witness_from_g(other)
        if t is None or t == 0:
            return None
        if other._similitude != (t**2) * self._similitude:
            return None
        return t

    def is_scalar_equivalent(self, other):
        return self.scalar_equiv_witness(other) is not None

    # --------------------------
    # Canonical representative for PGSp class (field case)
    # --------------------------

    def pgsp_normalized(self):
        r"""Scale so first nonzero entry of g becomes 1."""
        for i in range(self.dim):
            for j in range(self.dim):
                a = self._matrix[i, j]
                if a != 0:
                    t = a ** (-1)
                    return self.scalar_action(t)
        raise RuntimeError("Unexpected: g has no nonzero entry (should not happen).")

    def pgsp_equal(self, other):
        if not isinstance(other, GenSympGroup):
            return False
        if self.K != other.K or self.n != other.n or self.J != other.J:
            return False
        return self.pgsp_normalized() == other.pgsp_normalized()

    # --------------------------
    # PGSp-level operations
    # --------------------------

    def pgsp_mul(self, other):
        if not isinstance(other, GenSympGroup):
            raise TypeError("other must be a GenSympGroup.")
        if self.K != other.K or self.n != other.n or self.J != other.J:
            raise ValueError("Incompatible elements.")
        return (self * other).pgsp_normalized()

    def pgsp_inverse(self):
        return self.inverse().pgsp_normalized()

    @classmethod
    def identity(cls, K, n, *, J=None):
        if J is None:
            J = cls.standard_J(K, n)
        return cls(identity_matrix(K, 2 * n), K(1), J=J, check=True)

    def __pow__(self, k):
        try:
            k = int(k)
        except Exception as e:
            raise TypeError("Exponent must be an integer.") from e

        if k == 0:
            return GenSympGroup.identity(self.K, self.n, J=self.J)
        if k < 0:
            return (self.inverse()) ** (-k)

        result = GenSympGroup.identity(self.K, self.n, J=self.J)
        base = self
        e = k
        while e > 0:
            if e & 1:
                result = result * base
            base = base * base
            e >>= 1
        return result

    def pgsp_pow(self, k):
        return (self ** k).pgsp_normalized()

    def Ad(self, lie_elem, check=True):
        r"""Adjoint action: Ad_g(X,ell) = (g X g^{-1}, ell)."""
        if not isinstance(lie_elem, GenSympLie):
            raise TypeError("lie_elem must be a GenSympLie.")
        if self.K != lie_elem.K or self.n != lie_elem.n or self.J != lie_elem.J:
            raise ValueError("Incompatible (different base ring, n, or J).")

        Xnew = self._matrix * lie_elem.X * self._matrix.inverse()
        return GenSympLie(Xnew, lie_elem.ell, J=self.J, check=check)


class GenSympLie:
    r"""Lie algebra: {(X, ell) | X^T J + J X = ell J}."""

    @staticmethod
    def _is_matrix(obj):
        try:
            from sage.matrix.matrix_generic import Matrix_generic

            return isinstance(obj, Matrix_generic)
        except Exception:
            req = ("nrows", "ncols", "base_ring", "transpose")
            return all(hasattr(obj, a) for a in req)

    @staticmethod
    def _scalar_from_X(X, J):
        M = X.transpose() * J + J * X
        for i in range(J.nrows()):
            for j in range(J.ncols()):
                if J[i, j] != 0:
                    ell = M[i, j] / J[i, j]
                    if M != ell * J:
                        raise ValueError("No scalar ell satisfies X^T J + J X = ell J for this X.")
                    return ell
        raise ValueError("J is the zero matrix (unexpected).")

    def __init__(self, X, ell=None, *, J=None, check=True):
        if not self._is_matrix(X):
            raise TypeError("X must be a Sage matrix (Matrix_generic-like).")
        if X.nrows() != X.ncols() or X.nrows() % 2 != 0:
            raise ValueError("X must be a square matrix of even size 2n.")

        self.K = X.base_ring()
        self.dim = X.nrows()
        self.n = self.dim // 2

        if J is None:
            J = GenSympGroup.standard_J(self.K, self.n)
        if J.nrows() != self.dim or J.ncols() != self.dim:
            raise ValueError("J has incompatible size with X.")
        self.J = J

        self.X = Matrix(self.K, X)
        if ell is None:
            ell = self._scalar_from_X(self.X, self.J)
        self.ell = self.K(ell)

        if check:
            if self.X.transpose() * self.J + self.J * self.X != self.ell * self.J:
                raise ValueError("Condition X^T J + J X = ell J is not satisfied.")

    def _latex_(self):
        from sage.misc.latex import latex

        return r"\left(%s,\\ %s\right)" % (latex(self.X), latex(self.ell))

    def __add__(self, other):
        if not isinstance(other, GenSympLie):
            return NotImplemented
        if self.K != other.K or self.n != other.n or self.J != other.J:
            raise ValueError("Incompatible Lie algebra elements.")
        return GenSympLie(self.X + other.X, self.ell + other.ell, J=self.J, check=True)

    def __neg__(self):
        return GenSympLie(-self.X, -self.ell, J=self.J, check=True)

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, a):
        a = self.K(a)
        return GenSympLie(a * self.X, a * self.ell, J=self.J, check=True)

    def bracket(self, other):
        if not isinstance(other, GenSympLie):
            raise TypeError("other must be GenSympLie.")
        if self.K != other.K or self.n != other.n or self.J != other.J:
            raise ValueError("Incompatible Lie algebra elements.")
        comm = self.X * other.X - other.X * self.X
        return GenSympLie(comm, self.K(0), J=self.J, check=True)

    def __matmul__(self, other):
        return self.bracket(other)

    @classmethod
    def zero(cls, K, n, *, J):
        return cls(zero_matrix(K, 2 * n, 2 * n), K(0), J=J, check=True)

    def transpose(self):
        return GenSympLie(self.X.transpose(), self.ell, J=self.J, check=True)

    def scalar_action(self, c):
        r"""(X,ell) -> (X + c*I, ell + 2c)."""
        c = self.K(c)
        return GenSympLie(self.X + c * identity_matrix(self.K, self.X.nrows()), self.ell + 2 * c, J=self.J, check=True)

    def to_sp(self):
        return self.scalar_action(-self.ell / 2)


class GenSympGroups:
    def __init__(self, n, K):
        """Convenience wrapper for gsp/sp computations used in the notebook."""
        self.n = n
        self.base_ring = K
        self.J = self.create_standard_form()

        # indexing for the (one particular) Lie basis used in the notebook
        index_positive_roots = sum([[(i, j) for j in range(i + 1, 2 * n - i)] for i in range(n)], [])
        self.index_positive_roots = [(k, hight, 1) for k, hight in index_positive_roots]
        self.index_negative_roots = [(k, hight, -1) for k, hight in index_positive_roots]

    def identity(self):
        return GenSympGroup(identity_matrix(self.base_ring, 2 * self.n), self.base_ring(1))

    def create_standard_form(self):
        n = self.n

        def elem(i, j):
            if i + j == n - 1:
                return (-1) ** (i)
            return 0

        I = matrix(self.base_ring, n, n, elem)
        J = block_matrix(2, 2, [zero_matrix(n, n), I, -I.transpose(), zero_matrix(n, n)])
        return J

    def is_symplectic(self, M, l):
        if M.dimensions() != (self.n * 2, self.n * 2):
            return False
        return (M.transpose() * self.J * M - l * self.J).is_zero()

    def is_symplectic_lie(self, M, l):
        if M.dimensions() != (self.n * 2, self.n * 2):
            return False
        return (M.transpose() * self.J + self.J * M - l * self.J).is_zero()

    def dimension(self):
        return self.n * (2 * self.n + 1)

    def x(self, i, t):
        n = self.n
        if i < n - 1:
            g = (
                elementary_matrix(self.base_ring, 2 * n, row1=i, row2=i + 1, scale=t)
                + elementary_matrix(self.base_ring, 2 * n, row1=2 * n - i - 2, row2=2 * n - i - 1, scale=t)
                - identity_matrix(self.base_ring, 2 * n)
            )
            return GenSympGroup(g)
        if i == n - 1:
            g = elementary_matrix(self.base_ring, 2 * n, row1=n - 1, row2=n, scale=t)
            return GenSympGroup(g)
        raise ValueError("invalid i")

    def y(self, i, t):
        n = self.n
        if i < n - 1:
            g = (
                elementary_matrix(self.base_ring, 2 * n, row1=i + 1, row2=i, scale=t)
                + elementary_matrix(self.base_ring, 2 * n, row1=2 * n - i - 1, row2=2 * n - i - 2, scale=t)
                - identity_matrix(self.base_ring, 2 * n)
            )
            return GenSympGroup(g)
        if i == n - 1:
            g = elementary_matrix(self.base_ring, 2 * n, row1=n, row2=n - 1, scale=t)
            return GenSympGroup(g)
        raise ValueError("invalid i")

    def s_bar(self, i):
        return self.y(i, 1) * self.x(i, -1) * self.y(i, 1)

    def s_bar_bar(self, i):
        return self.x(i, 1) * self.y(i, -1) * self.x(i, 1)

    def w0(self):
        from sage.all import RootSystem, WeylGroup, prod

        rs = RootSystem(["C", self.n])
        W = WeylGroup(rs.root_lattice(), prefix="s")
        w = W.long_element(as_word=True)
        return prod(self.s_bar(i - 1) for i in w)

    def alpha(self, i, h):
        h = h.diagonal()
        n = self.n
        if i < n - 1:
            return h[i] / h[i + 1]
        if i == n - 1:
            return h[i] ** 2
        raise ValueError("invalid i")

    def alpha_check(self, i, t):
        n = self.n
        if i < n - 1:

            def ent(k):
                if k == i or k == 2 * n - 2 - i:
                    return t
                if k == i + 1 or k == 2 * n - 1 - i:
                    return 1 / t
                return 1

            return GenSympGroup(diagonal_matrix([ent(k) for k in range(2 * self.n)]))

        if i == n - 1:

            def ent(k):
                if k == n - 1:
                    return t
                if k == n:
                    return 1 / t
                return 1

            return GenSympGroup(diagonal_matrix([ent(k) for k in range(2 * self.n)]))

        raise ValueError("invalid i")

    def __repr__(self):
        return "Symplectic Group Sp"

    def star(self, h):
        return self.w0() * h.inverse().transpose() * self.w0().inverse()

    def lie_at(self, i):
        n = self.n

        def ent(k):
            if i < n - 1:
                if k == i or k == 2 * n - 2 - i:
                    return 1
                if k == i + 1 or k == 2 * n - 1 - i:
                    return -1
                return 0
            if i == n - 1:
                if k == n - 1:
                    return 1
                if k == n:
                    return -1
                return 0
            raise ValueError("invalid i")

        X = diagonal_matrix(self.base_ring, [ent(x) for x in range(2 * n)])
        return GenSympLie(X)

    def lie_at_positive_root(self, i, j):
        n = self.n
        if not (0 <= i < n and i + 1 <= j < 2 * n - i):
            raise ValueError("out of index")

        def ent(ii, jj):
            if ii == i and jj == j:
                if i + j == 2 * n - 1:
                    return 1 / 2
                return 1
            return 0

        A = matrix(self.base_ring, 2 * n, 2 * n, ent)
        X = A + (self.w0().to_matrix() * A * self.w0().to_matrix()).transpose()
        return GenSympLie(X)

    def lie_at_root(self, i, j, sign):
        if sign == 1:
            return self.lie_at_positive_root(i, j)
        if sign == -1:
            return self.lie_at_positive_root(i, j).transpose()
        raise ValueError("invalid sign")

    def lie_basis(self, i):
        # keep the match/case style from the notebook for compatibility
        match i:
            case i if i in range(self.n):
                return self.lie_at(i)
            case (ii, jj, sign):
                return self.lie_at_root(ii, jj, sign)
            case _:
                raise ValueError("invalid basis index")

    def coefficient(self, X, index):
        X = X.to_sp().X
        if index in self.index_positive_roots or index in self.index_negative_roots:
            i, j, sign = index
            if sign == 1:
                return X[i][j]
            if sign == -1:
                return X[j][i]
            raise ValueError(f"invalid input: {index}")
        if index in range(self.n):
            i = index
            return sum(X[j][j] for j in range(i + 1))
        raise ValueError(f"invalid input: {index}")
