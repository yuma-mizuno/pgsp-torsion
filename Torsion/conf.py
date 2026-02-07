# Intended to be used inside a SageMath environment.

from sage.all import QQ, LaurentPolynomialRing, MatrixSpace, diagonal_matrix

from gensymp import GenSympGroup, GenSympGroups


class Conf3:
    def __init__(self, base_ring=None, variables=None):
        if base_ring is None:
            self.base_ring = LaurentPolynomialRing(
                QQ, 8, "A10, A11, A12, A20, A21, A22, A_1e, A_2e"
            )
            self.variables = self.base_ring.gens()
        else:
            self.base_ring = base_ring
            self.variables = variables
        self.G = GenSympGroups(2, self.base_ring)

    def h(self):
        G = self.G
        A10, A11, A12, A20, A21, A22, A1e, A2e = self.variables
        _ = MatrixSpace(self.base_ring, 2 * self.G.n, 2 * self.G.n)
        return G.alpha_check(0, A10) * G.alpha_check(1, A20)

    def hp(self):
        G = self.G
        A10, A11, A12, A20, A21, A22, A1e, A2e = self.variables
        return G.alpha_check(0, A12) * G.alpha_check(1, A22)

    def angle_at(self, i):
        A10, A11, A12, A20, A21, A22, A1e, A2e = self.variables
        if i == 0:
            return A1e / A10 / A11 * A20
        if i == 1:
            return 1 / A20 / A21 * A11**2
        if i == 2:
            return 1 / A11 / A12 * A21
        if i == 3:
            return A2e / A21 / A22 * A12**2
        raise ValueError("invalid i")

    def angle(self):
        G = self.G
        return (
            G.x(0, self.angle_at(0))
            * G.x(1, self.angle_at(1))
            * G.x(0, self.angle_at(2))
            * G.x(1, self.angle_at(3))
        )

    def angle_decomposition(self):
        _, L, U = (self.angle().to_matrix() * self.G.w0().to_matrix()).LU(pivot="nonzero")
        phip_u = GenSympGroup(L)
        uw0 = GenSympGroup(diagonal_matrix(U.diagonal()))
        phi_u = self.G.w0() * uw0.inverse() * GenSympGroup(U) * self.G.w0().inverse()
        return (phip_u, uw0, phi_u)

    def alpha_op(self, i, j):
        h = self.h()
        hp = self.hp()
        G = self.G
        h_star = G.star(h)
        hp_star = G.star(hp)
        phip_u, uw0, phi_u = self.angle_decomposition()

        if (i, j) == (0, 1):
            return hp.inverse() * G.w0()
        if (i, j) == (1, 0):
            return self.alpha_op(j, i).inverse() * (G.w0() ** 2)
        if (i, j) == (1, 2):
            return hp_star.inverse() * uw0.inverse() * h.inverse() * G.w0()
        if (i, j) == (2, 1):
            return self.alpha_op(j, i).inverse() * (G.w0() ** 2)
        if (i, j) == (2, 0):
            return G.w0() * h
        if (i, j) == (0, 2):
            return h.inverse() * G.w0()
        raise ValueError("invalid (i,j)")

    def alpha(self, i, j):
        return self.alpha_op(i, j).inverse()

    def beta_op(self, v):
        h = self.h()
        hp = self.hp()
        G = self.G
        h_star = G.star(h)
        hp_star = G.star(hp)
        phip_u, uw0, phi_u = self.angle_decomposition()

        if v == 0:
            return self.angle()
        if v == 1:
            return G.w0().inverse() * hp * phi_u.inverse() * G.w0() * hp_star
        if v == 2:
            return G.w0().inverse() * h * phip_u.inverse() * h.inverse() * G.w0()
        raise ValueError("invalid v")

    def beta(self, v):
        return self.beta_op(v).inverse()


class Conf4:
    def __init__(self, base_ring, variables):
        (
            a01,
            a02,
            a03,
            a10,
            a12,
            a13,
            a20,
            a21,
            a23,
            a30,
            a31,
            a32,
            f10,
            f20,
            f11,
            f21,
            f12,
            f22,
            f13,
            f23,
        ) = variables

        self.a01 = base_ring(a01)
        self.a02 = base_ring(a02)
        self.a03 = base_ring(a03)
        self.a10 = base_ring(a10)
        self.a12 = base_ring(a12)
        self.a13 = base_ring(a13)
        self.a20 = base_ring(a20)
        self.a21 = base_ring(a21)
        self.a23 = base_ring(a23)
        self.a30 = base_ring(a30)
        self.a31 = base_ring(a31)
        self.a32 = base_ring(a32)
        self.f10 = base_ring(f10)
        self.f20 = base_ring(f20)
        self.f11 = base_ring(f11)
        self.f21 = base_ring(f21)
        self.f12 = base_ring(f12)
        self.f22 = base_ring(f22)
        self.f13 = base_ring(f13)
        self.f23 = base_ring(f23)
        self.base_ring = base_ring

        variables012 = [- self.a02, self.f13, self.a12, self.a20, self.f23, self.a21, self.a01, self.a10]
        variables023 = [self.a30, self.f11, self.a23, self.a03, self.f21, self.a32, self.a02, self.a20]
        variables123 = [self.a31, self.f10, self.a23, self.a13, self.f20, self.a32, self.a12, self.a21]
        variables013 = [self.a30, self.f12, -self.a31, self.a03, self.f22, self.a13, self.a01, self.a10]

        self.T012 = Conf3(base_ring, variables012)
        self.T023 = Conf3(base_ring, variables023)
        self.T123 = Conf3(base_ring, variables123)
        self.T013 = Conf3(base_ring, variables013)

    def alpha(self, i, j):
        T012 = self.T012
        T023 = self.T023
        T123 = self.T123
        T013 = self.T013

        if (i, j) == (2, 0):
            if not (T012.alpha(0, 2) * T023.alpha(1, 2).inverse()).is_scalar_equivalent(
                GenSympGroup.identity(self.base_ring, 2)
            ):
                raise ValueError("inconsistent edge holonomy")
            return T012.alpha(0, 2)
        if (i, j) == (0, 2):
            return T012.alpha(2, 0)

        if (i, j) == (1, 0):
            if not (T012.alpha(1, 2) * T013.alpha(1, 2).inverse()).is_scalar_equivalent(
                GenSympGroup.identity(self.base_ring, 2)
            ):
                raise ValueError("inconsistent edge holonomy")
            return T012.alpha(1, 2)
        if (i, j) == (0, 1):
            return T012.alpha(2, 1)

        if (i, j) == (1, 2):
            if not (T012.alpha(0, 1) * T123.alpha(1, 2).inverse()).is_scalar_equivalent(
                GenSympGroup.identity(self.base_ring, 2)
            ):
                raise ValueError("inconsistent edge holonomy")
            return T012.alpha(0, 1)
        if (i, j) == (2, 1):
            return T012.alpha(1, 0)

        if (i, j) == (0, 3):
            if not (T013.alpha(2, 0) * T023.alpha(2, 0).inverse()).is_scalar_equivalent(
                GenSympGroup.identity(self.base_ring, 2)
            ):
                raise ValueError("inconsistent edge holonomy")
            return T013.alpha(2, 0)
        if (i, j) == (3, 0):
            return T013.alpha(0, 2)

        if (i, j) == (1, 3):
            if not (T013.alpha(1, 0) * T123.alpha(2, 0).inverse()).is_scalar_equivalent(
                GenSympGroup.identity(self.base_ring, 2)
            ):
                raise ValueError("inconsistent edge holonomy")
            return T013.alpha(1, 0)
        if (i, j) == (3, 1):
            return T013.alpha(0, 1)

        if (i, j) == (2, 3):
            if not (T023.alpha(0, 1) * T123.alpha(0, 1).inverse()).is_scalar_equivalent(
                GenSympGroup.identity(self.base_ring, 2)
            ):
                raise ValueError("inconsistent edge holonomy")
            return T023.alpha(0, 1)
        if (i, j) == (3, 2):
            return T023.alpha(1, 0)

        raise ValueError("invalid (i,j)")

    def beta(self, v, i, j):
        T012 = self.T012
        T023 = self.T023
        T123 = self.T123
        T013 = self.T013

        # face 012
        if (v, i, j) == (0, 1, 2):
            return T012.beta(2)
        if (v, i, j) == (0, 2, 1):
            return T012.beta(2).inverse()
        if (v, i, j) == (1, 2, 0):
            return T012.beta(1)
        if (v, i, j) == (1, 0, 2):
            return T012.beta(1).inverse()
        if (v, i, j) == (2, 0, 1):
            return T012.beta(0)
        if (v, i, j) == (2, 1, 0):
            return T012.beta(0).inverse()

        # face 013
        if (v, i, j) == (0, 1, 3):
            return T013.beta(2)
        if (v, i, j) == (0, 3, 1):
            return T013.beta(2).inverse()
        if (v, i, j) == (1, 3, 0):
            return T013.beta(1)
        if (v, i, j) == (1, 0, 3):
            return T013.beta(1).inverse()
        if (v, i, j) == (3, 0, 1):
            return T013.beta(0)
        if (v, i, j) == (3, 1, 0):
            return T013.beta(0).inverse()

        # face 023
        if (v, i, j) == (0, 2, 3):
            return T023.beta(2)
        if (v, i, j) == (0, 3, 2):
            return T023.beta(2).inverse()
        if (v, i, j) == (2, 3, 0):
            return T023.beta(1)
        if (v, i, j) == (2, 0, 3):
            return T023.beta(1).inverse()
        if (v, i, j) == (3, 0, 2):
            return T023.beta(0)
        if (v, i, j) == (3, 2, 0):
            return T023.beta(0).inverse()

        # face 123
        if (v, i, j) == (1, 2, 3):
            return T123.beta(2)
        if (v, i, j) == (1, 3, 2):
            return T123.beta(2).inverse()
        if (v, i, j) == (2, 3, 1):
            return T123.beta(1)
        if (v, i, j) == (2, 1, 3):
            return T123.beta(1).inverse()
        if (v, i, j) == (3, 1, 2):
            return T123.beta(0)
        if (v, i, j) == (3, 2, 1):
            return T123.beta(0).inverse()

        raise ValueError("invalid (v,i,j)")