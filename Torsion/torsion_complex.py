# Intended to be used inside a SageMath environment.

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from sage.all import matrix, vector


@dataclass(frozen=True)
class ComplexIndices:
    """Index sets for C_0..C_3 used by the notebook."""

    basis_index: List[Any]
    cell0_index: List[Tuple[Any, int]]
    cell1_index: List[Tuple[Any, Tuple[str, int]]]
    cell2_index: List[Tuple[Any, Tuple[str, int]]]
    cell3_index: List[Tuple[Any, int]]


class TorsionComplexBuilder:
    """Build differentials for the specific cell structure used in the notebook.

    Parameters
    ----------
    tet0, tet1:
        Instances of Conf4.
    G:
        A GenSympGroups instance. If omitted, we take tet0.T012.G.

    Notes
    -----
    This class is a relatively direct extraction of notebook code. The intent is
    to keep it easy to compare while allowing re-use for other (tet0, tet1).
    """

    def __init__(self, tet0, tet1, G=None):
        self.tet0 = tet0
        self.tet1 = tet1
        self.G = G if G is not None else tet0.T012.G
        self.indices = self._build_indices()

    # -----------------
    # Index conventions
    # -----------------

    def _build_indices(self) -> ComplexIndices:
        G = self.G
        basis_index = list(range(G.n)) + G.index_positive_roots + G.index_negative_roots
        cell0_index = [(k, i) for i in range(4) for k in basis_index]
        cell1_index = (
            [(k, ("long", i)) for i in range(2) for k in basis_index]
            + [(k, ("short", i)) for i in range(12) for k in basis_index]
        )
        cell2_index = (
            [(k, ("hexagon", i)) for i in range(4) for k in basis_index]
            + [(k, ("triangle", i)) for i in range(8) for k in basis_index]
        )
        cell3_index = [(k, i) for i in range(2) for k in basis_index]
        return ComplexIndices(basis_index, cell0_index, cell1_index, cell2_index, cell3_index)

    # -----------------
    # Low-level helpers
    # -----------------

    @staticmethod
    def _dic_update(dic: Dict[Hashable, Any], key: Hashable, value: Any) -> None:
        if key in dic:
            dic[key] += value
        else:
            dic[key] = value

    def _d_aux(
        self,
        X,
        g_list: Sequence,
        e_list: Sequence[Hashable],
        degree_list: Optional[Sequence[Any]] = None,
    ) -> Dict[Hashable, Any]:
        G = self.G
        if degree_list is None:
            degree_list = [1] * len(g_list)
        Y_list = [g.Ad(X) for g in g_list]

        dic: Dict[Hashable, Any] = {}
        for i in range(G.n):
            c_list = [G.coefficient(Y, i) for Y in Y_list]
            for k, c in enumerate(c_list):
                if c != 0:
                    self._dic_update(dic, (i, e_list[k]), degree_list[k] * c)

        for ii, jj, sign in G.index_positive_roots + G.index_negative_roots:
            c_list = [G.coefficient(Y, (ii, jj, sign)) for Y in Y_list]
            for k, c in enumerate(c_list):
                if c != 0:
                    self._dic_update(dic, ((ii, jj, sign), e_list[k]), degree_list[k] * c)
        return dic

    def _to_vector(self, dic: Dict[Hashable, Any], index_set: Sequence[Hashable]):
        v = [dic.get(i, 0) for i in index_set]
        return vector(self.G.base_ring, v)

    # -----------------
    # d1 components
    # -----------------

    def _d1_short(self, X, i: int) -> Dict[Hashable, Any]:
        tet0 = self.tet0
        G = self.G

        if i == 0:
            return self._d_aux(X, [G.identity(), tet0.beta_op(0, 1, 2)], [0, 1], [-1, 1])
        if i == 3:
            return self._d_aux(X, [G.identity(), tet0.beta_op(1, 2, 0)], [1, 2], [-1, 1])
        if i == 6:
            return self._d_aux(X, [G.identity(), tet0.beta_op(3, 2, 1)], [2, 3], [-1, 1])
        if i == 9:
            return self._d_aux(X, [G.identity(), tet0.beta_op(2, 1, 3)], [3, 0], [-1, 1])

        if i == 1:
            return self._d_aux(X, [G.identity(), tet0.beta_op(0, 3, 1)], [0, 0], [-1, 1])
        if i == 4:
            return self._d_aux(X, [G.identity(), tet0.beta_op(1, 3, 2)], [1, 1], [-1, 1])
        if i == 7:
            return self._d_aux(X, [G.identity(), tet0.beta_op(3, 0, 2)], [2, 2], [-1, 1])
        if i == 10:
            return self._d_aux(X, [G.identity(), tet0.beta_op(2, 0, 1)], [3, 3], [-1, 1])

        if i == 2:
            return self._d_aux(X, [G.identity(), tet0.beta_op(0, 2, 3)], [1, 0], [-1, 1])
        if i == 5:
            return self._d_aux(X, [G.identity(), tet0.beta_op(1, 0, 3)], [2, 1], [-1, 1])
        if i == 8:
            return self._d_aux(X, [G.identity(), tet0.beta_op(3, 1, 0)], [3, 2], [-1, 1])
        if i == 11:
            return self._d_aux(X, [G.identity(), tet0.beta_op(2, 3, 0)], [0, 3], [-1, 1])

        raise ValueError(f"invalid short edge index i={i}")

    def _d1_long(self, X, i: int) -> Dict[Hashable, Any]:
        tet0 = self.tet0
        G = self.G

        if i == 0:
            return self._d_aux(
                X,
                [
                    G.identity(),
                    tet0.alpha_op(1, 2)
                ],
                [1, 3],
                [-1, 1],
            )
        if i == 1:
            return self._d_aux(
                X,
                [G.identity(), tet0.alpha_op(0, 1)],
                [0, 2],
                [-1, 1],
            )
        raise ValueError(f"invalid long edge index i={i}")

    # -----------------
    # d2 components
    # -----------------

    def _d2_triangle(self, X, i: int) -> Dict[Hashable, Any]:
        tet0 = self.tet0
        tet1 = self.tet1
        G = self.G

        if i == 0:
            return self._d_aux(X, [G.identity(), tet0.beta_op(0, 1, 2), tet0.beta_op(0, 1, 3)], [("short", 0), ("short", 2), ("short", 1)], [1, 1, 1])
        if i == 1:
            return self._d_aux(X, [G.identity(), tet0.beta_op(1, 2, 0), tet0.beta_op(1, 2, 3)], [("short", 3), ("short", 5), ("short", 4)], [1, 1, 1])
        if i == 2:
            return self._d_aux(X, [G.identity(), tet0.beta_op(3, 2, 1), tet0.beta_op(3, 2, 0)], [("short", 6), ("short", 8), ("short", 7)], [1, 1, 1])
        if i == 3:
            return self._d_aux(X, [G.identity(), tet0.beta_op(2, 1, 3), tet0.beta_op(2, 1, 0)], [("short", 9), ("short", 11), ("short", 10)], [1, 1, 1])
        if i == 4:
            return self._d_aux(X, [G.identity(), tet1.beta_op(0, 2, 3), tet1.beta_op(0, 2, 1)], [("short", 0), ("short", 4), ("short", 2)], [1, 1, 1])
        if i == 5:
            return self._d_aux(X, [G.identity(), tet1.beta_op(2, 3, 0), tet1.beta_op(2, 3, 1)], [("short", 3), ("short", 7), ("short", 5)], [1, 1, 1])
        if i == 6:
            return self._d_aux(X, [G.identity(), tet1.beta_op(3, 1, 0), tet1.beta_op(3, 1, 2)], [("short", 6), ("short", 10), ("short", 8)], [1, 1, 1])
        if i == 7:
            return self._d_aux(X, [G.identity(), tet1.beta_op(1, 0, 3), tet1.beta_op(1, 0, 2)], [("short", 9), ("short", 1), ("short", 11)], [1, 1, 1])
        raise ValueError(f"invalid triangle index i={i}")

    def _d2_hexagon(self, X, i: int) -> Dict[Hashable, Any]:
        tet0 = self.tet0
        G = self.G

        if i == 1:
            g_list = [
                tet0.beta_op(0, 3, 2),
                tet0.beta_op(2, 0, 3) * tet0.alpha_op(0, 2) * tet0.beta_op(0, 3, 2),
                G.identity(),
                tet0.beta_op(2, 0, 3) * tet0.alpha_op(0, 2) * tet0.beta_op(0, 3, 2),
                tet0.alpha_op(0, 3),
                tet0.beta_op(0, 3, 2),
            ]
            e_list = [("long", 0), ("long", 1), ("long", 1), ("short", 11), ("short", 7), ("short", 2)]
            degree_list = [1, 1, -1, -1, -1, -1]
            return self._d_aux(X, g_list, e_list, degree_list)

        if i == 3:
            g_list = [
                G.identity(),
                tet0.beta_op(1, 0, 2) * tet0.alpha_op(0, 1),
                tet0.beta_op(0, 1, 2),
                tet0.beta_op(1, 0, 2) * tet0.alpha_op(0, 1),
                tet0.alpha_op(0, 2) * tet0.beta_op(0, 1, 2),
                G.identity(),
            ]
            e_list = [("long", 1), ("long", 0), ("long", 0), ("short", 3), ("short", 10), ("short", 0)]
            degree_list = [1, 1, -1, -1, -1, -1]
            return self._d_aux(X, g_list, e_list, degree_list)

        if i == 0:
            g_list = [
                tet0.beta_op(1, 3, 2),
                tet0.beta_op(2, 1, 3) * tet0.alpha_op(1, 2) * tet0.beta_op(1, 3, 2),
                G.identity(),
                tet0.alpha_op(1, 2) * tet0.beta_op(1, 3, 2),
                tet0.beta_op(3, 1, 2) * tet0.alpha_op(1, 3),
                G.identity(),
            ]
            e_list = [("long", 0), ("long", 1), ("long", 0), ("short", 9), ("short", 6), ("short", 4)]
            degree_list = [1, 1, -1, 1, 1, 1]
            return self._d_aux(X, g_list, e_list, degree_list)

        if i == 2:
            g_list = [
                tet0.alpha_op(1, 0) * tet0.beta_op(1, 3, 0),
                G.identity(),
                tet0.beta_op(0, 1, 3) * tet0.alpha_op(1, 0) * tet0.beta_op(1, 3, 0),
                tet0.beta_op(1, 3, 0),
                tet0.alpha_op(1, 3),
                tet0.beta_op(0, 1, 3) * tet0.alpha_op(1, 0) * tet0.beta_op(1, 3, 0),
            ]
            e_list = [("long", 1), ("long", 0), ("long", 1), ("short", 5), ("short", 8), ("short", 1)]
            degree_list = [1, 1, -1, 1, 1, 1]
            return self._d_aux(X, g_list, e_list, degree_list)

        raise ValueError(f"invalid hexagon index i={i}")

    # -----------------
    # d3 component
    # -----------------

    def _d3(self, X, i: int) -> Dict[Hashable, Any]:
        tet0 = self.tet0
        tet1 = self.tet1
        G = self.G

        if i == 0:
            g_hexagon = [
                tet0.beta_op(1, 0, 3) * tet0.alpha_op(0, 1),
                tet0.beta_op(0, 1, 3),
                tet0.beta_op(1, 0, 3) * tet0.alpha_op(0, 1),
                G.identity(),
            ]
            g_triangle = [
                G.identity(),
                tet0.beta_op(1, 0, 2) * tet0.alpha_op(0, 1),
                tet0.beta_op(3, 0, 2) * tet0.alpha_op(0, 3) * tet0.beta_op(0, 1, 3),
                tet0.beta_op(2, 0, 1) * tet0.alpha_op(0, 2) * tet0.beta_op(0, 1, 2),
            ]
            return self._d_aux(
                X,
                g_hexagon + g_triangle,
                [("hexagon", k) for k in range(4)] + [("triangle", k) for k in range(4)],
                [-1, 1, -1, 1] + [1] * 4,
            )

        if i == 1:
            g_hexagon = [
                tet1.beta_op(0, 1, 3),
                tet1.beta_op(0, 1, 2),
                tet1.beta_op(2, 0, 3) * tet1.alpha_op(0, 2) * tet1.beta_op(0, 1, 2),
                tet1.beta_op(0, 1, 2),
            ]
            g_triangle = [
                tet1.beta_op(0, 1, 2),
                tet1.beta_op(2, 0, 3) * tet1.alpha_op(0, 2) * tet1.beta_op(0, 1, 2),
                tet1.beta_op(3, 0, 1) * tet1.alpha_op(0, 3) * tet1.beta_op(0, 1, 3),
                tet1.alpha_op(0, 1),
            ]
            return self._d_aux(
                X,
                g_hexagon + g_triangle,
                [("hexagon", k) for k in range(4)] + [("triangle", k) for k in range(4, 8)],
                [1, -1, 1, -1] + [-1] * 4,
            )

        raise ValueError(f"invalid 3-cell index i={i}")

    # -----------------
    # Public API
    # -----------------

    def differential(self, dim: int, idx):
        """Return the differential as a Sage vector, matching notebook convention."""
        G = self.G
        ind = self.indices

        if dim == 1:
            match idx:
                case (i, ("long", c)):
                    return self._to_vector(self._d1_long(G.lie_basis(i), c), ind.cell0_index)
                case (i, ("short", c)):
                    return self._to_vector(self._d1_short(G.lie_basis(i), c), ind.cell0_index)
                case _:
                    raise ValueError(f"invalid dim=1 index {idx}")

        if dim == 2:
            match idx:
                case (i, ("hexagon", c)):
                    return self._to_vector(self._d2_hexagon(G.lie_basis(i), c), ind.cell1_index)
                case (i, ("triangle", c)):
                    return self._to_vector(self._d2_triangle(G.lie_basis(i), c), ind.cell1_index)
                case _:
                    raise ValueError(f"invalid dim=2 index {idx}")

        if dim == 3:
            match idx:
                case (i, c):
                    return self._to_vector(self._d3(G.lie_basis(i), c), ind.cell2_index)
                case _:
                    raise ValueError(f"invalid dim=3 index {idx}")

        raise ValueError(f"invalid dim={dim}")

    def differential_matrices(self):
        """Return (d1, d2, d3) as Sage matrices, transposed as in the notebook."""
        ind = self.indices
        d1 = matrix([self.differential(1, i) for i in ind.cell1_index]).transpose()
        d2 = matrix([self.differential(2, i) for i in ind.cell2_index]).transpose()
        d3 = matrix([self.differential(3, i) for i in ind.cell3_index]).transpose()
        return d1, d2, d3
