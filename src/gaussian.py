"""Gaussian elimination solver for square linear systems."""

import re

import numpy as np


class Gaussian:
    """Solve A·x = b using Gaussian elimination with partial pivoting.

    Parses equation strings, builds the augmented matrix [A|b], performs
    forward elimination to row echelon form, then back-substitutes to RREF.

    Attributes:
        A (np.ndarray): Coefficient matrix of shape (n, n).
        B (np.ndarray): Right-hand side vector of shape (n, 1).
        n (int): System dimension.
        x (list[str]): Sorted list of variable names.
        solution (dict[str, float] | None): Dict mapping variable names to
                                            solution values, or None if
                                            unsolved.

    Example:
        >>> ge = Gaussian(["x + y = 3", "x - y = -1"])
        >>> ge.solve()
        >>> ge.solution
        {'x': 1.0, 'y': 2.0}
    """

    # -------------------------------------------------------------------------
    #                        DUNDER METHODS
    # -------------------------------------------------------------------------

    def __init__(self, equations: list[str]) -> None:
        """Initialize the solver by parsing equation strings.

        Args:
            equations (list[str]): Equation strings in the form
                ``'2x + 3y = 8'``. Implicit coefficient of 1 is supported
                (e.g. ``'x - y = 0'``).

        Raises:
            ValueError: If the system is not square or an equation cannot
                be parsed.
        """
        A, b, x = self._parse_equations(equations)
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(b.reshape(-1, 1), dtype=np.float64)
        self.x = x
        self.n = self.A.shape[0]
        self._augmented_M: np.ndarray = self._build_augmented(self.A, self.B)
        self._echelon_M: np.ndarray | None = None
        self._reduced_echelon_M: np.ndarray | None = None
        self.solution: dict[str, float] | None = None

    def __repr__(self) -> str:
        """Return a concise developer-facing representation.

        Returns:
            str: String of the form
            ``'Gaussian(n=3, variables=[x, y, z], status=solved)'``.
        """
        status = "solved" if self.solution else "unsolved"
        return f"\nGaussian(n={self.n}, variables={self.x}, status={status})\n"

    def __str__(self) -> str:
        """Return a human-readable summary of the system and its status.

        Returns:
            str: Multi-line string showing dimension, variables, solution
            values (if solved), and residual magnitude.
        """
        lines = [f"\nGaussian -- {self.n}x{self.n} system"]
        lines.append(f"  Variables : {self.x}")
        if self.solution:
            sol_str = ", ".join(
                f"{k}={v:.2f}" for k, v in self.solution.items()
            )
            lines.append(f"  Solution  : {sol_str}")
            lines.append(f"  Residual  : {self.residual():.2e}")
        else:
            lines.append("  (not yet solved -- call .solve())")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    #                        PUBLIC METHODS
    # -------------------------------------------------------------------------

    def solve(self) -> None:
        """Solve the system and store the result in ``self.solution``.

        Runs forward elimination followed by back substitution. Prints a warning
        and leaves ``solution`` as ``None`` if the matrix is singular.
        """
        self._echelon_M = self._echelon_form()
        if self._echelon_M is None:
            print("Singular Matrix")
            return None
        self._reduced_echelon_M = self._reduced_echelon_form()
        if self._reduced_echelon_M is not None:
            solution_vec = self._reduced_echelon_M[:, -1]
            solution_vec.round(6)
            self.solution = {
                self.x[i]: float(value) for i, value in enumerate(solution_vec)
            }

    def show_solution(self) -> None:
        """Print each variable's value and the solution residual."""
        print("\nSolution:")
        if self.solution is not None:
            for var, val in self.solution.items():
                print(f"  {var} = {val:.2f}")
        else:
            print(f"  {self.solution}")

    def augmented_M(self) -> None:
        """Print the augmented matrix ``[A | b]`` to stdout."""
        print("\nAugmented Matrix:")
        print(self._augmented_M)
        print("\n")

    def echelon_M(self) -> None:
        """Print the row echelon matrix to stdout, or 'None' if unset."""
        print("\nEchelon Matrix:")
        if self._echelon_M is not None:
            print(self._echelon_M)
        else:
            print(" None")
        print("\n")

    def reduced_echelon_M(self) -> None:
        """Print the reduced echelon matrix to stdout, or 'None' if unset."""
        print("\nReduced Echelon Matrix:")
        if self._reduced_echelon_M is not None:
            print(self._reduced_echelon_M)
        else:
            print(" None")
        print("\n")

    def system(self) -> None:
        """Print the system of equations in equation form."""
        print("\nSystem of Equations:")

        def format_number(num: float) -> str:
            """Format as an int string if whole, else 2 decimal places."""
            if np.isclose(num, round(num)):
                return str(round(num))
            else:
                return f"{num:.2f}"

        for i in range(self.n):
            terms: list[str] = []
            for j in range(self.n):
                coeff = self.A[i, j]
                var = self.x[j]

                if not np.isclose(coeff, 0):
                    coeff_str = format_number(abs(coeff))
                    if terms:
                        # Not the first term, add sign
                        if coeff > 0:
                            terms.append(f"+ {coeff_str}{var}")
                        else:
                            terms.append(f"- {coeff_str}{var}")
                    else:
                        # First term
                        if coeff > 0:
                            terms.append(f"{coeff_str}{var}")
                        else:
                            terms.append(f"- {coeff_str}{var}")

            rhs = self.B[i, 0]
            rhs_str = format_number(rhs)
            equation = " ".join(terms) + f" = {rhs_str}"
            print(f"  {equation}")

    def residual(self) -> float:
        """Compute the infinity-norm residual ``||Ax - b||∞``.

        Returns:
            float: Maximum absolute element of ``A·x - b``. Near-zero
            values (~1e-16) indicate a numerically accurate solution.

        Raises:
            RuntimeError: If ``solve()`` has not been called yet.
        """
        if self.solution is None:
            raise RuntimeError("Call .solve() before checking the residual.")
        x_vector = np.array([self.solution[v] for v in self.x])
        return float(np.max(np.abs(self.A @ x_vector - self.B.flatten())))

    # -------------------------------------------------------------------------
    #                        PRIVATE METHODS
    # -------------------------------------------------------------------------

    def _echelon_form(self) -> np.ndarray | None:
        """Reduce the augmented matrix to row echelon form (forward pass).

        Uses partial pivoting for numerical stability. Each pivot is
        scaled to 1 and elements below it are zeroed out.

        Returns:
            np.ndarray | None: Augmented matrix in row echelon form, or
            ``None`` if A is singular.
        """
        det_A = np.linalg.det(self.A)
        if np.isclose(det_A, 0):
            return None

        M = self._augmented_M.copy()
        for row in range(self.n):
            pivot_candidate = M[row, row]
            if np.isclose(pivot_candidate, 0):
                index_first_non_zero_value_below_pivot_candidate = (
                    self._get_index_first_non_zero_value_from_column(
                        M, row, row
                    )
                )
                M = self._swap_rows(
                    M, row, index_first_non_zero_value_below_pivot_candidate
                )
                pivot = M[row, row]
            else:
                pivot = pivot_candidate

            M[row] = (1 / pivot) * M[row]

            for i in range(row + 1, self.n):
                value_below_pivot = M[i, row]
                M[i] = M[i] - value_below_pivot * M[row]
        return M

    def _reduced_echelon_form(self) -> np.ndarray | None:
        """Reduce echelon form to RREF via back substitution.

        Returns:
            dict[str, float] | None: Dict mapping variable names to float
            solution values, or ``None`` if echelon form is unavailable
            (singular matrix).
        """
        if self._echelon_M is None:
            return None
        M = self._echelon_M.copy()

        for row in reversed(range(self.n)):
            index_pivot_column = self._get_index_first_non_zero_value_from_row(
                M, row
            )
            for i in range(row):
                value_to_reduce = M[i, index_pivot_column]
                M[i] = M[i] - value_to_reduce * M[row]

        return M

    # -------------------------------------------------------------------------
    #                        STATIC METHODS
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_equations(
        equations: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Parse equations into a coefficient matrix, RHS vector, and variables.

        Handles implicit coefficients of 1, mixed signs, and arbitrary
        whitespace. Variables are sorted alphabetically for consistent
        matrix ordering.

        Args:
            equations (list[str]): Strings in the form ``'2x + 3y - z = 10'``.

        Returns:
            tuple[np.ndarray, np.ndarray, list[str]]: A tuple ``(A, b, x)``
            where:

            - A : coefficient matrix of shape ``(n, n)``.
            - b : RHS vector of shape ``(n,)``.
            - x : sorted list of variable name strings.

        Raises:
            ValueError: If the system is not square, the RHS is malformed,
                or an unknown variable is encountered.
        """
        TERM_RE = re.compile(
            r"([+-]?)\s*" r"(\d+\.?\d*)?\s*" r"([A-Za-z][0-9]?\w*)"
        )
        RHS_RE = re.compile(r"=\s*([+-]?\s*\d+\.?\d*)\s*$")

        variables: set[str] = set()
        for equation in equations:
            lhs = equation.split("=")[0]
            for _, _, variable in TERM_RE.findall(lhs):
                if variable not in variables:
                    variables.add(variable)

        variables_sorted = sorted(list(variables))
        vars_index = {
            value: index for index, value in enumerate(variables_sorted)
        }
        num_equations = len(equations)
        num_variables = len(variables_sorted)

        if num_equations != num_variables:
            raise ValueError(
                f"System has {num_equations} equations but "
                f"{num_variables} variables {variables_sorted}. Gaussian "
                "elimination requires a square system."
            )

        A = np.zeros((num_equations, num_variables), dtype=np.float64)
        b = np.zeros(num_equations, dtype=np.float64)
        x = variables_sorted.copy()

        for row, equation in enumerate(equations):
            rhs_match = RHS_RE.search(equation)
            if rhs_match is None:
                raise ValueError(f"Couldn't parse the RHS of: {equation}")
            b[row] = float(rhs_match.group(1).replace(" ", ""))

            lhs = equation.split("=")[0]
            for sign, coeff, variable in TERM_RE.findall(lhs):
                coeff = float(coeff) if coeff else 1.0
                if sign == "-":
                    coeff *= -1.0
                if variable not in vars_index:
                    raise ValueError(
                        f"Unknown variable '{variable}' in: {equation}"
                    )
                A[row, vars_index[variable]] = coeff
        return (A, b, x)

    @staticmethod
    def _build_augmented(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Build the augmented matrix ``[A | B]``.

        Args:
            A (np.ndarray): Coefficient matrix of shape ``(n, n)``.
            B (np.ndarray): RHS matrix of shape ``(n, 1)``.

        Returns:
            np.ndarray: Horizontally stacked matrix of shape ``(n, n+1)``.
        """
        return np.hstack((A, B))

    @staticmethod
    def _swap_rows(
        M: np.ndarray, row_index_1: int, row_index_2: int
    ) -> np.ndarray:
        """Return a copy of M with two rows swapped (used for partial pivoting).

        Args:
            M (np.ndarray): Input matrix (not mutated).
            row_index_1 (int): First row index.
            row_index_2 (int): Second row index.

        Returns:
            np.ndarray: New matrix with the specified rows exchanged.
        """
        M = M.copy()
        M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
        return M

    @staticmethod
    def _get_index_first_non_zero_value_from_column(
        M: np.ndarray, column: int, starting_row: int
    ) -> int:
        """Find the first non-zero row in a column, from ``starting_row`` down.

        Args:
            M (np.ndarray): Matrix to search.
            column (int): Column index to inspect.
            starting_row (int): Row offset to begin the search.

        Returns:
            int: Absolute row index of the first non-zero entry, or ``-1``
            if none found.
        """
        column_vector = M[starting_row:, column]
        for index, value in enumerate(column_vector):
            if not (np.isclose(value, 0)):
                return index + starting_row
        return -1

    @staticmethod
    def _get_index_first_non_zero_value_from_row(
        M: np.ndarray, row: int
    ) -> int:
        """Find the first non-zero column in a row, excluding the last column.

        Args:
            M (np.ndarray): Matrix to search.
            row (int): Row index to inspect.

        Returns:
            int: Column index of the first non-zero entry, or ``-1`` if
            none found.
        """
        M = M[:, :-1]
        row_vector = M[row]
        for index, value in enumerate(row_vector):
            if not np.isclose(value, 0):
                return index
        return -1


# -------------------------------------------------------------------------
#                        DEMO
# -------------------------------------------------------------------------
if __name__ == "__main__":
    equations = [
        "x + y + z = 6",
        "2x + 3y + z = 11",
        "x - y + 2z = 5",
    ]
    solver = Gaussian(equations)
    solver.system()
    solver.augmented_M()
    solver.solve()
    solver.echelon_M()
    solver.reduced_echelon_M()
    solver.show_solution()
    try:
        print(f"\nResidual: {solver.residual():.2e}\n")
    except RuntimeError:
        print("\nResidual: (no solution)\n")
