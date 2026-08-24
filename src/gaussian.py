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
                    prefix = "- " if coeff < 0 else "+ " if terms else ""
                    terms.append(f"{prefix}{coeff_str}{var}")

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

        Uses partial pivoting: for each column, the row with the largest
        remaining magnitude becomes the pivot, which is both more
        numerically stable and doubles as the singularity check (no
        separate determinant computation is needed). Each pivot is scaled
        to 1 and the rows below it are zeroed out in a single vectorized
        update.

        Returns:
            np.ndarray | None: Augmented matrix in row echelon form, or
            ``None`` if A is singular.
        """
        M = self._augmented_M.copy()
        for row in range(self.n):
            pivot_row = row + int(np.argmax(np.abs(M[row:, row])))
            pivot = M[pivot_row, row]
            if np.isclose(pivot, 0):
                return None
            if pivot_row != row:
                M[[row, pivot_row]] = M[[pivot_row, row]]

            M[row] /= pivot
            M[row + 1 :] -= np.outer(M[row + 1 :, row], M[row])
        return M

    def _reduced_echelon_form(self) -> np.ndarray | None:
        """Reduce echelon form to RREF via back substitution.

        Column-only partial pivoting during the forward pass never
        permutes columns, so pivot columns line up with the diagonal --
        row ``row``'s pivot is always column ``row``, and each back
        substitution step clears it above in a single vectorized update.

        Returns:
            np.ndarray | None: Augmented matrix in reduced row echelon
            form, or ``None`` if echelon form is unavailable (singular
            matrix).
        """
        if self._echelon_M is None:
            return None
        M = self._echelon_M.copy()

        for row in reversed(range(1, self.n)):
            M[:row] -= np.outer(M[:row, row], M[row])

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

        # Parse each equation exactly once, then build the matrix in a
        # second pass -- the variable-to-column mapping isn't known until
        # every equation has been scanned.
        variables: set[str] = set()
        parsed_equations: list[tuple[list[tuple[str, str, str]], float]] = []
        for equation in equations:
            lhs = equation.partition("=")[0]
            terms = TERM_RE.findall(lhs)
            variables.update(variable for _, _, variable in terms)

            rhs_match = RHS_RE.search(equation)
            if rhs_match is None:
                raise ValueError(f"Couldn't parse the RHS of: {equation}")
            rhs_value = float(rhs_match.group(1).replace(" ", ""))
            parsed_equations.append((terms, rhs_value))

        x = sorted(variables)
        vars_index = {name: index for index, name in enumerate(x)}
        num_equations = len(equations)
        num_variables = len(x)

        if num_equations != num_variables:
            raise ValueError(
                f"System has {num_equations} equations but "
                f"{num_variables} variables {x}. Gaussian "
                "elimination requires a square system."
            )

        A = np.zeros((num_equations, num_variables), dtype=np.float64)
        b = np.zeros(num_equations, dtype=np.float64)

        for row, (terms, rhs_value) in enumerate(parsed_equations):
            b[row] = rhs_value
            for sign, coeff, variable in terms:
                if variable not in vars_index:
                    raise ValueError(
                        f"Unknown variable '{variable}' in: {equations[row]}"
                    )
                value = float(coeff) if coeff else 1.0
                if sign == "-":
                    value *= -1.0
                A[row, vars_index[variable]] = value
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
