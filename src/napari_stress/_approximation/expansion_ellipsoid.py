import numpy as np
from .expansion_base import Expander

class EllipsoidExpander(Expander):
    """
    Expand a set of points to fit an ellipsoid using least squares fitting.

    The ellipsoid equation is of the form:

    .. math::
        Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

    where A, B, C, D, E, F, G, H, I are the coefficients of the ellipsoid equation and
    x, y, z are the coordinates of the points. The parameters of this equation are
    fitted to the input points using least squares fitting.

    Methods
    -------
    fit(points: "napari.types.PointsData")
        Fit an ellipsoid to a set of points using leaast square fitting.

    expand(points: "napari.types.PointsData")
        Project a set of points onto their respective position on the fitted ellipsoid.

    fit_expand(points: "napari.types.PointsData")
        Fit an ellipsoid to a set of points and then expand them.

    Examples
    --------

    .. code-block:: python

        # Instantiate and fit an ellipsoid expander to a set of points
        expander  = EllipsoidExpander()
        expander.fit(points)

        # Expand the points on the fitted ellipsoid
        fitted_points = expander.expand(points)

    """

    def __init__(self):
        super().__init__()

    def _fit(
        self, points: "napari.types.PointsData"
    ) -> "napari.types.VectorsData":
        """
        Fit a 3D ellipsoid to given points using least squares fitting.

        The ellipsoid equation is: Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit an ellipsoid to.

        Returns
        -------
        ellipsoid_fitted_ : napari.types.VectorsData
            The fitted ellipsoid.
        """
        coefficients = self._fit_ellipsoid_to_points(points)
        self._center, self._axes, self._eigenvectors = (
            self._extract_characteristics(coefficients)
        )

        vectors = (
            self._eigenvectors
            / np.linalg.norm(self._eigenvectors, axis=1)[:, np.newaxis]
        ).T * self._axes[:, None]
        base = np.stack([self._center] * 3)
        ellipsoid_fitted_ = np.stack([base, vectors], axis=1)

        return ellipsoid_fitted_

    def _expand(self, points: "napari.types.PointsData"):
        """
        Expand a set of points to fit an ellipsoid.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to expand.

        Returns
        -------
        expanded_points : napari.types.PointsData
            The expanded points.
        """
        from .._utils.coordinate_conversion import (
            cartesian_to_elliptical,
            elliptical_to_cartesian,
        )

        U, V = cartesian_to_elliptical(self._coefficients, points, invert=True)
        expanded_points = elliptical_to_cartesian(
            U, V, self._coefficients, invert=True
        )

        return expanded_points

    def _calculate_properties(self, input_points, output_points):
        """
        Measure properties of the expansion.
        """
        self._measure_residuals(input_points, output_points)
        self._measure_max_min_curvatures()

    @property
    def coefficients_(self):
        """
        The coefficients of the fitted ellipsoid

        Returns
        -------
        coefficients_ : napari.types.VectorsData
            The coefficients of the ellipsoid equation. The coefficients are of the form
            (3, 2, 3); The first dimension represents the three axes of the ellipsoid
            (major, medial and minor). The second dimension represents the components of
            the ellipsoid vectors (base point and direction vector). The third dimension
            represents the dimension of the space (z, y, x).
        """
        return super().coefficients_

    @coefficients_.setter
    def coefficients_(self, value: "napari.types.VectorsData"):
        """
        value: (3, 2, D) matrix representing the ellipsoid coefficients.
        """
        if value is not None:
            self._center = value[0, 0]
            self._axes = np.linalg.norm(value[:, 1], axis=1)
            self._coefficients = value

    @property
    def axes_(self):
        """
        The lengths of the axes of the ellipsoid.

        Returns
        -------
        axes_ : np.ndarray
            The lengths of the axes of the ellipsoid.
        """
        return self._axes

    @property
    def center_(self):
        """
        The center of the ellipsoid.

        Returns
        -------
        center_ : np.ndarray
            The center of the ellipsoid.
        """
        return self._center

    @property
    def properties(self):
        """
        Get properties of the expansion.

        Returns
        -------
        properties : dict
            Dictionary containing properties of the expansion with following keys:

            - residuals: np.ndarray
                Residual euclidian distance between input points and expanded points.
            - maximum_mean_curvature: float
                Maximum mean curvature of the ellipsoid.
            - minimum_mean_curvature: float
                Minimum mean curvature of the ellipsoid.

            The maximum and minimum curvatures :math:`H_{max}` and :math:`H_{min}` are calculated as follows:

            .. math::
                H_{max} = a / (2 * c^2) + a / (2 * b^2)

                H_{min} = c / (2 * b^2) + c / (2 * a^2)

            where a, b and c are the lengths of the ellipsoid axes along the three spatial dimensions.
        """
        return self._properties

    def _measure_max_min_curvatures(self):
        """
        Measure maximum and minimum curvatures of the ellipsoid.

        Returns
        -------
        None

        """
        # get and remove the largest, smallest and medial axis
        semi_axis_sorted = np.sort(self._axes)
        a = semi_axis_sorted[2]
        b = semi_axis_sorted[1]
        c = semi_axis_sorted[0]

        # accoording to paper (https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1.full)
        maximum_mean_curvature = a / (2 * c**2) + a / (2 * b**2)
        minimum_mean_curvature = c / (2 * b**2) + c / (2 * a**2)

        self._properties["maximum_mean_curvature"] = maximum_mean_curvature
        self._properties["minimum_mean_curvature"] = minimum_mean_curvature

    def _measure_residuals(self, input_points, output_points):
        """
        Measure residuals of the expansion.

        Parameters
        ----------
        input_points : napari.types.PointsData
            The points before expansion.
        output_points : napari.types.PointsData
            The points after expansion.
        """
        output_points = self._expand(input_points)

        distance = np.linalg.norm(input_points - output_points, axis=1)
        self._properties["residuals"] = distance

    def _fit_ellipsoid_to_points(
        self,
        points: "napari.types.PointsData",
    ) -> np.ndarray:
        """
        Fit an ellipsoid to a set of points.

        The used equation is of the form:
        x^2 / a^2 + y^2 / b^2 + z^2 / c^2 + 2xy / ab + 2xz / ac + 2yz / bc + 2dx / a + 2ey / b + 2fz / c = 1

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit an ellipsoid to.

        Returns
        -------
        ellipsoid_coefficients : np.ndarray
            The coefficients of the ellipsoid equation.
        """
        # Extract x, y, z coordinates from points and reshape to column vectors
        x = points[:, 0, np.newaxis]
        y = points[:, 1, np.newaxis]
        z = points[:, 2, np.newaxis]

        # Construct the design matrix for the ellipsoid equation
        design_matrix = np.hstack(
            (x**2, y**2, z**2, x * y, x * z, y * z, x, y, z)
        )
        column_of_ones = np.ones_like(x)  # Column vector of ones

        # Perform least squares fitting to solve for the coefficients
        transposed_matrix = design_matrix.transpose()
        matrix_product = np.dot(transposed_matrix, design_matrix)
        inverse_matrix = np.linalg.inv(matrix_product)
        coefficients = np.dot(
            inverse_matrix, np.dot(transposed_matrix, column_of_ones)
        )

        # Append -1 to the coefficients to represent the constant term on the right side of the equation
        ellipsoid_coefficients = np.append(coefficients, -1)

        return ellipsoid_coefficients

    def _extract_characteristics(self, coefficients: np.ndarray):
        # Construct the augmented matrix from the coefficients
        Amat = np.array(
            [
                [
                    coefficients[0],
                    coefficients[3] / 2.0,
                    coefficients[4] / 2.0,
                    coefficients[6] / 2.0,
                ],
                [
                    coefficients[3] / 2.0,
                    coefficients[1],
                    coefficients[5] / 2.0,
                    coefficients[7] / 2.0,
                ],
                [
                    coefficients[4] / 2.0,
                    coefficients[5] / 2.0,
                    coefficients[2],
                    coefficients[8] / 2.0,
                ],
                [
                    coefficients[6] / 2.0,
                    coefficients[7] / 2.0,
                    coefficients[8] / 2.0,
                    coefficients[9],
                ],
            ]
        )

        # Extract the quadratic part and find its inverse
        A3 = Amat[:3, :3]
        A3inv = np.linalg.inv(A3)

        # Compute the center of the ellipsoid
        ofs = coefficients[6:9] / 2.0
        center = -np.dot(A3inv, ofs)

        # Transform the matrix to center the ellipsoid at the origin
        Tofs = np.eye(4)
        Tofs[3, :3] = center
        R = np.dot(Tofs, np.dot(Amat, Tofs.T))

        # Extract the transformed quadratic part
        R3 = R[:3, :3]

        # Perform eigendecomposition to find axes and orientation
        eigenvalues, eigenvectors = np.linalg.eig(R3 / -R[3, 3])

        # Compute the lengths of the axes
        axes_lengths = np.sqrt(1.0 / np.abs(eigenvalues))

        return center, axes_lengths, eigenvectors
