import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.transform import Rotation
from scipy.stats import zscore
from sklearn import linear_model
import matplotlib.pyplot as plt


def linear_plane_estimation(points):
    """Find the plane that best fits the input points using ordinary 
    least squares.

    Inputs:
        points: Input points Mx3.
    Outputs:
        params: Plane parameters (w0,w1,w2,w3).
    """
    # Use OLS to estimate the plane parameters
    model = linear_model.LinearRegression()
    model.fit(points[:,:2], points[:,2:])
    params = (model.intercept_.tolist() 
              + model.coef_[0].tolist() + [-1])
    return params

def pca_plane_estimation(points):
    """Find the plane that best fits the input poitns using PCA.

    Inputs:
        points: Input points Mx3.
    Returns:
        params: Plane parameters (w0,w1,w2,w3).
    """
    # Fit a linear model to determine the plane normal orientation
    linear_params = linear_plane_estimation(points)
    linear_model_normal = np.array(linear_params[1:])

    # Zero mean the points and compute the covariance
    # matrix eigenvalues/eigenvectors
    point_mean = np.mean(points, axis=0)
    zero_mean_points = points - point_mean
    cov_matrix = np.cov(zero_mean_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort the eigenvectors in descending eigenvalue orded
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Use the least important component's eigenvector as plane normal
    normal = eigenvectors[:,2]
    # Align the PCA plane normal with the linear model plane normal
    if np.dot(normal, linear_model_normal) < 0:
        normal = normal * (-1)
    params = [-np.dot(normal, point_mean), normal[0], normal[1], normal[2]]

    return params

def align_plane_with_axis(plane_params, axis):
    """Compute the translation vector and rotation matrix to align 
    given plane with axis.
    
    Inputs:
        plane_params: Plane parameters (w0,w1,w2,w3).
        axis: Unit axis to align plane to.
    Returns:
        translation_vec: Translation vector.
        rotation_matrix: Rotation matrix.
    """
    plane_params = np.array(plane_params)
    plane_normal = plane_params[1:] / np.sqrt(np.sum(plane_params[1:]**2))
    # Compute translation vector
    d = plane_params[0] / (np.dot(plane_normal, axis) 
                           * np.sqrt(np.sum(plane_params[1:]**2)))
    translation_vec = d * axis
    # Compute rotation matrix
    rot_axis = np.cross(plane_normal, axis)
    rot_axis_norm = rot_axis / np.sqrt(np.sum(rot_axis**2))
    angle = np.arccos(np.dot(plane_normal, axis))
    r = Rotation.from_rotvec(angle * rot_axis_norm)
    rotation_matrix = r.as_dcm()
    return translation_vec, rotation_matrix


def sor_filter(points, z_max=1, inlier_ratio=0.5):
    """Statistical outlier filtering of point cloud data.

    Inputs:
        points: Input points [n,x,y,z].
        z_max: Maximum z-score for inliers.
        inlier_ratio: Assumption of min inliers to outliers ratio.
    Returns:
        inliers: Inlier points in input set.
        sor_mask: Mask of inlier points.
    """
    # Find max k-neighbor distance to use as distance score
    # where k is determined by the assumed inlier to outlier ratio
    kdtree = cKDTree(points)
    k = inlier_ratio * points.shape[0]
    distances, _ = kdtree.query(points, k)
    z_scores = zscore(np.max(distances, axis=1))
    # Filter out points outside given z-score range
    sor_mask = np.abs(z_scores) < z_max
    inliers = points[sor_mask]
    return inliers, sor_mask

def pc_to_volume(points, alpha=0.01):
    """Compute the volume of a point cloud using the alpha shape.
    Modified from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

    Inputs:
        points: Mx3 array of points.
        alpha: Alpha value.
    Returns:
        total_volume: Volume defined by the point cloud surface.
        alpha_simplices_array: Alpha shape simplices.
    """
    # Triangulate using Delaunay triangulation
    tri = Delaunay(points[:,:2])
    alpha_simplices = set()
    total_volume = 0

    # Loop over created triangles and keep only those with radius < alpha
    for simplex in tri.simplices:
        v1 = points[simplex[0], :2]
        v2 = points[simplex[1], :2]
        v3 = points[simplex[2], :2]

        # Lengths of sides
        a = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
        b = np.sqrt((v2[0] - v3[0]) ** 2 + (v2[1] - v3[1]) ** 2)
        c = np.sqrt((v3[0] - v1[0]) ** 2 + (v3[1] - v1[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's Formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # Radius of circle circumscribed about  the triangle
        circum_r = a * b * c / (4.0 * area)

        if circum_r < alpha:
            # Add volume defined by this triangle to the total volume
            total_volume += area * ((points[simplex[0], 2] + points[simplex[1], 2]
                               + points[simplex[2], 2]) / 3)
            alpha_simplices.add((simplex[0], simplex[1], simplex[2]))

    alpha_simplices_array = np.array(list(alpha_simplices))
    return total_volume, alpha_simplices_array

def pretty_plotting(imgs, tiling, titles, suptitle=None):
    """Plot images in a pretty grid.

    Inputs:
        imgs: List of images to plot.
        tiling: Subplot tiling tuple (rows,cols).
        titles: List of subplot titles.
        suptitle: Suptitle above all plots.
    """
    n_plots = len(imgs)
    rows = str(tiling[0])
    cols = str(tiling[1])
    fig = plt.figure()
    for r in range(tiling[0] * tiling[1]):
        plt.subplot(rows + cols + str(r + 1))
        plt.title(titles[r])
        plt.imshow(imgs[r])
        if ('Depth' in titles[r]) or ('depth' in titles[r]):
            plt.colorbar()

    if suptitle is not None:
        fig.suptitle(suptitle)

