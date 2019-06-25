import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.transform import Rotation
from scipy.stats import zscore
from sklearn import linear_model
import matplotlib.pyplot as plt


def ransac_plane_estimation(points, k=10):
    """
    Estimate plane that fits the input points best, using RANSAC.
        Inputs:
            points: Input points [n,x,y,z].
            k: Number of nearest neighbors to use for residual threshold
               calculation.
        Outputs:
            params: Plane parameters (w0,w1,w2,w3).
    """
    # Find mean k-neighbor distance to use as residual threshold
    kdtree = cKDTree(points)
    distances, _ = kdtree.query(points, k+1)
    mean_distances = np.mean(distances[:,1:])
    # Use ransac to estimate the plate surface parameters
    ransac = linear_model.RANSACRegressor(
        residual_threshold=mean_distances,
        random_state=int(np.random.rand() * 100))
    ransac.fit(points[:,:2], points[:,2:])
    params = (ransac.estimator_.intercept_.tolist() 
              + ransac.estimator_.coef_[0].tolist() + [-1])
    return params


def align_plane_with_axis(plane_params, axis):
    """
    Compute translation vector and rotation matrix to align given plane
    with axis.
        Inputs:
            plane_params: Plane parameters (w0,w1,w2,w3).
            axis: Unit axis to align plane to.
        Outputs:
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
    """
    Statistical outlier filtering of point cloud data.
        Inputs:
            points: Input points [n,x,y,z].
            z_max: Maximum z-score for inliers.
            inlier_ratio: Assumption of min inliers to outliers ratio.
        Outputs:
            inliers: Inlier points in input set.
    """
    # Find max k-neighbor distance to use as distance score
    # where k is determined by the assumed inlier to outlier ratio
    kdtree = cKDTree(points)
    k = inlier_ratio * points.shape[0]
    distances, _ = kdtree.query(points, k)
    z_scores = zscore(np.max(distances, axis=1))
    # Filter out points outside given z-score range
    sor_filter = np.abs(z_scores) < z_max
    inliers = points[sor_filter]
    return inliers 


def estimate_volume(points):
    """
    Estimate point cloud volume using z-axis as height.
        Inputs:
            points: Volume-defining points.
        Outputs:
            total_volume: Estimated volume.
            simplices: Triangulation resulting vertices.
    """
    # Generate triangulation of x-y plane
    tri = Delaunay(points[:,:2])
    tri_vertices = points[tri.simplices, :]
    # Compute the area and mean height of each triangle
    side_a = tri_vertices[:,1,:2] - tri_vertices[:,0,:2]
    side_b = tri_vertices[:,2,:2] - tri_vertices[:,0,:2]
    area = 0.5 * np.abs(np.cross(side_a, side_b, axis=1))
    mean_height = np.mean(tri_vertices[:,:,2], axis=-1)
    # Estimate volume as the sum of all triangulated region volumes
    volumes = area * mean_height
    total_volume = np.abs(np.sum(volumes))
    simplices = tri.simplices.copy()
    return total_volume, simplices


def pretty_plotting(imgs, tiling, titles):
    """
    Plot images in a pretty grid.
        Inputs:
            imgs: List of images to plot.
            tiling: Subplot tiling tuple (rows,cols).
            titles: List of subplot titles.
    """
    n_plots = len(imgs)
    rows = str(tiling[0])
    cols = str(tiling[1])
    plt.figure()
    for r in range(tiling[0] * tiling[1]):
        plt.subplot(rows + cols + str(r + 1))
        plt.title(titles[r])
        plt.imshow(imgs[r])
