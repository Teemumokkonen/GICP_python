import math
import numpy as np
import open3d as o3d
import sys
from .utils import skewd, se3_exp
from scipy.linalg import cholesky, solve
from concurrent.futures import ThreadPoolExecutor, as_completed

lm_init_lambda_factor_ = 1e-9

def linearize_point(i, x, source_points, correspondences, source_cov, target_points, target_cov, mahalanobis):
    target_idx = correspondences[i]
    if target_idx < 0:
        return None  # Skip if no valid correspondence
    
    a = source_points[i]
    # cov_a = source_cov[i]
    b_mean = target_points[target_idx]
   # cov_b = target_cov[target_idx]
    trans_a = x @ np.append(a, 1)
    error = np.append(b_mean, 1) - trans_a
    error_term = error.T @ mahalanobis[i] @ error
    
    dtdx0 = np.zeros((4, 6))
    dtdx0[:3, :3] = skewd(trans_a[:3])
    dtdx0[:3, 3:] = -np.eye(3)
    
    H_local = dtdx0.T @ mahalanobis[i] @ dtdx0
    b_local = dtdx0.T @ mahalanobis[i] @ error
    
    return error_term, H_local, b_local.T

def update_correspondence_point(i, x, source_points, source_cov, kdtree, corr_dist_threshold, target_cov):
    trans_p = x @ np.append(source_points[i], 1)
    _, idx, d = kdtree.search_knn_vector_3d(trans_p[:3], 1)
    sq_distance = d[0]
    correspondence = idx[0] if d[0] < corr_dist_threshold ** 2 else -1

    if correspondence < 0:
        return sq_distance, correspondence, np.zeros((4, 4))  # No valid correspondence

    # Mahalanobis calculation
    cov_A = source_cov[i]
    cov_B = target_cov[correspondence]
    rcr = cov_B @ x @ cov_A @ x.T
    rcr[3, 3] = 1.0
    mahalanobis = np.linalg.inv(rcr)
    mahalanobis[3, 3] = 0.0

    return sq_distance, correspondence, mahalanobis

class GICP():
    def __init__(self) -> None:
        self.source = None
        self.source_cov = None
        self.target = None
        self.target_cov = None
        self.correspondaces = None
        self.sq_distance = None
        self.mahalanobis = None
        self.kdtree = None
        self.corr_dist_threshold = 400

    def covariance(self, cloud, cov):
        """
        Calculate the cloud distribution covariance
        """
        kdtree = o3d.geometry.KDTreeFlann(cloud)
        xyz = np.asarray(cloud.points)
        #print(cloud.size)
        cov = np.zeros((len(cloud.points), 4, 4))
        k_correspondences = 20
        neighbors = np.zeros((4, k_correspondences))
        for c, p in enumerate(cloud.points):
            [k, idx, _] = kdtree.search_knn_vector_3d(p, k_correspondences)  # Find the 5 nearest neighbors
            

            # add neighbors to array
            for i in range(len(idx)):
                arr = np.append(cloud.points[idx[i]], 1)
                neighbors[:, i] = arr
            
            neighbors_centered = neighbors - neighbors.mean(axis=0)
            cov_temp = neighbors_centered @ neighbors_centered.T / k_correspondences
            values = np.array([1, 1, 1e-3])
            u, s, v = np.linalg.svd(cov_temp[:3, :3])
            cov[c][:3, :3] = u @ np.diag(values) @ v.T     

        return cov
    
    def set_target(self, cloud):
        self.target = cloud
        self.kdtree = o3d.geometry.KDTreeFlann(cloud)

    def set_source(self, cloud):
        self.source = cloud

    def update_correspondances(self, x):
        self.correspondaces = np.zeros(len(self.source.points), dtype=int)
        self.sq_distance = np.zeros(len(self.source.points))
        self.mahalanobis = np.zeros((len(self.source.points), 4, 4))

        for i, p in enumerate(self.source.points):
            trans_p = x @ np.append(p, 1)
            [k, idx, d] = self.kdtree.search_knn_vector_3d(trans_p[:3], 1)
            self.sq_distance[i] = d[0]
            self.correspondaces[i] = idx[0] if d[0] < self.corr_dist_threshold ** 2 else -1

            if self.correspondaces[i] < 0:
                continue

            # not sure about this...
            target_indx = self.correspondaces[i]
            cov_A = self.source_cov[i]
            cov_B = self.target_cov[target_indx]
            rcr = cov_B @ x @ cov_A @ x.T
            rcr[3, 3] = 1.0
            self.mahalanobis[i] = np.linalg.inv(rcr)
            self.mahalanobis[i][3, 3] = 0.0
            #print(self.mahalanobis)

    def linearize(self, x):
        self.update_correspondances(x)
        H = np.zeros((6, 6))
        b = np.zeros((1, 6))
        sum_of_errors = 0.0
        for i, p in enumerate(self.source.points):
            target_idx = self.correspondaces[i]
            if target_idx < 0:
                continue

            a = self.source.points[i]
            cov_a = self.source_cov[i]
            b_mean = self.target.points[target_idx]
            cov_b = self.target_cov[target_idx]
            trans_a = x @ np.append(a, 1)
            error = np.append(b_mean, 1) - trans_a
            sum_of_errors += error.T @ self.mahalanobis[i] @ error
            dtdx0 = np.zeros((4, 6))
            dtdx0[:3, :3] = skewd(trans_a[:3])
            # Fill the next 3x3 block with the negative identity matrix
            dtdx0[:3, 3:] = -np.eye(3)
            H += dtdx0.T @ self.mahalanobis[i] @ dtdx0
            temp = dtdx0.T @ self.mahalanobis[i] @ error
            b += temp.T
                        
        return sum_of_errors, H, b


    def update_correspondances_parallel(self, x):
        self.correspondaces = np.zeros(len(self.source.points), dtype=int)
        self.sq_distance = np.zeros(len(self.source.points))
        self.mahalanobis = np.zeros((len(self.source.points), 4, 4))
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    update_correspondence_point, i, x, self.source.points, self.source_cov, 
                    self.kdtree, self.corr_dist_threshold, self.target_cov
                )
                for i in range(len(self.source.points))
            ]
            
            for i, future in enumerate(as_completed(futures)):
                sq_distance, correspondence, mahalanobis = future.result()
                self.sq_distance[i] = sq_distance
                self.correspondaces[i] = correspondence
                self.mahalanobis[i] = mahalanobis

    def linearize_parallel(self, x):
        self.update_correspondances(x)
        H = np.zeros((6, 6))
        b = np.zeros((1, 6))
        sum_of_errors = 0.0
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    linearize_point, i, x, self.source.points, self.correspondaces,
                    self.source_cov, self.target.points, self.target_cov, self.mahalanobis
                )
                for i in range(len(self.source.points))
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    error_term, H_local, b_local = result
                    sum_of_errors += error_term
                    H += H_local
                    b += b_local

        return sum_of_errors, H, b

    def compute_transformation(self, guess):
        self.source_cov = self.covariance(self.source, self.source_cov)
        self.target_cov = self.covariance(self.target, self.target_cov)
        sum_of_errors = 0.0
        # 50 iterations to solve
        for i in range(50):
            sum_of_errors, H, b = self.linearize_parallel(guess)
            #eigenvalues = np.linalg.eigvals(H)
            if sum_of_errors > 0.000000001:
                #print(eigenvalues)
                epsilon = 1e-7  # Regularization factor
                H_reg = H + np.eye(H.shape[0]) * epsilon
                #L = cholesky(H_reg, lower=True)  # LDLT decomposition: H = L * L.T
                #d = solve(L.T, solve(L, -b))  # First solve L * y = -b, then L.T * d = y
                d = np.linalg.solve(H, -b.T)
                guess = guess @ se3_exp(d)
            else: 
                print("registeration complete")
                break

        return guess
