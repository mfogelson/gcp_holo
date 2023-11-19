from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import concurrent.futures
import time

import numpy as np 

from copy import deepcopy


def normalize_curve(input, scale=None, shift=None):
    """normalizes input curve with shift and scale factors

    Args:
        input (Nd.Array): Shape: (n, 2)
        scale (float, optional): scaling factor. Defaults to None.
        shift (Nd.Array, optional): shift factor [x offset, y offset]. Defaults to None.

    Returns:
        Nd.Array: updated curve
    """
    if input.shape[0] != 2:
        input = input.T
        
    mu = np.mean(input, axis=1).reshape(-1,1)
    std = max(np.std(input, axis=1).reshape(-1,1))


    output = (input - mu) # Shift

    output /= (std+1e-10) # Scale
    
    if scale is not None:
        output*=scale
    
    if shift is not None: 
        output+=shift

    return output

def rotate_points(points, theta):
    """Rotate points by theta radians."""
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    return np.dot(points, rotation_matrix.T)

def sum_of_squared_residuals(curve_i, curve_j, theta):
    """Calculate the sum of squared residuals between rotated curve_i and curve_j."""
    rotated_curve_i = rotate_points(curve_i, theta)
    residuals = curve_j - rotated_curve_i
    return np.sum(residuals**2)

def gauss_newton(curve_i, curve_j, initial_theta=0, iterations=100, alpha=0.4, beta=0.5):
    """Gauss-Newton method to find theta that minimizes the distance."""
    theta = initial_theta

    for _ in range(iterations):
        # Calculate the Jacobian (derivative of residuals w.r.t theta)
        epsilon = 1e-7  # Small value to approximate the derivative
        jacobian = (sum_of_squared_residuals(curve_i, curve_j, theta + epsilon) - 
                    sum_of_squared_residuals(curve_i, curve_j, theta - epsilon)) / (2 * epsilon)

        # Backtracking line search
        step_size = 1.0
        while sum_of_squared_residuals(curve_i, curve_j, theta - step_size * jacobian) > sum_of_squared_residuals(curve_i, curve_j, theta) - alpha * step_size * jacobian**2:
            step_size *= beta

        # Update theta
        theta -= step_size * jacobian

        # Optional: Add convergence check here
        if np.abs(jacobian) < 1e-5 or np.abs(step_size * jacobian) < 1e-5:
            break

    return theta, sum_of_squared_residuals(curve_i, curve_j, theta)

def process_variant(args):
    index_variant, curve_i, curve_j = args
    local_best_obj = np.inf
    local_best_theta = None
    local_best_index_variant = None

    # for initial_theta in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
    optimal_theta, obj = gauss_newton(curve_i[index_variant], curve_j, initial_theta=np.pi/2.0, iterations=100, alpha=0.4, beta=0.5)
    if obj < local_best_obj:
        local_best_obj = obj
        local_best_theta = optimal_theta
        local_best_index_variant = index_variant

    return local_best_obj, local_best_theta, local_best_index_variant

# Parallel method
def parallel_method(index_variants, curve_i, curve_j):
    # Ensure the curves are in the correct shape
    if curve_i.shape[1] != 2:
        curve_i = curve_i.T
    if curve_j.shape[1] != 2:
        curve_j = curve_j.T
        
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        # results = executor.map(process_variant, [(index_variant, curve_i, curve_j) for index_variant in index_variants])
    # for index_variant in index_variants:
        # yield process_variant((index_variant, curve_i, curve_j))
    results = [process_variant((index_variant, curve_i, curve_j)) for index_variant in index_variants]
    
    best_obj = np.inf
    best_theta = None
    best_index_variant = None

    for result in results:
        obj, theta, index_variant = result
        if obj < best_obj:
            best_obj = obj
            best_theta = theta
            best_index_variant = index_variant

    return best_obj, best_theta, best_index_variant

def distance(curve_i, curve_j, ordered=False, distance_metric='euclidean'):
    """distance between curve_i and curve_j

    Args:
        curve_i (Nd.Array): Array of x,y points that is from curve_i Shape: (n, 2)
        curve_j (Nd.Array): Array of x,y points that is from curve_j Shape: (n, 2)
        ordered (bool, optional): if ordering of points matters. Defaults to False.
        distance_metric (str, optional): check scipy.cdist for various options. Defaults to 'euclidean'.

    Returns:
        Nd.Array: Distance between all points
    """

    ## Correct Shape
    if curve_i.shape[1] != 2:
        curve_i = curve_i.T
        
    if curve_j.shape[1] != 2:
        curve_j = curve_j.T 
    
    ## Get distance between all sets of points
    C = cdist(curve_i, curve_j, metric=distance_metric)
    
    row_ind = np.arange(curve_i.shape[0])
    
    if ordered:
        row_inds = row_ind[row_ind[:,None]-np.zeros_like(row_ind)].T
        col_inds = row_ind[row_ind[:,None]-row_ind].T 
        
        min_clock_wise = np.amin(C[row_inds, col_inds].sum(1))
        argmin_clock_wise = np.argmin(C[row_inds, col_inds].sum(1))
        
        min_count_clock_wise = np.amin(C[row_inds, col_inds[:,::-1]].sum(1))
        argmin_count_clock_wise = np.argmin(C[row_inds, col_inds[:,::-1]].sum(1))
        
        ## Check both directions of ordering
        cw_dir = int(min_clock_wise < min_count_clock_wise)

        col_ind = col_inds[int(cw_dir * argmin_clock_wise + (1-cw_dir)*argmin_count_clock_wise), :]
        # col_ind = col_inds[np.argmin(C[row_inds, col_inds].sum(1)), :]
    else:   
        row_ind, col_ind = linear_sum_assignment(C)
            
    return C[row_ind, col_ind]


def uniquify(path):
    import os
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def circIntersectionVect(jointA, jointB, jointC_pos):
    """Circle Intersection Method for linkage FK 

    Args:
        jointA (Nd.Array): jointA path
        jointB (Nd.Array): jointB path
        jointC_pos (Nd.Array): New joint being added

    Returns:
        Nd.Array: Path of jointC
    """
    _, N = jointA.shape
    lengthA = np.linalg.norm(jointA[:,0] - jointC_pos)
    lengthB = np.linalg.norm(jointB[:,0] - jointC_pos)

    d = np.linalg.norm(jointB-jointA, axis=0).reshape(1,N) # (1, N)

    a = np.divide(lengthA**2 - lengthB**2 + np.power(d,2), 2.0*d)

    
    h = np.sqrt(lengthA**2-np.power(a,2))

    P2 = jointA + np.divide(np.multiply(a, (jointB - jointA)),d)


    sol1x = P2[0,:] + np.divide(np.multiply(h, (jointB[1,:].reshape(1,N)- jointA[1,:].reshape(1,N))), d) 
    sol1y = P2[1,:] - np.divide(np.multiply(h, (jointB[0,:].reshape(1,N)- jointA[0,:].reshape(1,N))), d)

    sol1 = np.vstack([sol1x, sol1y])

    sol2x = P2[0,:] - np.divide(np.multiply(h, (jointB[1,:].reshape(1,N)- jointA[1,:].reshape(1,N))), d) 
    sol2y = P2[1,:] + np.divide(np.multiply(h, (jointB[0,:].reshape(1,N)- jointA[0,:].reshape(1,N))), d)

    sol2 = np.vstack([sol2x, sol2y])
    
    if np.linalg.norm(sol1[:,0]-jointC_pos)<1e-4:
        return sol1
    elif np.linalg.norm(sol2[:,0]-jointC_pos)<1e-4:
        return sol2
    else:
        print(f"Error: Neither solution fit initial position, sol1 : {sol1[:,0]}, sol2: {sol2[:,0]}, orig: {jointC_pos} ")
        sol1[:] = np.nan
        return sol1
        # pdb.set_trace()

    # return sol1, sol2

def symbolic_kinematics(xi, xj, xk0):
    """Symbolic Kinematics implementation

    Args:
        xi (Nd.Array): Path of revolute joint xi Shape: (2, n)
        xj (Nd.Array): Path of revolute joint xj Shape: (2, n)
        xk0 (Nd.Array): Initial position of new point xk Shape: (2,)

    Returns:
        Nd.Array: Path of revolute joint xk Shape: (2,n)
    """

    _, N = xi.shape
    l_ij = np.linalg.norm(xj-xi, axis=0).squeeze() # (N, )
    l_ik = np.linalg.norm(xi[:,0] - xk0) # float
    l_jk = np.linalg.norm(xj[:, 0] - xk0) # float

    ## Triangle inequality ##
    valid = np.logical_and.reduce((np.all(l_ik+l_jk > l_ij), 
                                np.all(l_ik+l_ij > l_jk),  
                                np.all(l_jk+l_ij > l_ik), 
                                np.all(l_ij > 0), 
                                np.all(l_ik > 0)))  
    
    if not valid:
        return np.full_like(xi, np.nan)

    f = l_ik / l_ij # (N, )

    t = (l_ij**2 + (l_ik**2 - l_jk**2))/(2*l_ij*l_ik) # (N, )
    if any(1.-t**2 < 0.0):
        return np.full_like(xi, np.nan)
    
    R = np.array([[t, -np.sqrt(1.-t**2)], [np.sqrt(1.-t**2), t]]) # (2, 2, N)
    Q = (R*f).T # (N, 2, 2)

    diff = xj-xi # ()
    diff = diff[np.newaxis, :,:].T
    xk = np.matmul(Q,diff).T.squeeze() + xi
    
    ## found solution path
    if np.linalg.norm(xk[:,0]-xk0) < 1e-3:
        return xk
    
    ## flip orientation
    R = np.array([[t, np.sqrt(1.-t**2)], [-np.sqrt(1.-t**2), t]]) # (2, 2, N)
    Q = (R*f).T # (N, 2, 2)
    xk = np.matmul(Q,diff).T.squeeze() + xi
    
    ## found solution path
    if np.linalg.norm(xk[:,0]-xk0) < 1e-3:
        return xk
    
    ## Passes through singularity 
    return np.full_like(xi, np.nan)

    