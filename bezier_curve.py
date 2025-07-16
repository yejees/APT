import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def nonlinear_transformation(slices):

    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    # points_1 = [[0, 0], [0, 0], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1

    return slices, nonlinear_slices_1

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation_rand(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def nonlinear_transformation_multi(x, num_images=10):
    images = []
    
    # Define the start and end points for interpolation
    start_points = np.array([[0, 0], [0.0, 0.0], [1.0, 1.0], [1, 1]], dtype=np.float32)
    end_points = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=np.float32)
    mid_points = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    
    start_points = np.array([[-1, -1], [-1, -1], [1.0, 1.0], [1, 1]], dtype=np.float32)
    end_points = np.array([[1, 1], [1, 1], [-1, -1], [-1, -1]], dtype=np.float32)
    mid_points = np.array([[0, 0], [0.0, 0.0],  [0, 0], [0.0, 0.0]], dtype=np.float32)
    
    # Generate interpolated points between start and end points
    for i in range(num_images):
        # if i <num_images//2:
        #     pass
        
        t = (i%(num_images//2)) / (num_images//2 - 1)  # linear interpolation factor between 0 and 1
        if i <num_images//2:
            interpolated_points  = start_points.copy()
        else:
            interpolated_points = end_points.copy()

        interpolated_points[1:3] = (1 - t) * start_points[1:3] + t * end_points[1:3]  # interpolation
        if i <num_images//2:
            interpolated_points[0:1] = (1 - t) * interpolated_points[0:1] + t * mid_points[0:1] 
            interpolated_points[3:4] = (1 - t) *  interpolated_points[3:4] + t * mid_points[3:4]
        else:
            interpolated_points[0:1] = t * interpolated_points[0:1] + (1-t) * mid_points[0:1] 
            interpolated_points[3:4] = t *  interpolated_points[3:4] + (1-t) * mid_points[3:4]

        
        # Generate the Bezier curve based on the interpolated control points
        xvals, yvals = bezier_curve(interpolated_points, nTimes=100000)
        
        # Sort xvals for interpolation and apply the nonlinear transformation
        xvals = np.sort(xvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        
        images.append(nonlinear_x)
    
    return images