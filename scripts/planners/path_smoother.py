import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    from scipy.interpolate import splrep, splev 
    #extract x,y from path
    path_array=np.array(path)
    x=np.array(path_array[:,0])
    y=np.array(path_array[:,1])
    #the independent varibale for spl will be the time to complte all trajectory at fixed spped
    #t=d/V
    total_distance=0
    for i in range(1,len(x)):
        total_distance+=np.linalg.norm(np.array([x[i]-x[i-1],y[i]-y[i-1]]))
    total_time=total_distance/V_des
    time_array = np.linspace(0, total_time, len(x))
    #B-spline line representation splrep(x,y). k=3 cubic
    #path_array=np.asarray(path,np.dtype('float,float'))
    spl_y= splrep(time_array,y,k=3,s=alpha)
    spl_x= splrep(time_array,x,k=3,s=alpha)

    #new array of time with more points to create smooth curve
    t_smoothed = np.linspace(0, total_time, round(total_time/dt,0))
    #get x,y
    x2 = splev(t_smoothed, spl_x)
    y2 = splev(t_smoothed, spl_y)
    #get dx, dy
    dx2 = splev(t_smoothed, spl_x,der=1)
    dy2 = splev(t_smoothed, spl_y,der=1)
    #get ddx, ddy
    ddy2 = splev(t_smoothed, spl_x,der=2)
    ddx2 = splev(t_smoothed, spl_y,der=2)

    #theta is atan(dy/dx)
    theta=np.arctan2(dy2,dx2)

    traj_smoothed=np.column_stack((x2,y2,theta,dx2,dy2,ddx2,ddy2))

    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
