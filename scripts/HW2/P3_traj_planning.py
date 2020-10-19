import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        #check if we are in time to change controller. 
        #final time
        tf=self.traj_controller.traj_times[-1]
        if t<tf-self.t_before_switch: #we are in trayectory control
            V,om=self.traj_controller.compute_control( x, y, th, t)
        else: #position control
            V,om=self.pose_controller.compute_control( x, y, th, t)

        
        return V,om
        ########## Code ends here ##########

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
        traj_smoothed (np.array [N,7]): Smoothed trajectory (x,y,th,dx,dy,ddx,ddy)
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

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    #first we need to know which imputs are capable to follow the original trajectory
    V, om =compute_controls(traj)

    #now, lets make sure the inputs are inside the max values
    s = compute_arc_length(V, t) #compute arc lenght as a function of time
    V_tilde = rescale_V(V, om, V_max, om_max) #Reescale V input
    tau = compute_tau(V_tilde, s) #compute new time sequency.
    om_tilde = rescale_om(V, om, V_tilde)# rescale om input

    #now we cant to rewrite time and inputs dt time apart
    n=len(traj[:,0])-1
    s_f = State(x=traj[n,0], y=traj[n,1], V=V_tilde[n], th=traj[n,2]) #final state
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
