import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.


    ########## Code ends here ##########
    
    ## Get input values
    x_old=xvec[0]
    y_old=xvec[1]
    theta_old=xvec[2]
    V = u[0]
    om = u[1]
    if abs(om)>EPSILON_OMEGA:
        ## Get g
        theta = theta_old + om*dt
        d_sin_theta = np.sin(theta)-np.sin(theta_old)
        d_cos_theta = np.cos(theta)-np.cos(theta_old)
        x = x_old + d_sin_theta*V/om
        y = y_old -d_cos_theta*V/om
        g = np.array([x, y, theta])

        ## Get Gx
        dy_dtheta = V*d_sin_theta/om
        dx_dtheta = V*d_cos_theta/om
        Gx = np.array([[1, 0, dx_dtheta],
                    [0, 1, dy_dtheta],
                    [0, 0, 1]])
        

        ## Get Gu
        dx_dv = d_sin_theta/om
        dy_dv = -d_cos_theta/om
        dx_dom =-V/om**2*d_sin_theta+V/om*(np.cos(theta))*dt
        dy_dom = V/om**2*d_cos_theta+V/om*np.sin(theta)*dt 
        Gu = np.array([[dx_dv, dx_dom],
                    [dy_dv, dy_dom],
                    [0, dt]]) 


    else: #case small omega
        theta=theta_old+ om*dt
        x = x_old + V*np.cos(theta)*dt
        y = y_old +V*np.sin(theta)*dt
        g = np.array([x, y, theta])
        ## Get Gx
        dy_dtheta = V*np.cos(theta)*dt
        dx_dtheta = -V*np.sin(theta)*dt
        Gx = np.array([[1, 0, dx_dtheta],
                    [0, 1, dy_dtheta],
                    [0, 0, 1]])
        ## Get Gu
        dx_dv = np.cos(theta)*dt
        dy_dv = np.sin(theta)*dt
        dx_dom = -V*np.sin(theta)*dt**2/2
        dy_dom =V*np.cos(theta)*dt**2/2
        Gu = np.array([[dx_dv, dx_dom],
                    [dy_dv, dy_dom],
                    [0, dt]]) 




    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)


    ########## Code ends here ##########
    x_base=x[0]
    y_base=x[1]
    theta_base=x[2]
    x_cam_b=tf_base_to_camera[0]
    y_cam_b=tf_base_to_camera[1]
    R=np.array([[np.cos(theta_base),-np.sin(theta_base)],[np.sin(theta_base),np.cos(theta_base)]])
    transformed_cam=np.matmul(R,tf_base_to_camera[0:2])
    total_cam=x[0:2]+transformed_cam
    #coordinates of the camara in world frame
    x_cam=total_cam[0]
    y_cam=total_cam[1]
    th_cam=theta_base+tf_base_to_camera[2] 
    
    
    #using formulas of SNS page 241
    alpha_in_cam=alpha-th_cam
    r_in_cam=r-(x_cam*np.cos(alpha)+y_cam*np.sin(alpha))
    h=np.array([alpha_in_cam,r_in_cam])

    #Hx SNS page 242
    sin_th_b = np.sin(theta_base)
    cos_th_b = np.cos(theta_base)
    drdth=-np.cos(alpha)*(-sin_th_b*x_cam_b - y_cam_b*cos_th_b) - np.sin(alpha)*(x_cam_b*cos_th_b - sin_th_b*y_cam_b)
    Hx=np.array([[0,0,-1],[-np.cos(alpha),-np.sin(alpha),drdth]])



    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
