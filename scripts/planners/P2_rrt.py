import numpy as np
import matplotlib.pyplot as plt
#from utils import plot_line_segments, line_line_intersection

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    def solve(self, eps=0.2, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters, state_dim))
        V[0,:] = self.x_init    # RRT is rooted at self.x_init
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters, dtype=int)

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        ## Hints:
        #   - use the helper functions find_nearest, steer_towards, and is_free_motion
        #   - remember that V and P always contain max_iters elements, but only the first n
        #     are meaningful! keep this in mind when using the helper functions!

        ########## Code starts here ##########
        #for loop up to iteration limit
        for k in range (1,max_iters):
            #sample random value between 0 and 1, to determine if we are going in direction to goal or random point
            z=np.random.uniform(0,1)
            if z<goal_bias:
                x_rand=self.x_goal
            else:
                x_rand=self.random_state()
            #find the nearest neighbor of that point
            x_near_V_index=self.find_nearest(V[0:n,:],x_rand)
            x_near=V[x_near_V_index]
            #check the point the algorithm will move that is close to x_near, in direction to x_rand
            x_new=self.steer_towards(x_near,x_rand,eps)
            #check if path between x_new and x_near is collision free
            if self.is_free_motion(self.obstacles,x_near,x_new):
                #add vertex(V) and edge(P) to tree
                V[n]=x_new
                P[n]=x_near_V_index
                n+=1
                #check if we reached the goal, complete the path and break
                if np.array_equal(x_new,self.x_goal):
                    self.reconstruct_path(V[0:n,:],P[0:n])
                    success=True
                    break


        
        ########## Code ends here ##########

        # plt.figure()
        # self.plot_problem()
        # self.plot_tree(V, P, color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
        # if success:
        #     if shortcut:
        #         self.plot_path(color="purple", linewidth=2, label="Original solution path")
        #         self.shortcut_path()
        #         self.plot_path(color="green", linewidth=2, label="Shortcut solution path")
        #     else:
        #         self.plot_path(color="green", linewidth=2, label="Solution path")
        #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        #     plt.scatter(V[:n,0], V[:n,1])
        # else:
        #     print "Solution not found!"

        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        success=False
        while not success:
            success=True
            i=1 #START IN 1 TO AVOID CONSIDERING X_INIT
            while i< len(self.path)-1:
                x=self.path[i]
                parent=self.path[i-1]
                child=self.path[i+1]
                if self.is_free_motion(self.obstacles,parent,child):
                    self.path=[ node for node in self.path if not (node==x).all()]     
                    success=False
                else:
                    i+=1
            break
        
         

        
        ########## Code ends here ##########
    
    def random_state(self):
        """
        function that return a random configuration that belong to the state space
        """
        dif=self.statespace_hi-self.statespace_lo
        sol=np.zeros(len(dif))
        for i in range(len(dif)):
            rand=np.random.uniform(0,1)
            sol[i]=dif[i]*rand
        return self.statespace_lo+sol

    def reconstruct_path(self,V,P):
        """
        Complete path.
        Go from x_goal to X_init
        """
        n=len(P)-1
        path = [V[n]] #this should be the same as the goal
        parent_index=P[n]
        parent=V[parent_index]
        while not np.array_equal(parent , self.x_init):
            path.append(parent)
            parent_index=P[parent_index]
            parent=V[parent_index]
        path.append(self.x_init)
        self.path= list(reversed(path))

class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        ########## Code starts here ##########
        # Hint: This should take one line.
        return np.argmin(np.linalg.norm(V-x,axis=1))

        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (straigh line). 
        Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        ########## Code starts here ##########
        # Hint: This should take one line.
        d= x2-x1
        norm=np.linalg.norm(d)
        if norm<eps:
            return x2
        else:
            return x1+d/norm*eps
        
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2):
        # motion = np.array([x1, x2])
        # for line in obstacles:
        #     if line_line_intersection(motion, line):
        #         return False
        # return True
        """
        Function that check all element sin the grid between 2 points in a straight line
        return false if finds an obstacle
        else true
        """
        x1=np.array([x1[0],x1[1]])
        x2=np.array([x2[0],x2[1]])
        print("start {} end {}".format(x1,x2) )
        #for loop to check moving resolution, p2 a point dx from x1
        dx=obstacles.resolution
        p2_old=x1 #point to keep track where we are
        goal_distance=np.linalg.norm(x1-x2)
        p2_distance=np.linalg.norm(p2_old-x1)
        m=(x2[1]-x1[1])/(x2[0]-x1[0]) #slope
        #stop while if we are farther that the point we are trying to connect
        while p2_distance<goal_distance:
            p2_x=p2_old[0]+dx
            p2_y=m*(p2_x-p2_old[0])+p2_old[1]
            p2=np.array([p2_x,p2_y])
            #print("p2",p2)
            new_p2=obstacles.snap_to_grid(p2)
            is_free=obstacles.is_free(new_p2)#check if that element in the gree has an obstacle
            if not is_free:
                print("NOT free point {}".format(new_p2))
                return False
            else:#update for next iteration
                print("free point {}".format(new_p2))
                goal_distance=np.linalg.norm(x1-x2)
                p2_distance=np.linalg.norm(new_p2-x1)
                p2_old=new_p2


        return True


    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DubinsRRT(RRT):
    """
    Represents a planning problem for the Dubins car, a model of a simple
    car that moves at a constant speed forward and has a limited turning
    radius. We will use this v0.9.2 of the package at
    https://github.com/AndrewWalker/pydubins/blob/0.9.2/dubins/dubins.pyx
    to compute steering distances and steering trajectories. In particular,
    note the functions dubins.path_length and dubins.path_sample (read
    their documentation at the link above). See
    http://planning.cs.uiuc.edu/node821.html
    for more details on how these steering trajectories are derived.
    """
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles, turning_radius):
        self.turning_radius = turning_radius
        super(self.__class__, self).__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def find_nearest(self, V, x):
        from dubins import path_length
        ########## Code starts here ##########
        #check all point in V, save the one with the smaller path lenght
        point_index=None
        distance=None
        for i in range(len(V)):
            sample=V[i,:]
            dubins_distance=path_length(sample,x,self.turning_radius)
            if distance is None or distance>dubins_distance:
                distance=dubins_distance
                point_index=i

        return point_index
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        ########## Code starts here ##########
        """
        A subtle issue: if you use dubins.path_sample to return the point
        at distance eps along the path from x to y, use a turning radius
        slightly larger than self.turning_radius
        (i.e., 1.001*self.turning_radius). Without this hack,
        dubins.path_sample might return a point that can't quite get to in
        distance eps (using self.turning_radius) due to numerical precision
        issues.
        """
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        from dubins import path_sample, path_length
        dubins_distance=path_length(x1,x2,1.001*self.turning_radius)
        if dubins_distance<eps:
            return x2
        else:
            dubinspath= path_sample(x1,x2,1.001*self.turning_radius,eps)
            return dubinspath[0][1]

        
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/6):
        from dubins import path_sample
        pts = path_sample(x1, x2, self.turning_radius, self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution = np.pi/24, **kwargs):
        from dubins import path_sample
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[P[i],:], V[i,:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[i,:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution = np.pi/24, **kwargs):
        from dubins import path_sample
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            pts.extend(path_sample(path[i], path[i+1], self.turning_radius, self.turning_radius*resolution)[0])
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)