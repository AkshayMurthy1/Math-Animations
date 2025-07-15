import numpy as np
from scipy.integrate import solve_ivp
from manim import *
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def double_pendulum_system(t, state, l1=.5, l2=.5, m1=1, m2=1, g=9.8):
    theta1, theta2, theta1_dot, theta2_dot = state

    delta_theta = theta1 - theta2
    sin_delta_theta = np.sin(delta_theta)
    cos_delta_theta = np.cos(delta_theta)
    
    denominator1 = l1 * (2 * m1 + m2 - m2 * cos_delta_theta**2)
    denominator2 = l2 * (2 * m1 + m2 - m2 * cos_delta_theta**2)

    theta1_dot_dot = (
        -g * (2 * m1 + m2) * np.sin(theta1) 
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * sin_delta_theta * m2 * (theta2_dot**2 * l2 + theta1_dot**2 * l1 * cos_delta_theta)
    ) / denominator1
    
    theta2_dot_dot = (
        2 * sin_delta_theta * (
            theta1_dot**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + theta2_dot**2 * l2 * m2 * cos_delta_theta
        )
    ) / denominator2

    return [theta1_dot, theta2_dot, theta1_dot_dot, theta2_dot_dot]

def double_pendulum_simple(t,state,l1=1, l2=1, m1=1, m2=1, g=9.8):
    theta1, theta2, theta1_dot, theta2_dot = state

    #use matrices to solved coupled differential equations
    theta_diff = theta1-theta2
    A = np.array([[(m1+m2)*l1,m2*l2*np.cos(theta_diff)],[l1*np.cos(theta_diff),l2]])
    F = np.array([-(m1+m2)*g*np.sin(theta1)-m2*l2*theta2_dot**2*np.sin(theta_diff),-g*np.sin(theta2)+theta1_dot**2*np.sin(theta_diff)])

    theta_dot_dot = np.linalg.solve(A,F)
    return theta1_dot,theta2_dot,theta_dot_dot[0],theta_dot_dot[1]

def solution_points(state0,time,dt=0.1):
    solution = solve_ivp(
        double_pendulum_simple,
        t_span=[0,time],
        y0=state0,
        t_eval=np.arange(0,time,dt)
    )
    print(solution.y[0:2].shape)
    solution.y[0, :] = solution.y[0, :] % (2 * np.pi)  # Normalize theta1
    solution.y[1, :] = solution.y[1, :] % (2 * np.pi)  # Normalize theta2
    print(solution.y.shape)
    return solution.y.T #return all

class DoublePendulum(Scene):
    def construct(self):
        L1 = 1
        L2 = 1
        m1 = 1.0
        m2 = 1.0
        g = 9.8
        axes = Axes(
            x_range=[-2,2,1],
            y_range=[-2,2,1],
            x_length=6,
            y_length=6,
            axis_config={"color":WHITE}
        )
        axes.center()
        self.add(axes)
        


        def coord(thetas):
            #print("Thetas; ",thetas)
            theta1,theta2 = thetas[:2]
            x1 = L1*np.sin(theta1)
            y1 = -L1*np.cos(theta1)
            x2 = x1 + L2*np.sin(theta2)
            y2 = y1 - L2*np.cos(theta2)
            return axes.c2p(x1,y1),axes.c2p(x2,y2)
        
        all_curves = []
        RUN_TIME=30
        def add_pendulum(state0,color1,color2,time=RUN_TIME):
            points = solution_points(state0,time)
            points = solution_points(state0,time)
            theta1s = points[:,0]
            theta2s = points[:,1]
            xypoints1 = [coord(point[:2])[0] for point in points]
            xypoints2 = [coord(point[:2])[1] for point in points]
            curve1 = VMobject(stroke_color=color1,stroke_width=1)
            curve1.set_points_as_corners(xypoints1)
            curve2 = VMobject(stroke_color=color2,stroke_width=1)
            curve2.set_points_as_corners(xypoints2)
            curve1.set_fill(opacity=0)
            curve1.set_opacity(0)
            curve2.set_fill(opacity=0)
            curve2.set_opacity(0)
            all_curves.append(Create(curve1))
            all_curves.append(Create(curve2))
            ini = coord(state0[:2]) #coord returns axes.c2p
            pivot = Dot(axes.c2p(0,0))
            line1 = Line(start = pivot.get_center(),end=ini[0])
            p1 = Circle(radius=0.1,color=color1,fill_color=color1)
            line2 = Line(start = p1.get_center(),end=ini[1])
            p2 = Circle(radius=0.1,color=color2,fill_color=color2)
            def updateline1(line):
                line.put_start_and_end_on(pivot.get_center(),curve1.get_end())
            def updatep1(p):
                p.move_to(line1.get_end())
            def updateline2(line):
                line.put_start_and_end_on(p1.get_center(),curve2.get_end())
            def updatep2(p):
                p.move_to(line2.get_end())

            line1.add_updater(updateline1)
            p1.add_updater(updatep1)
            line2.add_updater(updateline2)
            p2.add_updater(updatep2)
            self.add(pivot,line1,p1,line2,p2)
            

        #curves.set_opacity(0)
        
        #self.play(*[Create(curve1),Create(curve2),Create(flow1_curve),Create(flow2_curve)],run_time=time,rate_func=linear)
        ep = 1e-7
        state0=[PI/2,PI/2,0,0]
        states = [[PI/2,PI/2,0,0],[PI/2+1.1*ep,PI/2+ep,0,0]]
        n=100
        for i in range(n):
            state =[state0[0]+ep*i,state0[1]+ep*i,state0[2],state0[3]]
            add_pendulum(state,int(i/n*255),int(i/n*255))
        self.play(*all_curves,run_time = RUN_TIME,rate_func=linear)
        
def heatmap():
    ini_conditions = np.linspace(0,2*PI,250)
    grid = np.zeros((len(ini_conditions),len(ini_conditions)))
    for i,theta1 in enumerate(ini_conditions):
        print("Theta 1: ", theta1)
        for j,theta2 in enumerate(ini_conditions):
            state0 = [theta1,theta2,0,0]
            s = 1e-3
            state0p = [theta1+s,theta2,0,0]
            sol1 = solution_points(state0,10)
            sol2 = solution_points(state0p,10)
            divergence = np.mean(np.linalg.norm(sol1-sol2,axis =1))
            grid[i][j] = divergence
    plt.imshow(grid,extent=[-PI,PI,-PI,PI],origin="lower",cmap="inferno")
    plt.colorbar(label = "Divergence")
    plt.title("Heatmap of Chaos Indicators")
    plt.xlabel("Initial theta 1")
    plt.ylabel("Initial theta 2")
    plt.show()

def fractal():
    ini_conditions = np.linspace(0,2*PI,400)
    color_grid = np.zeros((len(ini_conditions),len(ini_conditions),3))#hsv = [hue (0-360),saturation(0-255),value]
    grid = np.zeros((len(ini_conditions),len(ini_conditions),3))
    def func(th1,th2):
        th1-=PI
        th2-=PI
        # Calculate basic polar coordinates
        radius = np.sqrt(th1**2 + th2**2)
        angle = np.arctan2(th2, th1)
        
        # Create smooth flowing patterns
        flow1 = np.sin(radius + angle)
        flow2 = np.cos(radius - angle)
        
        # Combine flows for smooth color transitions
        hue = (flow1 + flow2 + 2) / 4  # Normalize to [0,1]
        
        # Smooth saturation that increases towards edges
        saturation = np.clip(radius / (2 * np.pi) * 1.2, 0.5, 1.0)
        
        return hue, saturation
    for i,theta1 in enumerate(ini_conditions):
        for j,theta2 in enumerate(ini_conditions):
            if (theta2<0):
                print(theta1)
            theta1_nor = theta1/(2*PI)
            theta2_nor = theta2/(2*PI)
            color_grid[i][j] = hsv_to_rgb([*func(theta1,theta2),1])
   
    plt.imshow(color_grid, origin='lower', extent=(0, 2 * np.pi, 0, 2 * np.pi))
    plt.colorbar(label='Initial Color Mapping (Hue)')
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 2')
    plt.title('Initial Color Values for Double Pendulum')
    plt.show()


if __name__ == "__main__":
    #heatmap()
    #solution_points([PI,PI,0,0],10)
    fractal()

#print(double_pendulum_simple(0,[0,0,0,0]))