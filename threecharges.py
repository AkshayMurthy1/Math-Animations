import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from manim import *

def three_charge_system(t, state, k = 9e9, m1=1e2,m2=1e-1,m3=1e-1, q1=1e1, q2=-1e-4, q3=-1e-4,endl=-20,endr=20,endb=-20,endt=20):
    """
    Define the system of differential equations for three charged particles.
    State vector: [x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot, x3, y3, x3_dot, y3_dot]
    Returns derivatives: [x1_dot, y1_dot, x1_dot_dot, y1_dot_dot, x2_dot, y2_dot, x2_dot_dot, y2_dot_dot, 
                        x3_dot, y3_dot, x3_dot_dot, y3_dot_dot]
    """
    # Unpack state variables
    x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot, x3, y3, x3_dot, y3_dot = state
    # c=-10
    # if x1<endl or x1>endr:
    #     x1_dot = c*x1_dot
    # if y1<endb or y1>endt:
    #     y1_dot = c*y1_dot
    # if x2<endl or x2>endr:
    #     x2_dot = c*x2_dot
    # if y2<endb or y2>endt:
    #     y2_dot = c*y2_dot
    # if x3<endl or x3>endr:
    #     x3_dot = c*x3_dot
    # if y3<endb or y3>endt:
    #     y3_dot = c*y3_dot
    
    # calc distances for pairs
    r12_squared = (x1-x2)**2 + (y1-y2)**2
    r13_squared = (x1-x3)**2 + (y1-y3)**2
    r23_squared = (x2-x3)**2 + (y2-y3)**2
    
    # r^(3/2)
    r12_32 = r12_squared**(3/2)
    r13_32 = r13_squared**(3/2)
    r23_32 = r23_squared**(3/2)
    
    # calc second derivatives (accelerations)
    x1_dot_dot = (k*q1/m1) * (q2*(x1-x2)/r12_32 + q3*(x1-x3)/r13_32)
    y1_dot_dot = (k*q1/m1) * (q2*(y1-y2)/r12_32 + q3*(y1-y3)/r13_32)
    
    x2_dot_dot = (k*q2/m2) * (q1*(x2-x1)/r12_32 + q3*(x2-x3)/r23_32)
    y2_dot_dot = (k*q2/m2) * (q1*(y2-y1)/r12_32 + q3*(y2-y3)/r23_32)
    
    x3_dot_dot = (k*q3/m3) * (q1*(x3-x1)/r13_32 + q2*(x3-x2)/r23_32)
    y3_dot_dot = (k*q3/m3) * (q1*(y3-y1)/r13_32 + q2*(y3-y2)/r23_32)
    
    # return [ẋ₁, ẏ₁, ẍ₁, ÿ₁, ẋ₂, ẏ₂, ẍ₂, ÿ₂, ẋ₃, ẏ₃, ẍ₃, ÿ₃]
    return np.array([
        x1_dot, y1_dot, x1_dot_dot, y1_dot_dot,
        x2_dot, y2_dot, x2_dot_dot, y2_dot_dot,
        x3_dot, y3_dot, x3_dot_dot, y3_dot_dot
    ])
def three_charge_system_basic(t, state, k = 9e9, m1=1e2,m2=1e-1,m3=1e-1, q1=1e1, q2=-1e-4, q3=-1e-4,endl=-20,endr=20,endb=-20,endt=20):
    """
    Define the system of differential equations for three charged particles.
    State vector: [x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot, x3, y3, x3_dot, y3_dot]
    Returns derivatives: [x1_dot, y1_dot, x1_dot_dot, y1_dot_dot, x2_dot, y2_dot, x2_dot_dot, y2_dot_dot, 
                        x3_dot, y3_dot, x3_dot_dot, y3_dot_dot]
    """
    # Unpack state variables
    x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot, x3, y3, x3_dot, y3_dot = state
    # c=-10
    # if x1<endl or x1>endr:
    #     x1_dot = c*x1_dot
    # if y1<endb or y1>endt:
    #     y1_dot = c*y1_dot
    # if x2<endl or x2>endr:
    #     x2_dot = c*x2_dot
    # if y2<endb or y2>endt:
    #     y2_dot = c*y2_dot
    # if x3<endl or x3>endr:
    #     x3_dot = c*x3_dot
    # if y3<endb or y3>endt:
    #     y3_dot = c*y3_dot
    
    # calc distances for pairs
    r12_squared = (x1-x2)**2 + (y1-y2)**2
    r13_squared = (x1-x3)**2 + (y1-y3)**2
    r23_squared = (x2-x3)**2 + (y2-y3)**2
    
    
    # calc second derivatives (accelerations)
    x1_dot_dot = 1/m1*(k*q1*q2/r12_squared * np.cos(np.arctan2(y1-y2,x1-x2)) + k*q1*q3/r13_squared * np.cos(np.arctan2(y1-y3,x1-x3)))
    y1_dot_dot = 1/m1 * (k*q1*q2/r12_squared * np.sin(np.arctan2(y1-y2,x1-x2)) + k*q1*q3/r13_squared * np.sin(np.arctan2(y1-y3,x1-x3)))
    
    x2_dot_dot = 1/m2*(k*q1*q2/r12_squared * np.cos(np.arctan2(y2-y1,x2-x1)) + k*q1*q3/r23_squared * np.cos(np.arctan2(y2-y3,x2-x3)))
    y2_dot_dot = 1/m2 * (k*q1*q2/r12_squared * np.sin(np.arctan2(y2-y1,x2-x1)) + k*q1*q3/r23_squared * np.sin(np.arctan2(y2-y3,x2-x3)))
    
    x3_dot_dot = 1/m3*(k*q1*q2/r13_squared * np.cos(np.arctan2(y3-y1,x3-x1)) + k*q1*q3/r23_squared * np.cos(np.arctan2(y3-y2,x3-x2)))
    y3_dot_dot = 1/m3 * (k*q1*q2/r13_squared * np.sin(np.arctan2(y3-y1,x3-x1)) + k*q1*q3/r23_squared * np.sin(np.arctan2(y3-y2,x3-x2)))
    
    # return [ẋ₁, ẏ₁, ẍ₁, ÿ₁, ẋ₂, ẏ₂, ẍ₂, ÿ₂, ẋ₃, ẏ₃, ẍ₃, ÿ₃]
    return np.array([
        x1_dot, y1_dot, x1_dot_dot, y1_dot_dot,
        x2_dot, y2_dot, x2_dot_dot, y2_dot_dot,
        x3_dot, y3_dot, x3_dot_dot, y3_dot_dot
    ])

def simulate_charges_points(state0,time,dt=.01):
    """
    Simulate the motion of three charged particles.
    
    Parameters:
    t_span : tuple
        (t_start, t_end) for simulation
    initial_state : array
        Initial conditions [x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot, x3, y3, x3_dot, y3_dot]
    k, m1, q1, q2, q3 : float
        System parameters
    """
    solution = solve_ivp(
        fun=three_charge_system_basic,
        t_span=[0,time],
        y0=state0,
        t_eval=np.arange(0,time,dt)
    )
    #only want 0,1,4,5,8,9th columns
    #return solution.y.T[:,[0,1,4,5,8,9]]
    return solution.y.T

def plot_trajectories(solution):
    """Plot the trajectories of the three charged particles."""
    plt.figure(figsize=(10, 10))
    
    # Extract positions
    x1, y1 = solution[0], solution[1]
    x2, y2 = solution[4], solution[5]
    x3, y3 = solution[8], solution[9]
    
    # Plot trajectories
    plt.plot(x1, y1, 'r-', label='Particle 1')
    plt.plot(x2, y2, 'b-', label='Particle 2')
    plt.plot(x3, y3, 'g-', label='Particle 3')
    
    # Plot starting positions
    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'bo')
    plt.plot(x3, y3, 'go')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of Three Charged Particles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

class ThreeCharges(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-100,100,10],
            y_range=[-100,100,10],
            x_length=8,
            y_length=8,
            axis_config={"color":WHITE}
        )
        axes.center()
        self.add(axes)
        state0 = [-50,-50,0,0,50,50,0,0,-50,50,0,0]
        time = 30
        points1 = simulate_charges_points(state0,time)[:,[0,1]]  # x1,y1 coordinates
        points2 = simulate_charges_points(state0,time)[:,[4,5]]  # x2,y2 coordinates
        points3 = simulate_charges_points(state0,time)[:,[8,9]]  # x3,y3 coordinates

        points1 = [axes.c2p(*point) for point in points1]
        points2 = [axes.c2p(*point) for point in points2]
        points3 = [axes.c2p(*point) for point in points3]

        curve1 = VMobject().set_points_as_corners(points1)
        curve2 = VMobject().set_points_as_corners(points2)
        curve3 = VMobject().set_points_as_corners(points3)
        
        c1 = Circle(radius=0.1,color=GREEN)
        c1.add_updater(lambda m: m.move_to(curve1.get_end()))
        c2 = Circle(radius=0.1,color=RED)
        c2.add_updater(lambda m: m.move_to(curve2.get_end()))
        c3 = Circle(radius=0.1,color=RED)
        c3.add_updater(lambda m: m.move_to(curve3.get_end()))

        # Create text labels
        label1 = Text("+q1", font_size=24, color=WHITE).next_to(c1, UP)
        label2 = Text("-q2", font_size=24, color=WHITE).next_to(c2, UP)
        label3 = Text("-q3", font_size=24, color=WHITE).next_to(c3, UP)

        # Group circles with their labels
        c1_group = VGroup(c1, label1)
        c2_group = VGroup(c2, label2)
        c3_group = VGroup(c3, label3)

        # Add updaters to keep labels with circles
        label1.add_updater(lambda m: m.next_to(c1, UP))
        label2.add_updater(lambda m: m.next_to(c2, UP))
        label3.add_updater(lambda m: m.next_to(c3, UP))
        self.add(c1_group, c2_group, c3_group)

        curve1.set_opacity(0)
        curve2.set_opacity(0)
        curve3.set_opacity(0)
        self.play(*[Create(curve1),Create(curve2),Create(curve3)],run_time=time/2,rate_func=linear)

