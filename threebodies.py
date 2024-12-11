import numpy as np
from scipy.integrate import solve_ivp
from manim import *
def three_body_system(t,state,m1=3.74e13,m2=1,m3=1e10,G=6.67430e-11):
    x1,y1,x1_dot,y1_dot,x2,y2,x2_dot,y2_dot,x3,y3,x3_dot,y3_dot = state
    eps = 1e-5
    r12_squared = (x1-x2)**2 + (y1-y2)**2 + eps
    r13_squared = (x1-x3)**2 + (y1-y3)**2 + eps
    r23_squared = (x2-x3)**2 + (y2-y3)**2 + eps

    x1_dot_dot = G * (m2/r12_squared * np.cos(np.arctan2(y2-y1,x2-x1)) + m3 / r13_squared*np.cos(np.arctan2(y3-y1,x3-x1)))
    y1_dot_dot = G * (m2/r12_squared * np.sin(np.arctan2(y2-y1,x2-x1)) + m3 / r13_squared*np.sin(np.arctan2(y3-y1,x3-x1)))

    x2_dot_dot = G * (m1/r12_squared * np.cos(np.arctan2(y1-y2,x1-x2)) + m3 / r23_squared*np.cos(np.arctan2(y3-y2,x3-x2)))
    y2_dot_dot = G * (m1/r12_squared * np.sin(np.arctan2(y1-y2,x1-x2)) + m3 / r23_squared*np.sin(np.arctan2(y3-y2,x3-x2)))

    x3_dot_dot = G * (m1/r13_squared * np.cos(np.arctan2(y1-y3,x1-x3)) + m2 / r23_squared*np.cos(np.arctan2(y2-y3,x2-x3)))
    y3_dot_dot = G * (m1/r13_squared * np.sin(np.arctan2(y1-y3,x1-x3)) + m2 / r23_squared*np.sin(np.arctan2(y2-y3,x2-x3)))

    return np.array([x1_dot,y1_dot,x1_dot_dot,y1_dot_dot,x2_dot,y2_dot,x2_dot_dot,y2_dot_dot,x3_dot,y3_dot,x3_dot_dot,y3_dot_dot])

def three_body_system_solution(state0,t,dt=.01):
    sol = solve_ivp(fun=three_body_system, t_span=[0,t], y0=state0, t_eval=np.arange(0,t,dt))
    return sol.y.T

class ThreeBodySystem(Scene):
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
        G=6.67430e-11
        state0 = [0,0,0,0,100,0,0,np.sqrt(G*1e10/10),90,0,0,np.sqrt(10/9)*5]
        time = 60
        points = three_body_system_solution(state0,time)
        points1 = points[:,[0,1]]  # x1,y1 coordinates
        points2 = points[:,[4,5]]  # x2,y2 coordinates
        points3 = points[:,[8,9]]  # x3,y3 coordinates

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
        c3 = Circle(radius=0.1,color=BLUE)
        c3.add_updater(lambda m: m.move_to(curve3.get_end()))

        # Create text labels
        label1 = Text("Sun", font_size=24, color=WHITE).next_to(c1, UP)
        label2 = Text("Moon", font_size=24, color=WHITE).next_to(c2, UP)
        label3 = Text("Earth", font_size=24, color=WHITE).next_to(c3, UP)

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
        self.play(*[Create(curve1),Create(curve2),Create(curve3)],run_time=time/3,rate_func=linear)
