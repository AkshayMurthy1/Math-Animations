import numpy as np
from scipy.integrate import solve_ivp
from manim import *

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

def solution_points(state0,time,dt=0.01):
    solution = solve_ivp(
        double_pendulum_simple,
        t_span=[0,time],
        y0=state0,
        t_eval=np.arange(0,time,dt)
    )
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
        axes.set_aspect(1)
        axes.shift(2*LEFT)
        self.add(axes)
        phase_graph = Axes(
            x_range=[-3,3,1],
            y_range=[-15,15,1],
            x_length=6,
            y_length=6,
            axis_config={"color":RED},
        )
        
        phase_graph.shift(3*RIGHT)
        x_label = MathTex(r"\theta").next_to(phase_graph.x_axis, DOWN)
        y_label = MathTex(r"I*\dot{\theta}").next_to(phase_graph.y_axis, LEFT)
        self.add(x_label,y_label)
        phase_graph.add(x_label, y_label)
        self.add(phase_graph)


        def coord(thetas):
            #print("Thetas; ",thetas)
            theta1,theta2 = thetas[:2]
            x1 = L1*np.sin(theta1)
            y1 = -L1*np.cos(theta1)
            x2 = x1 + L2*np.sin(theta2)
            y2 = y1 - L2*np.cos(theta2)
            return axes.c2p(x1,y1),axes.c2p(x2,y2)
        
        
        time = 30
        state0 = [PI/6,PI/6,0,0]
        points = solution_points(state0,time)
        print(points.shape)
        print("BEGINNING",coord(points[0]))
        theta1s = points[:,0]
        theta2s = points[:,1]
        dtheta1s = points[:,2]
        dtheta2s = points[:,3]

        # theta1_flow = np.stack([theta1s,m1*L1**2*dtheta1s],axis=1)
        # flow1_points = [phase_graph.c2p(*point) for point in theta1_flow]
        # theta2_flow = np.stack([theta2s,m2*L2**2*dtheta2s],axis=1)
        # flow2_points = [phase_graph.c2p(*point) for point in theta2_flow]

        # flow1_curve = VMobject(stroke_color=RED,stroke_width=4)
        # flow1_curve.set_points_as_corners(flow1_points)
        # flow2_curve = VMobject(stroke_color = BLUE,stroke_width=4)
        # flow2_curve.set_points_as_corners(flow2_points)

        #print(theta1_flow.shape)
        #print(theta1s.shape)
        #points = [axes.c2p(*point) for point in points]
        #print("Shape of points: ",(points).shape)
        xypoints1 = [coord(point[:2])[0] for point in points]
        xypoints2 = [coord(point[:2])[1] for point in points]
        curve1 = VMobject(stroke_color=RED,stroke_width=1)
        curve1.set_points_as_corners(xypoints1)
        curve2 = VMobject(stroke_color=BLUE,stroke_width=1)
        curve2.set_points_as_corners(xypoints2)
        curves = VGroup(curve1,curve2)
        
        #should play simulatenously        
        fade_dur = 5
        curve1.add_updater(
            lambda o, dt: o.set_opacity(max(0, o.get_stroke_opacity() - dt / fade_dur))
        )
        ini = coord(state0[:2])
        #ini_c = Dot(point=ini[0],radius=1)
        #ini_c2 = Dot(point=ini[1],radius=1)
        #self.add(ini_c,ini_c2)

        pivot = Dot(axes.c2p(0,0))
        line1 = Line(start = pivot.get_center(),end=ini[0])
        p1 = Circle(radius=0.1,color=RED,fill_color=RED)
        line2 = Line(start = p1.get_center(),end=ini[1])
        p2 = Circle(radius=0.1,color=BLUE,fill_color=BLUE)

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

        all_curves = []
        RUN_TIME=10
        def add_pendulum(state0,time=RUN_TIME):
            points = solution_points(state0,time)
            points = solution_points(state0,time)
            theta1s = points[:,0]
            theta2s = points[:,1]
            xypoints1 = [coord(point[:2])[0] for point in points]
            xypoints2 = [coord(point[:2])[1] for point in points]
            curve1 = VMobject(stroke_color=RED,stroke_width=1)
            curve1.set_points_as_corners(xypoints1)
            curve2 = VMobject(stroke_color=BLUE,stroke_width=1)
            curve2.set_points_as_corners(xypoints2)
            all_curves.append(curve1)
            all_curves.append(curve2)
            ini = coord(state0[:2]) #coord returns axes.c2p
            pivot = Dot(axes.c2p(0,0))
            line1 = Line(start = pivot.get_center(),end=ini[0])
            p1 = Circle(radius=0.1,color=RED,fill_color=RED)
            line2 = Line(start = p1.get_center(),end=ini[1])
            p2 = Circle(radius=0.1,color=BLUE,fill_color=BLUE)
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
        curves.set_fill(opacity=0)
        #self.play(*[Create(curve1),Create(curve2),Create(flow1_curve),Create(flow2_curve)],run_time=time,rate_func=linear)
        self.play(*all_curves,run_time = RUN_TIME,rate_func=linear)
        

#print(double_pendulum_simple(0,[0,0,0,0]))