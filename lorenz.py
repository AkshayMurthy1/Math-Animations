from manim import *
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def lorenz_system(t,state,sigma=10,rho=28,beta=8/3):
    x,y,z = state
    dxdt = sigma * (y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-beta*z
    return [dxdt,dydt,dzdt]

def ode_solution_points(function,state0,time,dt=.005):
    solution = solve_ivp(
        function,
        t_span=[0,time], #how long are we running it for
        y0=state0,
        t_eval=np.arange(0,time,dt) #how long each step is?
    )
    return solution.y.T

class LorenzAttractor(ThreeDScene):
    def construct(self):
        #set up axes
        axes = ThreeDAxes(
            x_range=[-50,50,1],
            y_range=[-50,50,1],
            z_range=[0,50,1],
            x_length = 16,
            y_length= 16,
            z_length=8
        )
        axes.center()

        self.set_camera_orientation(phi=76*DEGREES, theta=43*DEGREES, gamma=1*DEGREES)
        self.add(axes)

        # state0=[10,10,10]
        # points = ode_solution_points(lorenz_system,state0,10) #have a bunch of points starting at t=0 to t=10 with step of .01 (one point for each step)
        ev_time = 25
        ep = 1e-2
        states = [[5+n*0,4+n*0,10+n*ep] for n in range(10)]
        colors = color_gradient([RED_B,BLUE_B],len(states))
        curves = VGroup() #many VMobjects

        for state,color in zip(states,colors):
            points = ode_solution_points(lorenz_system,state,ev_time)
            curve = VMobject(stroke_color=color,stroke_width=1)
            points_3d = [axes.c2p(*point) for point in points]
            curve.set_points_as_corners(points_3d)
            curves.add(curve)
        
        dots = Group(*[Dot(color=color,radius=.1) for color in colors])
        def update_dots(dots):
            for dot,curve in zip(dots,curves):
                dot.move_to(curve.get_end())
        dots.add_updater(update_dots) #function is called on group of dots every frame

        #tails = VGroup(*[TracedPath(dot.get_center,dissipating_time=.5,stroke_color=color) for dot,color in zip(dots,colors)])

        curves.set_opacity(.9)
        curves.set_fill(opacity=0)
        #self.add(tails)
        self.add(dots)
        # self.play(Create(curves,run_time=10,rate_func=linear))
        #you can play many create animations simmultaneously
        animations = [Create(curve,rate_func=linear) for curve in curves]
        self.play(*animations,run_time=ev_time,)

        # self.begin_ambient_camera_rotation(rate=0.2)
        # self.wait(10)  # Waits for 10 seconds while rotating
        # self.stop_ambient_camera_rotation()
# print(len(ode_solution_points(lorenz_system,[0,1,2],10)))

