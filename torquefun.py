import numpy as np
from scipy.integrate import solve_ivp
from manim import *

def torquefun_system(t,state,M=3,m=.5,g=9.8,L=10,k=100):
    theta,x,y,thetha_dot,x_dot,y_dot = state
    thetha_dot_dot = -M*g*L*np.cos(theta)/2 - m*g*np.cos(theta)*(x**2+y**2)**(1/2)
    x0 = np.cos(theta)*L/2
    y0 = np.sin(theta)*L/2

    r_dist = np.sqrt((x-x0)**2 + (y-y0)**2)
    if theta%(2*np.pi) > np.pi:
        r_dist = -r_dist

    x_dot_dot = np.cos(theta)*k*r_dist-m*g*np.cos(theta)*np.abs(np.sin(theta))
    y_dot_dot = np.sin(theta)*k*r_dist-m*g*np.sin(theta)**2
    # print(f"t={t}, state={state}, r_dist={r_dist}, thetha_dot_dot={thetha_dot_dot}")
    return np.array([thetha_dot,x_dot,y_dot,thetha_dot_dot,x_dot_dot,y_dot_dot])

def torquefun_system_polar(t,state,M=3,m=.5,g=9.8,L=10,k=100):
    I=1/6*M*L**2
    r0= L/2
    theta,r,theta_dot,r_dot = state
    theta_dot_dot = -1*(M*g*r*np.cos(theta)+1/2*m*g*L*np.cos(theta))/I
    r_dot_dot = -1*(k*(r-r0)+m*g*np.sin(theta))/m
    return np.array([theta_dot,r_dot,theta_dot_dot,r_dot_dot])

def torquefun_solution(state0,t,dt=.01):
    sol = solve_ivp(fun=torquefun_system, t_span=[0,t], y0=state0, t_eval=np.arange(0,t,dt))
    return sol.y.T
def torquefun_solution_polar(state0,t,dt=.01):
    sol = solve_ivp(fun=torquefun_system_polar, t_span=[0,t], y0=state0, t_eval=np.arange(0,t,dt))
    return sol.y.T

def test():
    state0 = np.array([np.pi, 0, 0, 0, 0, 0])
    points = torquefun_solution(state0,2)
    L=1
    print([[L*np.cos(theta),L*np.sin(theta)] for theta in points[:,[0]]][0])
    print()

class TorqueFunScene(Scene):
    def construct(self):
        
        axes = Axes(
            x_range=[-10,10,10],
            y_range=[-10,10,10],
            x_length=8,
            y_length=8,
            axis_config={"color":WHITE}
        )
        axes.center()
        self.add(axes)
        L=10
        M=3
        m=.5
        g=9.8
        I = 1/6*M*L**2
        k=100
        state0 = np.array([np.pi/2, L, 2*np.pi, -5])
        time = 7
        points = torquefun_solution_polar(state0,time)
        thetas = points[:,0]
        rs = points[:,1]
        theta_dots = points[:,2]
        r_dots = points[:,3]
        Ks = [1/2*I*theta_dot**2 + 1/2*m*r_dot**2 for theta_dot,r_dot in zip(theta_dots,r_dots)]
        Us = [1/2*k*(r-L/2)**2 + m*g*r*np.sin(theta) + M*g*L*np.sin(theta)/2 for r,theta in zip(rs,thetas)]
        ball_points = [[rs[i]*np.cos(thetas[i]),rs[i]*np.sin(thetas[i])] for i in range(len(thetas))]
        #print(np.array(ball_points))
        ball_points = [axes.c2p(*point) for point in ball_points]
        ball_curve = VMobject().set_points_as_corners(ball_points)
        ball_curve.set_color(RED)
        ball_curve.set_opacity(.2)
        rod_points = [[L*np.cos(theta),L*np.sin(theta)] for theta in thetas]
        rod_points = [axes.c2p(*point) for point in rod_points]
        rod_curve = VMobject().set_points_as_corners(rod_points)
        rod_curve.set_color(BLUE)
        rod_curve.set_opacity(.2)
        print(len(ball_curve.get_points()))
        # Display energy text
        K_text = Text(f"K: {Ks[0]:.2f} FIRST", color=RED).scale(0.5).to_corner(UP + LEFT)
        U_text = Text(f"U: {Us[0]:.2f}", color=BLUE).scale(0.5).next_to(K_text, DOWN, buff=0.2)
        E_text = Text(f"E: {Ks[0]+Us[0]:.2f}", color=GREEN).scale(0.5).next_to(U_text, DOWN, buff=0.2)
        def update_K_text(obj):
            # Find current index based on animation progress
            # current_idx = int(len(ball_curve.get_points()) * self.renderer.time / time)
            # current_idx = min(current_idx, len(Ks) - 1)  # Prevent overflow
            current_idx = int(self.renderer.time / time * len(Ks))
            obj.become(Text(f"K: {Ks[current_idx]:.2f} ci={current_idx}", color=RED).scale(0.5).to_corner(UP + LEFT))

        def update_U_text(obj):
            # current_idx = int(len(ball_curve.get_points()) * self.renderer.time / time)
            # current_idx = min(current_idx, len(Us) - 1)  # Prevent overflow
            current_idx = int(self.renderer.time / time * len(Us))
            obj.become(Text(f"U: {Us[current_idx]:.2f}", color=BLUE).scale(0.5).next_to(K_text, DOWN, buff=0.2))

        def update_E_text(obj):
            # current_idx = int(len(ball_curve.get_points()) * self.renderer.time / time)
            # current_idx = min(current_idx, len(Ks) - 1)  # Prevent overflow
            current_idx = int(self.renderer.time / time * len(Ks))
            obj.become(Text(f"E: {Ks[current_idx]+Us[current_idx]:.2f}", color=GREEN).scale(0.5).next_to(U_text, DOWN, buff=0.2))
        
        K_text.add_updater(update_K_text)
        U_text.add_updater(update_U_text)
        E_text.add_updater(update_E_text)

        # Add elements to the scene
        self.add(K_text, U_text, E_text)
        self.add(ball_curve, rod_curve)

        rod = Line(color=BLUE)
        ball = Circle(radius=0.1,color=RED)
        def update_rod(rod):
            rod.put_start_and_end_on(axes.c2p(0,0),rod_curve.get_end()) 
        def update_ball(ball):
            ball.move_to(ball_curve.get_end())
        rod.add_updater(update_rod)
        ball.add_updater(update_ball)
        self.add(rod,ball)
        self.play(Create(ball_curve),Create(rod_curve),run_time=time,rate_function=linear)

        # state0 = [np.pi,-.25,0,0,0,0]
        # time =2
        # points = torquefun_solution(state0,time)
        # theta_points = [[L*np.cos(theta),L*np.sin(theta)] for theta in points[:,0]]
        # print(theta_points)
        # theta_points = [axes.c2p(*point) for point in theta_points]
        # ball_points = points[:,[1,2]]
        # print(np.array(ball_points).shape)
        # ball_points = [axes.c2p(*point) for point in ball_points]

        # rod_curve = VMobject().set_points_as_corners(theta_points)
        # ball_curve = VMobject().set_points_as_corners(ball_points)
        # ball_curve.set_color(RED)
        # self.play(Create(rod_curve),Create(ball_curve),run_time=10,rate_function=linear)

        

if __name__ == "__main__":
    test()
