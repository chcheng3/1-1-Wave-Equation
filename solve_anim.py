import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




############ Solving the 1+1 scalar wave equation ############

# Number of points for discretization
NX = 201
NT = 4001

# Step size
DX = 0.100
DT = 0.010

# Endpoint for x and t
XF = 20.
TF = 40.

def fdiff_4(u, u_t, dt, i, nx, dx):
    """ The centered finite difference formula for d^2 u(x)/dt^2
        to order (dx)^4 accuracy 
    """

    # Take the adjacent spatial indices subject to periodic boundary conditions
    i1 = (i-2)%nx
    i2 = (i-1)%nx
    i3 = i
    i4 = (i+1)%nx
    i5 = (i+2)%nx

    # Advance the function by a time step at each position index  
    u1 = u[i1] + u_t[i1]*dt
    u2 = u[i2] + u_t[i2]*dt
    u3 = u[i3] + u_t[i3]*dt
    u4 = u[i4] + u_t[i4]*dt
    u5 = u[i5] + u_t[i5]*dt

    # Take the finite difference formula for second derivative
    return (-u1 + 16*u2 - 30*u3 +16*u4 -u5)/(12.*dx**2)



def rk4_step(phi, pi, nx, dt, dx):
    """ Advance phi(t), pi(t) by dt using a fourth-order Runge-Kutta step """
    # Take the vectors at the latest point in time
    u = phi[-1]
    v = pi[-1]

    # Initialize the array of new vectors
    uu = [0. for i in range(nx)]
    vv = [0. for i in range(nx)]

    # Compute the slopes for u and v
    k1 = [ -(v[i]) for i in range(nx)]
    l1 = [ -fdiff_4(u, uu, 0, i, nx, dx) for i in range(nx)]

    k2 = [ -(v[i] + 0.5*dt*l1[i]) for i in range(nx)]
    l2 = [ -fdiff_4(u, k1, 0.5*dt, i, nx, dx) for i in range(nx)]

    k3 = [ -(v[i] + 0.5*dt*l2[i]) for i in range(nx)]
    l3 = [ -fdiff_4(u, k2, 0.5*dt, i, nx, dx) for i in range(nx)]

    k4 = [ -(v[i] + dt*l3[i]) for i in range(nx)]
    l4 = [ -fdiff_4(u, k3, dt, i, nx, dx) for i in range(nx)]

    for i in range(nx):
        # Advance u and v by a weighted sum of the slopes at each position
        uu[i] = u[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])*dt/6.
        vv[i] = v[i] + (l1[i] + 2*l2[i] + 2*l3[i] + l4[i])*dt/6.
    return uu, vv




# Set up initial conditions
xlist = np.linspace(0, XF, endpoint=True, num=NX)
phi = [ [np.exp(-(x - XF*0.5)**2) for x in xlist] ]
pi  = [ [0. for x in xlist] ]




# Integrate the ODEs using RK4 until the end of time
for j in range(NT):
    phi2, pi2 = rk4_step(phi, pi, NX, DT, DX)
    phi.append(phi2)
    pi.append(pi2)





############ Making animations ############

fig = plt.figure()
ax = plt.axes(xlim=(0, XF), ylim = (-0.001, 1))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\phi(x, t)$')


line1, = ax.plot([], [], lw=1)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

SPEED=4
def animate(i):
    t = DT*i*SPEED
    y1 = [phi[i*SPEED][j] for j in range(NX)]
    line1.set_data(xlist, y1)
    time_text.set_text("time = {:.1f}".format(t))
    return line1, time_text

anim = animation.FuncAnimation(fig, animate, frames=NT//SPEED, interval=20, blit=True)
anim.save('anim.gif')
