import os
import time
import pickle
import datetime
import numpy as np
from numpy import sin  as s
from numpy import cos as c
from ruamel.yaml import YAML
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class DynamicModel():
    def __init__(self, cfg, mm=[1.5, 1.5], gam=[1.0, 1.0]):
        self.Mass = list(cfg['Robot']['Mass']['mass'])
        self.Inertia = list(cfg['Robot']['Mass']['inertia'])
        self.massCenter = list(cfg['Robot']['Mass']['massCenter'])
        self.GeoLength = list(cfg['Robot']['Geometry']['length'])
        self.dof = len(self.GeoLength)

        self.Mass.append(mm)
        self.gam = gam

        pass

    def Posture(self, q):
        L = self.GeoLength
        posx = np.zeros(self.dof+1)
        posy = np.zeros(self.dof+1)
        for i in range(self.dof+1):
            qj = 0
            for j in range(i+1):
                if j == 0:
                    posx[i]=0
                    posy[i]=0
                else:
                    qj += q[j-1]
                    posx[i] += L[j-1]*s(qj)
                    posy[i] += L[j-1]*c(qj)
        
        return posx, posy


class DataProcess():
    def __init__(self, cfg, q, dq, ddq, u, t, gamma, m, savepath, save_flag, OutputPath='./image/', **params):
        self.cfg = cfg
        self.q = q
        self.dq = dq
        self.ddq = ddq
        self.u = u
        self.t = t
        self.m = m
        self.gam = gamma
        self.date = params['date']
        self.savepath = savepath
        self.OutputPath = OutputPath
        self.save_flag = save_flag
        self.GeoLength = list(cfg['Robot']['Geometry']['length'])
        self.dof = len(self.GeoLength)

        # region plot
        plt.style.use("science")

        self.params = {
            'text.usetex': True,
            'font.size': 15,
            'axes.titlesize': 15,
            'legend.fontsize': 15,
            'axes.labelsize': 15,
            'lines.linewidth': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'axes.titlepad': 3.0,
            'axes.labelpad': 3.0,
            'lines.markersize': 8,
            'figure.subplot.wspace': 0.4,
            'figure.subplot.hspace': 0.8,
        }

        plt.rcParams.update(self.params)
        # endregion

        # self.save_dir, self.name, self.date = self.DirCreate()
        pass

    def TrajPlot(self):
        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(18/2.54, 16/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], wspace=0.3, hspace=0.5)
        g_data = gs[1].subgridspec(2, self.dof, wspace=0.3, hspace=0.6)
        ax_m = fig.add_subplot(gs[0])
        ax_v = [fig.add_subplot(g_data[0, i]) for i in range(self.dof)]
        ax_u = [fig.add_subplot(g_data[1, i]) for i in range(self.dof)]

        #! plot trajectory ------------------------------------------
        ax_m.axhline(y=-0.0, color='k', zorder=0)
        num_frame = 5
        T = self.cfg["Controller"]["Period"]
        
        for tt in np.linspace(0, T, num_frame):
            idx = np.argmin(np.abs(self.t-tt))
            model = DynamicModel(self.cfg, self.m, self.gam)
            posx, posy = model.Posture(self.q[idx, :])
            for j in range(self.dof):
                ax_m.plot([posx[j], posx[j+1]], [posy[j], posy[j+1]], 
                        'o-', ms=1, color=cmap(tt/T), alpha=tt/T*0.8+0.2, lw=1)
            pass

        ax_m.axis('equal')
        ax_m.text(-0.05, 1.1, r"$\mathbf{A}$",transform=ax_m.transAxes)
        ax_m.set_xlabel('x (m)')
        ax_m.set_ylabel('y (m)')

        #! plot vel -------------------------------------------
        titlelabel_v = [r"$\dot{\theta}_s\ (rad/s)$",r"$\dot{\theta}_e\ (rad/s)$"]
        axislabel = ["t (s)","angular vel (rad/s)"]
        index = [r"$\mathbf{B}$",r"$\mathbf{C}$"]
        [ax_v[i].plot(self.t, self.dq[:, i], color='C0') for i in range(self.dof)]
        [ax_v[i].set_title(titlelabel_v[i]) for i in range(self.dof)]
        [ax_v[i].set_xlim([0,T]) for i in range(self.dof)]
        ax_v[0].set_ylabel(axislabel[1])
        [ax_v[i].text(-0.05,1.1,index[i],transform=ax_v[i].transAxes) for i in range(self.dof)]

        #! plot force -------------------------------------------
        titlelabel_u = [r"$\tau_s\ (Nm)$",r"$\tau_e\ (N.m)$"]
        index = [r"$\mathbf{D}$",r"$\mathbf{E}$"]
        axislabel = ["t (s)","torque (N.m)"]
        ax_u[0].plot(self.t, self.u[:, 0], color='C0')
        ax_u[1].plot(self.t, self.u[:, 1], color='C0')
        [ax_u[i].set_title(titlelabel_u[i]) for i in range(self.dof)]
        [ax_u[i].set_xlim([0,T]) for i in range(self.dof)]
        [ax_u[i].set_xlabel(axislabel[0]) for i in range(self.dof)]
        ax_u[0].set_ylabel(axislabel[1])
        [ax_u[i].text(-0.05,1.1,index[i],transform=ax_u[i].transAxes) for i in range(self.dof)]

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/traj_opt.png", dpi=600)
        else:
            return fig


    def DirCreate(self, method_choice=2):
        trackingCoeff = self.cfg["Optimization"]["CostCoeff"]["trackingCoeff"]
        VelCoeff = self.cfg["Optimization"]["CostCoeff"]["VelCoeff"]
        powerCoeff = self.cfg["Optimization"]["CostCoeff"]["powerCoeff"]
        forceCoeff = self.cfg["Optimization"]["CostCoeff"]["forceCoeff"]
        smoothCoeff = self.cfg["Optimization"]["CostCoeff"]["smoothCoeff"]
        impactCoeff = self.cfg["Optimization"]["CostCoeff"]["ImpactCoeff"]
        Vt = self.cfg["Controller"]["Target"]
        Tp = self.cfg["Controller"]["Period"]
        # Tst = self.cfg["Controller"]["Stance"]
        dt = self.cfg["Controller"]["dt"]
        theta = round(self.theta, 3)
        
        date = time.strftime("%Y-%m-%d-%H-%M-%S")
        if method_choice==1:
            dirname = "-Traj-Tcf_"+str(trackingCoeff)+"-Pcf_"+str(powerCoeff)+"-Fcf_"+str(forceCoeff)+\
                        "-Scf_"+str(smoothCoeff)+"-Icf_"+str(impactCoeff)+"-Vt_"+str(Vt)+"-Tp_"+str(Tp)+"-Tst_"+str(Tst)
        if method_choice==2:
            dirname = "-Traj-Tcf_"+str(trackingCoeff)+"-Pcf_"+str(powerCoeff)+"-Fcf_"+str(forceCoeff)+\
                        "-Scf_"+str(smoothCoeff)+"-Icf_"+str(impactCoeff)+"-Vt_"+str(Vt)+"-Tp_"+str(Tp)+"-Ang_"+str(theta)
            # dirname = "Iarm_"+str(self.arm_I)+"-Marm_"+str(self.arm_M)
        elif method_choice==3:
            dirname = "-MPC-Pos_"+str(self.PostarCoef[1])+"-Tor_"+str(self.TorqueCoef[1])+"-DTor_"+str(self.DTorqueCoef[1]) +"-Vel_"+str(self.VeltarCoef[1])\
                    +"-dt_"+str(self.dt)+"-T_"+str(self.T)+"-Tp_"+str(self.Tp)+"-Tc_"+str(self.Nc)+"-ML_"+str(self.ML)+ "k" 

        # dirname = "-mM_"+str(m_M)
        # dirname = "-Ir_"+str(I_r)
        save_dir = self.savepath + date + dirname+ "/"

        if self.save_flag:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        return save_dir, dirname, date

    def DataPlot(self, saveflag=0):

        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ref = [0.04*0.2]*len(self.t)

        ax1.plot(self.t, self.q[:, 0], label="theta 1")
        # ax1.plot(t, q[:, 1], label="theta 2")
        ax1.plot(self.t, self.q[:, 2], label="theta 3")
        ax1.plot(self.t, ref, linestyle='--', color='r')

        ax11 = ax1.twinx()
        ax11.plot(self.t, self.q[:, 1], color='forestgreen', label="theta 2")
        # ax11.plot(t, q[:, 1], color='mediumseagreen', label="theta 2")
        ax11.legend(loc='lower right', fontsize = 12)
        ax11.yaxis.set_tick_params(labelsize = 12)

        ax1.set_ylabel('Angle ', fontsize = 15)
        ax1.xaxis.set_tick_params(labelsize = 12)
        ax1.yaxis.set_tick_params(labelsize = 12)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(self.t, self.dq[:, 0], label="theta 1 Vel")
        ax2.plot(self.t, self.dq[:, 1], label="theta 2 Vel")
        ax2.plot(self.t, self.dq[:, 2], label="theta 3 Vel")

        ax2.set_ylabel('Angular Vel ', fontsize = 15)
        ax2.xaxis.set_tick_params(labelsize = 12)
        ax2.yaxis.set_tick_params(labelsize = 12)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(self.t, self.u[:, 0], label="torque 1")
        ax3.plot(self.t, self.u[:, 1], label="torque 2")
        ax3.set_ylabel('Torque ', fontsize = 15)
        ax3.xaxis.set_tick_params(labelsize = 12)
        ax3.yaxis.set_tick_params(labelsize = 12)
        ax3.legend(loc='upper right', fontsize = 12)
        ax3.grid()

        date = self.date
        name = self.name + ".png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
    
        plt.show()

    ## Three link
    def animation(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        ## kinematic equation
        L0 = self.robot.L[0]
        L1 = self.robot.L[1]
        L2 = self.robot.L[2]
        L_max = L0+L1+L2
        x1 = L0*sin(self.q[:, 0])
        y1 = L0*cos(self.q[:, 0])
        x2 = L1*sin(self.q[:, 0] + self.q[:, 1]) + x1
        y2 = L1*cos(self.q[:, 0] + self.q[:, 1]) + y1
        x3 = L2*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + x2
        y3 = L2*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + y2

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L0, L0), ylim=(-0.05, (L0+L1)*1.2))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i], x3[i]]
            thisy = [0, y1[i], y2[i], y3[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[3])
            history_y.appendleft(thisy[3])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.1, save_count = 30, blit=True)

        ## animation save to gif
        date = self.date
        name = "traj_ani" + ".gif"

        savename = self.save_dir +date+ name

        if saveflag:
            ani.save(savename, writer='pillow', fps=30)

        # plt.show()
        
        pass
    
    ## Two Link
    def animationTwoLink(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        L = self.robot.L
        q1 = self.q[:,0]
        q2 = self.q[:,1]
        L0 = L[0]
        L1 = L[1]
        L_max = L0+L1
        x1 = L0*cos(q1)
        y1 = L0*sin(q1)
        x2 = L1*cos(q1 + q2) + x1
        y2 = L1*sin(q1 + q2) + y1

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L_max, L_max), ylim=(-0.4, (L0+L1)))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.2fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[2])
            history_y.appendleft(thisy[2])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.1, save_count = 30, blit=True)

        ## animation save to gif
        date = self.date
        name = "traj_ani" + ".gif"

        savename = self.save_dir +date+ name

        if saveflag:
            ani.save(savename, writer='pillow', fps=30)

        # plt.show()
        
        pass

    ## Three link
    def animationFourLink(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        ## kinematic equation
        L0 = self.robot.L[0]
        L1 = self.robot.L[1]
        L2 = self.robot.L[2]
        L3 = self.robot.L[3]
        L_max = L0+L1+L2
        x1 = L0*sin(self.q[:, 0])
        y1 = L0*cos(self.q[:, 0])
        x2 = L1*sin(self.q[:, 0] + self.q[:, 1]) + x1
        y2 = L1*cos(self.q[:, 0] + self.q[:, 1]) + y1
        x3 = L2*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + x2
        y3 = L2*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + y2
        x4 = L3*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]+self.q[:, 3]) + x3
        y4 = L3*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]+self.q[:, 3]) + y3

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L0, L0), ylim=(-0.05, (L0+L1)*1.2))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i], x3[i], x4[i]]
            thisy = [0, y1[i], y2[i], y3[i], y4[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[4])
            history_y.appendleft(thisy[4])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.1, save_count = 30, blit=True)

        ## animation save to gif
        date = self.date
        name = "traj_ani" + ".gif"

        savename = self.save_dir +date+ name

        if saveflag:
            ani.save(savename, writer='pillow', fps=30)

        # plt.show()
        
        pass

    def DataSave(self, saveflag):
        date = self.date
        name = self.name

        if saveflag:
            with open(self.save_dir+date+name+"-config.yaml", mode='w') as file:
                YAML().dump(self.cfg, file)
            Data = {'u': self.u, "q": self.q, "dq": self.dq, "ddq": self.ddq, "t": self.t}
            with open(os.path.join(self.save_dir, date+name+"-sol.pkl"), 'wb') as f:
                pickle.dump(Data, f)
            pass
        
        return self.save_dir
