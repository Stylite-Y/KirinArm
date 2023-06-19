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
        self.m_bd = list(cfg['Robot']['Mass']['mass'])
        self.I_bd = list(cfg['Robot']['Mass']['inertia'])
        self.g = cfg['Environment']['Gravity']
        self.massCenter = list(cfg['Robot']['Mass']['massCenter'])
        self.GeoLength = list(cfg['Robot']['Geometry']['length'])
        self.dof = len(self.GeoLength)
        self.T = cfg['Controller']['Period']
        self.Im = cfg['Robot']['Motor']['Inertia']
        self.N = cfg['Controller']['CollectionNum']
        self.dt = self.T / self.N

        self.L = cfg['Robot']['Geometry']['length']
        self.dof = len(self.L)
        self.l = cfg['Robot']['Mass']['massCenter']

        self.m = [mm[0], mm[1], self.m_bd[0]]
        self.gam = gam

        self.I = [self.m[0]*self.L[0]**2/12, self.m[1]*self.L[1]**2/12,
                    self.I_bd[0]]

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

    def MassMatrix(self, q, ifprint = False):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m_bd[0]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        I0 = self.I[0]
        I1 = self.I[1]
        I2 = self.I[2]
        gam0 = self.gam[0]
        gam1 = self.gam[1]
        gam2 = self.gam[2]
        Im = self.Im


        M11 = Im*gam0**2 + I0 + I1 + I2 + (L0**2 + L1**2 + lc2**2)*m2+\
            (L0**2 + lc1**2) * m1 + lc0**2*m0 + \
            2*L0*m2*(L1*c(q[1]) + lc2*c(q[1]+q[2])) + \
            2*L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2])

        M12 = I1 + I2 + (L1**2 + lc2**2)*m2 + lc1**2*m1 + \
            L0*L1*m2*c(q[1]) + L0*m2*lc2*c(q[1]+q[2]) + \
            L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2])

        M13 = I2 + lc2**2*m2 + L0*lc2*m2*c(q[1]+q[2]) + L1*lc2*m2*c(q[2])
        
        M21 = M12
        M22 = Im*gam1**2 + I1 + I2 + (L1**2 + lc2**2)*m2 + lc1**2*m1 + \
            2*L1*lc2*m2*c(q[2])
        M23 = I2 + lc2**2*m2 + L1*lc2*m2*c(q[2])

        M31 = M13
        M32 = M23
        M33 = I2 + Im*gam2**2 + lc2**2*m2

        # if ifprint:
        #     print(I0, I1, I2, Im)
        #     print(m0, m1, m2)
        #     print(gam0, gam1, gam2)
        #     print(((L0**2 + L1**2 + lc2**2)*m2))
        #     print( (L0**2 + lc1**2) * m1 + lc0**2*m0)
        #     print(2*L0*m2*(L1*c(q[1]) + lc2*c(q[1]+q[2])))
        #     print(2*L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2]))

        return [[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]]

    def coriolis(self, q, dq):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m_bd[0]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]

        C1 = -2*L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[1] \
            - 2*lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[2] \
            - L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[1]*dq[1] \
            - 2*lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[1]*dq[2] \
            - lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[2]*dq[2]

        C11 = -2*L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[1] \
            - L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[1]*dq[1] \
            - 2*lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[1]*dq[2]
        C12 = - 2*lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[2] \
            - lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[2]*dq[2]
        
        C21 = - 2*L1*lc2*m2*s(q[2]) * dq[1]*dq[2]
        C22 = L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[0] \
            - 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[2] \
            - L1*lc2*m2*s(q[2]) * dq[2]*dq[2]

        C31 = 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[1] \
            + L1*lc2*m2*s(q[2]) * dq[1]*dq[1]
        C32 = lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[0]

        return [C11, C21, C31], [C12, C22, C32]

    def gravity(self, q):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m_bd[0]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]

        G1 = -(L0*m1*s(q[0]) + L0*m2*s(q[0]) + L1*m2*s(q[0]+q[1]) + \
            lc0*m0*s(q[0]) + lc1*m1*s(q[0]+q[1]) + lc2*m2*s(q[0]+q[1]+q[2]))
        
        G2 = -(L1*m2*s(q[0]+q[1]) + lc1*m1*s(q[0]+q[1]) + lc2*m2*s(q[0]+q[1]+q[2]))

        G3 = -lc2*m2*s(q[0]+q[1]+q[2])

        return [G1*self.g, G2*self.g, G3*self.g]

    def inertia_force(self, q, acc):
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] for i in range(self.dof)]
        
        return inertia_force

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] for i in range(self.dof)]
        inertia_main = [mm[i][i]*acc[i] for i in range(self.dof)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(self.dof)]
        # endregion
        return inertia_main, inertia_coupling

    def get_endpos(self, q):
        L = self.GeoLength
        N = q.shape
        posx = np.zeros((self.dof, N[0]))
        posy = np.zeros((self.dof, N[0]))

        for i in range(self.dof):
            qj = 0
            for j in range(i+1):
                qj += q[:,j]
                posx[i:,] += L[j]*s(qj)
                posy[i:,] += L[j]*c(qj)
        
        return posx, posy

class DataProcess():
    def __init__(self, cfg, q, dq,ddq, u, t, gamma, m, savepath, save_flag, OutputPath='./image/', **params):
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
            'legend.fontsize': 10,
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
        fig = plt.figure(figsize=(24/2.54, 14/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35, hspace=0.3)
        g_data = gs[1].subgridspec(2, self.dof, wspace=0.35, hspace=0.5)
        ax_m = fig.add_subplot(gs[0])
        ax_v = [fig.add_subplot(g_data[0, i]) for i in range(self.dof)]
        ax_u = [fig.add_subplot(g_data[1, i]) for i in range(self.dof)]

        #! plot trajectory ------------------------------------------
        ax_m.axhline(y=-0.0, color='k', zorder=0)
        num_frame = 5
        T = self.cfg["Controller"]["Period"]
        
        lwd = [3.0, 3.0, 1.0]
        for tt in np.linspace(0, T, num_frame):
            idx = np.argmin(np.abs(self.t-tt))
            model = DynamicModel(self.cfg, self.m, self.gam)
            posx, posy = model.Posture(self.q[idx, :])
            for j in range(self.dof):
                ax_m.plot([posx[j], posx[j+1]], [posy[j], posy[j+1]], 
                        'o-', ms=3, color=cmap(tt/T), alpha=tt/T*0.8+0.2, lw=lwd[j])
            pass

        ax_m.axis('equal')
        ax_m.text(-0.1, 1.03, r"$\mathbf{A}$",transform=ax_m.transAxes)
        ax_m.set_xlabel('x (m)')
        ax_m.set_ylabel('y (m)')

        #! plot vel -------------------------------------------
        titlelabel_v = [r"$\dot{\theta}_s\ (rad/s)$",r"$\dot{\theta}_e\ (rad/s)$",r"$\dot{\theta}_w\ (rad/s)$"]
        axislabel = ["t (s)","angular vel (rad/s)"]
        index = [r"$\mathbf{B}$",r"$\mathbf{C}$",r"$\mathbf{D}$"]
        [ax_v[i].plot(self.t, self.dq[:, i], color='C0') for i in range(self.dof)]
        [ax_v[i].set_title(titlelabel_v[i]) for i in range(self.dof)]
        [ax_v[i].set_xlim([0,T]) for i in range(self.dof)]
        ax_v[0].set_ylabel(axislabel[1])
        [ax_v[i].text(-0.1,1.1,index[i],transform=ax_v[i].transAxes) for i in range(self.dof)]

        #! plot force -------------------------------------------
        titlelabel_u = [r"$\tau_s\ (Nm)$",r"$\tau_e\ (N.m)$",r"$\tau_w\ (N.m)$"]
        index = [r"$\mathbf{E}$",r"$\mathbf{F}$",r"$\mathbf{G}$"]
        axislabel = ["t (s)","torque (N.m)"]
        [ax_u[i].plot(self.t, self.u[:, i], color='C0') for i in range(self.dof)]
        [ax_u[i].set_title(titlelabel_u[i]) for i in range(self.dof)]
        [ax_u[i].set_xlim([0,T]) for i in range(self.dof)]
        [ax_u[i].set_xlabel(axislabel[0]) for i in range(self.dof)]
        ax_u[0].set_ylabel(axislabel[1])
        [ax_u[i].text(-0.1,1.1,index[i],transform=ax_u[i].transAxes) for i in range(self.dof)]

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/traj_opt.png", dpi=600)
        else:
            return fig

    def AccRefCmp(self, **params):
        dq_r = np.asarray(params["dq_r"])
        ddq_r = np.asarray(params["ddq_r"])

        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(24/2.54, 14/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35, hspace=0.3)
        g_data = gs[1].subgridspec(2, self.dof, wspace=0.35, hspace=0.5)
        ax_m = fig.add_subplot(gs[0])
        ax_v = [fig.add_subplot(g_data[0, i]) for i in range(self.dof)]
        ax_a = [fig.add_subplot(g_data[1, i]) for i in range(self.dof)]

        #! plot trajectory ------------------------------------------
        ax_m.axhline(y=-0.0, color='k', zorder=0)
        num_frame = 5
        T = self.cfg["Controller"]["Period"]
        
        lwd = [3.0, 3.0, 1.0]
        for tt in np.linspace(0, T, num_frame):
            idx = np.argmin(np.abs(self.t-tt))
            model = DynamicModel(self.cfg, self.m, self.gam)
            posx, posy = model.Posture(self.q[idx, :])
            for j in range(self.dof):
                ax_m.plot([posx[j], posx[j+1]], [posy[j], posy[j+1]], 
                        'o-', ms=3, color=cmap(tt/T), alpha=tt/T*0.8+0.2, lw=lwd[j])
            pass

        ax_m.axis('equal')
        ax_m.text(-0.1, 1.03, r"$\mathbf{A}$",transform=ax_m.transAxes)
        ax_m.set_xlabel('x (m)')
        ax_m.set_ylabel('y (m)')

        #! plot vel -------------------------------------------
        titlelabel_v = [r"$\dot{\theta}_s\ (rad/s)$",r"$\dot{\theta}_e\ (rad/s)$",r"$\dot{\theta}_w\ (rad/s)$"]
        axislabel = ["t (s)", "angular vel (rad/s)"]
        index = [r"$\mathbf{B}$",r"$\mathbf{C}$",r"$\mathbf{D}$"]
        [ax_v[i].plot(self.t, self.dq[:, i], color='C0', label = "opt traj vel") for i in range(self.dof)]
        [ax_v[i].plot(self.t, dq_r[i, :], color='C1', label = "ref traj vel") for i in range(self.dof)]
        [ax_v[i].set_title(titlelabel_v[i]) for i in range(self.dof)]
        [ax_v[i].set_xlim([0,T]) for i in range(self.dof)]
        ax_v[0].set_ylabel(axislabel[1])
        [ax_v[i].text(-0.1,1.1,index[i],transform=ax_v[i].transAxes) for i in range(self.dof)]
        [ax_v[i].legend(fontsize=10) for i in range(self.dof)]

        #! plot acc -------------------------------------------
        titlelabel_v = [r"$\ddot{\theta}_s\ (rad/s^2)$",r"$\ddot{\theta}_e\ (rad/s^2)$",r"$\ddot{\theta}_w\ (rad/s^2)$"]
        axislabel = ["t (s)", "angular acc (rad/s.2)"]
        index = [r"$\mathbf{E}$",r"$\mathbf{F}$",r"$\mathbf{G}$"]
        [ax_a[i].plot(self.t, self.ddq[:, i], color='C0', label = "opt traj acc") for i in range(self.dof)]
        [ax_a[i].plot(self.t, ddq_r[i, :], color='C1', label = "ref traj acc") for i in range(self.dof)]
        [ax_a[i].set_title(titlelabel_v[i]) for i in range(self.dof)]
        [ax_a[i].set_xlim([0,T]) for i in range(self.dof)]
        ax_a[0].set_ylabel(axislabel[1])
        [ax_a[i].text(-0.1,1.1,index[i],transform=ax_v[i].transAxes) for i in range(self.dof)]
        [ax_a[i].legend(fontsize=10) for i in range(self.dof)]
        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/AccRef.png", dpi=600)
        else:
            return fig

    def ParamsCalAndPlot(self, **params):
        Lam = params["Lam"]
        EndVel = params["EndVel"]

        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(24/2.54, 14/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35, hspace=0.3)
        g_data = gs[1].subgridspec(2, self.dof, wspace=0.6, hspace=0.5)
        ax_m = fig.add_subplot(gs[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(self.dof)]
        ax_L = [fig.add_subplot(g_data[1, i]) for i in range(self.dof)]

        #! plot trajectory ------------------------------------------
        ax_m.axhline(y=-0.0, color='k', zorder=0)
        num_frame = 5
        T = self.cfg["Controller"]["Period"]

        lwd = [3.0, 3.0, 1.0]
        for tt in np.linspace(0, T, num_frame):
            idx = np.argmin(np.abs(self.t-tt))
            model = DynamicModel(self.cfg, self.m, self.gam)
            posx, posy = model.Posture(self.q[idx, :])
            for j in range(self.dof):
                ax_m.plot([posx[j], posx[j+1]], [posy[j], posy[j+1]], 
                        'o-', ms=3, color=cmap(tt/T), alpha=tt/T*0.8+0.2, lw=lwd[j])
            pass

        ax_m.axis('equal')
        ax_m.text(-0.1, 1.03, r"$\mathbf{A}$",transform=ax_m.transAxes)
        ax_m.set_xlabel('x (m)')
        ax_m.set_ylabel('y (m)')

        #! plot vel -------------------------------------------
        titlelabel_p = [r"$\theta_s\ (rad)$",r"$\theta_e\ (rad)$",r"$\theta_w\ (rad)$"]
        axislabel = ["t (s)","angle (rad)"]
        index = [r"$\mathbf{B}$",r"$\mathbf{C}$",r"$\mathbf{D}$"]
        [ax_p[i].plot(self.t, self.q[:, i], color='C0') for i in range(self.dof)]
        [ax_p[i].set_title(titlelabel_p[i]) for i in range(self.dof)]
        [ax_p[i].set_xlim([0,T]) for i in range(self.dof)]
        ax_p[0].set_ylabel(axislabel[1])
        [ax_p[i].text(-0.1,1.1,index[i],transform=ax_p[i].transAxes) for i in range(self.dof)]

        #! plot force -------------------------------------------
        titlelabel_u = [r"$EndVel$",r"$H$",r"$ $"]
        index = [r"$\mathbf{E}$",r"$\mathbf{F}$",r"$\mathbf{G}$"]
        axislabel = ["t (s)","Vel (m/s)", "Ang-Momt (kg.m/s)"]
        ax_L[0].plot(self.t, EndVel, color='C0')
        ax_L[1].plot(self.t, Lam, color='C0')
        [ax_L[i].set_title(titlelabel_u[i]) for i in range(self.dof)]
        [ax_L[i].set_xlim([0,T]) for i in range(self.dof)]
        [ax_L[i].set_xlabel(axislabel[0]) for i in range(self.dof)]
        ax_L[0].set_ylabel(axislabel[1])
        ax_L[1].set_ylabel(axislabel[2])
        [ax_L[i].text(-0.1,1.1,index[i],transform=ax_L[i].transAxes) for i in range(self.dof)]

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/Params.png", dpi=600)
        else:
            return fig

    def VelAndMomtAnlysis(self, **params):
        Vend = np.asarray(params["Vend"])
        H = np.asarray(params["H"])

        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(24/2.54, 14/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35, hspace=0.3)
        g_data = gs[1].subgridspec(2, 2, wspace=0.6, hspace=0.5)
        ax_m = fig.add_subplot(gs[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(2)]
        ax_L = [fig.add_subplot(g_data[1, i]) for i in range(2)]

        #! plot trajectory ------------------------------------------
        ax_m.axhline(y=-0.0, color='k', zorder=0)
        num_frame = 5
        T = self.cfg["Controller"]["Period"]

        lwd = [3.0, 3.0, 1.0]
        for tt in np.linspace(0, T, num_frame):
            idx = np.argmin(np.abs(self.t-tt))
            model = DynamicModel(self.cfg, self.m, self.gam)
            posx, posy = model.Posture(self.q[idx, :])
            for j in range(self.dof):
                ax_m.plot([posx[j], posx[j+1]], [posy[j], posy[j+1]], 
                        'o-', ms=3, color=cmap(tt/T), alpha=tt/T*0.8+0.2, lw=lwd[j])
            pass

        ax_m.axis('equal')
        ax_m.text(-0.1, 1.03, r"$\mathbf{A}$",transform=ax_m.transAxes)
        ax_m.set_xlabel('x (m)')
        ax_m.set_ylabel('y (m)')

        #! plot vel -------------------------------------------
        linklabel = ["UpperArm", "Forearm", "Racket"]
        titlelabel_p = [r"$V_x\ (m/s)$",r"$V_y\ (m/s)$"]
        axislabel = ["t (s)","Vel (m/s)"]
        index = [r"$\mathbf{B}$",r"$\mathbf{C}$"]
        [ax_p[i].plot(self.t, Vend[i, :, 0], color='C0', label = linklabel[0]) for i in range(2)]
        [ax_p[i].plot(self.t, Vend[i, :, 1], color='C1', label = linklabel[1]) for i in range(2)]
        [ax_p[i].plot(self.t, Vend[i, :, 2], color='C2', label = linklabel[2]) for i in range(2)]
        [ax_p[i].set_title(titlelabel_p[i]) for i in range(2)]
        [ax_p[i].set_xlim([0,T]) for i in range(2)]
        ax_p[0].set_ylabel(axislabel[1])
        [ax_p[i].text(-0.1,1.1,index[i],transform=ax_p[i].transAxes) for i in range(2)]
        [ax_p[i].legend() for i in range(2)]

        #! plot force -------------------------------------------
        titlelabel_u = [r"$Vel Comp$",r"$Ang Comp$"]
        index = [r"$\mathbf{D}$",r"$\mathbf{E}$"]
        axislabel = ["t (s)","Momt (kg.m/s)"]
        [ax_L[i].plot(self.t, H[i, :, 0], color='C0', label = linklabel[0]) for i in range(2)]
        [ax_L[i].plot(self.t, H[i, :, 1], color='C1', label = linklabel[1]) for i in range(2)]
        [ax_L[i].plot(self.t, H[i, :, 2], color='C2', label = linklabel[2]) for i in range(2)]
        [ax_L[i].set_title(titlelabel_u[i]) for i in range(2)]
        [ax_L[i].set_xlim([0,T]) for i in range(2)]
        [ax_L[i].set_xlabel(axislabel[0]) for i in range(2)]
        ax_L[0].set_ylabel(axislabel[1])
        [ax_L[i].text(-0.1,1.1,index[i],transform=ax_L[i].transAxes) for i in range(2)]
        [ax_L[i].legend() for i in range(2)]

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/VelAndMomt.png", dpi=600)
        else:
            return fig

    def MultiMotorParams(self, **params):
        Lam = params["Lam"]
        EndVel = params["EndVel"]
        MoterNum = params["MoterNum"]
        T = self.cfg["Controller"]["Period"]

        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(24/2.54, 14/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(1, 1)
        g_data = gs[0].subgridspec(1, self.dof-1, wspace=0.6, hspace=0.5)
        ax_L = [fig.add_subplot(g_data[0, i]) for i in range(self.dof-1)]

        #! plot force -------------------------------------------
        titlelabel_u = [r"$EndVel$",r"$H$",]
        index = [r"$\mathbf{A}$",r"$\mathbf{B}$"]
        axislabel = ["t (s)","Vel (m/s)", "Ang-Momt (kg.m/s)"]
        legendlable = ["Ours", "Tmotor RI80", "Tmotor RI100" , "Maxon EC90 flat",
                        "KM109", "MF012"]
        [ax_L[0].plot(self.t, EndVel[i], label = legendlable[i]) for i in range(MoterNum)]
        [ax_L[1].plot(self.t, Lam[i], label = legendlable[i]) for i in range(MoterNum)]
        [ax_L[i].set_title(titlelabel_u[i]) for i in range(self.dof-1)]
        [ax_L[i].set_xlim([0,T]) for i in range(self.dof-1)]
        [ax_L[i].set_xlabel(axislabel[0]) for i in range(self.dof-1)]
        ax_L[0].set_ylabel(axislabel[1])
        ax_L[1].set_ylabel(axislabel[2])
        [ax_L[i].text(-0.1,1.1,index[i],transform=ax_L[i].transAxes) for i in range(self.dof-1)]
        [ax_L[i].legend(fontsize=15) for i in range(2)]

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/MultiMotor.png", dpi=600)
        else:
            return fig

    def ForceComponentAnalysis(self, **param):
        # if date is None:
        #     #! 解直接通过param传递，不需要从硬盘中进行读取
        #     assert param['cfg']!=None, " the configured yaml file must be transferred via param"
        #     cfg = param['cfg']
        # else:
        #     #! 解保存在硬盘中，需要进行读取
        #     solution_file = StorePath + date + "_sol.npy"
        #     config_file = StorePath + date + "config.yaml"
        #     solution = np.load(solution_file)
        #     cfg = YAML().load(open(config_file, 'r'))
        #     pass
        m = param['m']
        gam = param['gam']
        T = self.cfg["Controller"]["Period"]
        Im = self.cfg['Robot']['Motor']['Inertia']
        robot = DynamicModel(self.cfg, mm=m, gam=gam)  # create robot with model
        # data = SolutionData(old_solution=solution)

        #! calculate force
        # region
        Inertia_main = []
        Inertia_coupling = []
        Corialis1 = []
        Corialis2 = []
        Gravity = []
        Control = self.u
        
        for i in range(robot.N):
            temp1, temp2 = robot.inertia_force2(self.q[i], self.ddq[i])
            Inertia_main.append(temp1)
            if i ==0:
                M = robot.MassMatrix(self.q[i], ifprint=True)
                # print(M[0][0])
            Inertia_coupling.append(temp2)
            C1, C2 = robot.coriolis(self.q[i], self.dq[i])
            Corialis1.append(C1)
            Corialis2.append(C2)
            Gravity.append(robot.gravity(self.q[i]))
            pass
        Force = [np.asarray(temp) for temp in [Inertia_main, Inertia_coupling,
                                            Corialis1, Corialis2, Gravity, Control]]
        Force[5] = -Force[5]
        # endregion

        #! start visualization
        # region
        plt.style.use("science")
        cmap = mpl.cm.get_cmap('Paired')
        params = {
            'text.usetex': True,
            'font.size': 15,
            'axes.titlesize': 20,
            'legend.fontsize': 10,
            'axes.labelsize': 20,
            'lines.linewidth': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'axes.titlepad': 3.0,
            'axes.labelpad': 3.0,
            'lines.markersize': 8,
            'figure.subplot.wspace': 0.4,
            'figure.subplot.hspace': 0.8,
            "pgf.preamble": "\n".join([
            r"\usepackage{xcolor}",
            r"\setmainfont{DejaVu Serif}"])}
        mpl.rcParams.update(params)

        fig_zero_holder = plt.figure(figsize=(24/2.54, 14/2.54))
        ax_zero_holder = fig_zero_holder.subplots(1, 3, sharex=True)
        ColorCandidate = ['C'+str(i) for i in range(7)]

        title = [r"$\tau_{shoulder}\ (\mathrm{Nm})$",
                r"$\tau_{elbow}\ (\mathrm{Nm})$",
                r"$\tau_{wrist}\ (\mathrm{N})$"]
        
        index = [r"$\mathbf{A}$",
                r"$\mathbf{B}$",
                r"$\mathbf{C}$"]
        
        dofdof_id = [0, 1, 2]

        for i in range(3):
            ax_zero_holder[i].set_title(title[i])
            ax_zero_holder[i].set_xlabel('t(s)')
            ax_zero_holder[i].text(-0.15, 1.05, index[i],
                                        transform=ax_zero_holder[i].transAxes)
            outline1 = []
            outline2 = []

            for k in range(robot.N):
                dof_id = dofdof_id[i]
                pos = 0
                neg = 0
                for kk in range(6):
                    temp = 0
                    temp += Force[kk][k, dof_id]
                    ax_zero_holder[i].bar(self.t[k], temp, width=robot.dt,
                                                bottom=pos if temp >= 0 else neg, align='edge',
                                                color=ColorCandidate[kk], linewidth=0, ecolor=ColorCandidate[kk])
                    if temp >= 0:
                        pos += temp
                        pass
                    else:
                        neg += temp
                        pass
                    pass
                outline1.append(pos)
                outline2.append(neg)
                pass

            # ax_zero_holder[0, i].plot(self.t, -np.asarray(outline2), color='silver', lw=0.5, ls='--')
            # ax_zero_holder[0, i].plot(self.t, -np.asarray(outline1), color='silver', lw=0.5, ls='--')
            pass

        [a.set_xlim([0, T]) for a in ax_zero_holder.reshape(-1)]

        UB = [250, 100, 50]
        LB = [-250, -100, -50]
        
        [ax_zero_holder[i].set_ylim([LB[i], UB[i]]) for i in range(3)]

        dynamics_formula = [r'$M_{ii}\ddot{q}_i$', r'$+$',
                            r'$\sum M_{ij}\ddot{q}_j$', r'+',
                            r'$C_1(q,\dot{q})\dot{q}_1$', r'+',
                            r'$C_2(q,\dot{q})$', r'+',
                            r'$G({q})$', r'+',
                            r'$-\tau_a$', r'$=0$']
        color = ['C0', 'k', 'C1', 'k', 'C2', 'k', 'C3', 'k', 'C4', 'k', 'C5', 'k']

        offset = 0.0
        for s, c in zip(dynamics_formula, color):
            text = fig_zero_holder.text(0.22+offset,0.95,s,color=c)
            text.draw(fig_zero_holder.canvas.get_renderer())
            ex = text.get_window_extent()
            offset += ex.width / 800
            # offset += text.get_fontweight()
            pass
        
        fig_zero_holder.subplots_adjust(left=0.1,
                                        bottom=0.2, 
                                        right=0.9, 
                                        top=0.85, 
                                        wspace=0.3, 
                                        hspace=0.25)
        
        # endregion

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig_zero_holder.savefig(savedir + "/ForceComponents.png", dpi=600)
        else:
            return fig_zero_holder

        pass

    def CostFun(self, **params):
        CostFun = params['CostFun']
        CostSum = CostFun[0]
        Pw = CostFun[1]
        F = CostFun[2]
        Pos = CostFun[3]
        Vel = CostFun[4]
        Momt = CostFun[5]

        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(24/2.54, 14/2.54))
        # fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35, hspace=0.3)
        g_data = gs[1].subgridspec(2, self.dof, wspace=0.35, hspace=0.5)
        ax_m = fig.add_subplot(gs[0])
        ax_v = [fig.add_subplot(g_data[0, i]) for i in range(self.dof)]
        ax_u = [fig.add_subplot(g_data[1, i]) for i in range(self.dof)]

        #! plot trajectory ------------------------------------------
        ax_m.axhline(y=-0.0, color='k', zorder=0)
        num_frame = 5
        T = self.cfg["Controller"]["Period"]
        
        lwd = [3.0, 3.0, 1.0]
        for tt in np.linspace(0, T, num_frame):
            idx = np.argmin(np.abs(self.t-tt))
            model = DynamicModel(self.cfg, self.m, self.gam)
            posx, posy = model.Posture(self.q[idx, :])
            for j in range(self.dof):
                ax_m.plot([posx[j], posx[j+1]], [posy[j], posy[j+1]], 
                        'o-', ms=3, color=cmap(tt/T), alpha=tt/T*0.8+0.2, lw=lwd[j])
            pass

        ax_m.axis('equal')
        ax_m.text(-0.1, 1.03, r"$\mathbf{A}$",transform=ax_m.transAxes)
        ax_m.set_xlabel('x (m)')
        ax_m.set_ylabel('y (m)')

        #! plot vel -------------------------------------------
        titlelabel_v = [r"$CostSum$",r"$Power$",r"$Force$"]
        axislabel = ["t (s)","angular vel (rad/s)"]
        index = [r"$\mathbf{B}$",r"$\mathbf{C}$",r"$\mathbf{D}$"]
        [ax_v[i].plot(self.t, CostFun[i], color='C0') for i in range(self.dof)]
        [ax_v[i].set_title(titlelabel_v[i]) for i in range(self.dof)]
        [ax_v[i].set_xlim([0,T]) for i in range(self.dof)]
        # ax_v[0].set_ylabel(axislabel[1])
        [ax_v[i].text(-0.1,1.1,index[i],transform=ax_v[i].transAxes) for i in range(self.dof)]

        #! plot force -------------------------------------------
        titlelabel_u = [r"$Pos$",r"$Vel$",r"$Momt$"]
        index = [r"$\mathbf{E}$",r"$\mathbf{F}$",r"$\mathbf{G}$"]
        axislabel = ["t (s)","torque (N.m)"]
        [ax_u[i].plot(self.t, CostFun[i+3], color='C0') for i in range(self.dof)]
        [ax_u[i].set_title(titlelabel_u[i]) for i in range(self.dof)]
        [ax_u[i].set_xlim([0,T]) for i in range(self.dof)]
        [ax_u[i].set_xlabel(axislabel[0]) for i in range(self.dof)]
        # ax_u[0].set_ylabel(axislabel[1])
        [ax_u[i].text(-0.1,1.1,index[i],transform=ax_u[i].transAxes) for i in range(self.dof)]

        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            fig.savefig(savedir + "/CostFun.png", dpi=600)
        else:
            return fig


    def animation(self):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        ## kinematic equatio
        # L_max = L0+L1+L2
        L = self.cfg['Robot']['Geometry']['length']
        # model = DynamicModel(self.cfg, self.m, self.gam)
        # posx, posy = model.get_endpos(self.q)
        x1 = L[0]*sin(self.q[:, 0])
        y1 = L[0]*cos(self.q[:, 0])
        x2 = L[1]*sin(self.q[:, 0] + self.q[:, 1]) + x1
        y2 = L[1]*cos(self.q[:, 0] + self.q[:, 1]) + y1
        x3 = L[2]*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + x2
        y3 = L[2]*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + y2

        T = self.cfg["Controller"]["Period"]
        Num = self.cfg["Controller"]["CollectionNum"]
        dt = T/Num

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-1.0, 0.2), ylim=(-0.75, 1.3))
        # ax = fig.add_subplot(autoscale_on=False)
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line1, = ax.plot([], [], 'o-', lw=6, markersize=10)
        line2, = ax.plot([], [], 'o-', lw=2, markersize=10)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            # thisx = [0, posx[0,i], posx[1,i], posx[2,i]]
            # thisy = [0, posy[0,i], posy[1,i], posy[2,i]]
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]
            thisx2 = [x2[i], x3[i]]
            thisy2 = [y2[i], y3[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx2[1])
            history_y.appendleft(thisy2[1])

            alpha = (i / history_len) ** 2
            line1.set_data(thisx, thisy)
            line2.set_data(thisx2, thisy2)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*dt))
            return line1, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.0001, blit=True)

        ## animation save to gif
        if self.save_flag:
            todaytime=datetime.date.today()
            savedir = self.OutputPath + str(todaytime) + '/'+self.date
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            ani.save(savedir + "/traj_ani.gif", writer='pillow')
        else:
            return ani

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
