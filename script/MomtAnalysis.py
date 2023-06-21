import numpy as np
import matplotlib as mpl
from numpy import sin  as s
from numpy import cos as c
import matplotlib.pyplot as plt

class ParamsCal():
    def __init__(self, dof, m, L=[0.35, 0.35], l=[0.175, 0.175]):
        self.dof = dof
        self.L = L
        self.l = l
        self.m = m
        self.I = [m[i]*L[i]**2/12 for i in range(3)]
        self.g = 9.8
        pass

    def MassMatrix(self, q):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        I0 = self.I[0]
        I1 = self.I[1]
        I2 = self.I[2]


        M11 = I0 + I1 + I2 + (L0**2 + L1**2 + lc2**2)*m2+\
            (L0**2 + lc1**2) * m1 + lc0**2*m0 + \
            2*L0*m2*(L1*c(q[1]) + lc2*c(q[1]+q[2])) + \
            2*L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2])

        M12 = I1 + I2 + (L1**2 + lc2**2)*m2 + lc1**2*m1 + \
            L0*L1*m2*c(q[1]) + L0*m2*lc2*c(q[1]+q[2]) + \
            L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2])

        M13 = I2 + lc2**2*m2 + L0*lc2*m2*c(q[1]+q[2]) + L1*lc2*m2*c(q[2])
        
        M21 = M12
        M22 = I1 + I2 + (L1**2 + lc2**2)*m2 + lc1**2*m1 + \
            2*L1*lc2*m2*c(q[2])
        M23 = I2 + lc2**2*m2 + L1*lc2*m2*c(q[2])

        M31 = M13
        M32 = M23
        M33 = I2 + lc2**2*m2

        return [[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]]

    def coriolis(self, q, dq):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
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
        m2 = self.m[2]
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

    def get_endvel(self, q, dq):
        L = self.L

        vx = 0
        vy = 0
        for i in range(self.dof):
            qj = 0
            dqj = 0
            for j in range(i+1):
                qj += q[j]
                dqj += dq[j]
            vx += L[i]*np.cos(qj)*dqj
            vy += -L[i]*np.sin(qj)*dqj

        return vx, vy
    
    def get_endvel2(self, q, dq):
        L = self.L

        vx = L[0]*c(q[0])*dq[0] + L[1]*c(q[0]+q[1])*(dq[0]+dq[1]) + \
            L[2]*c(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])
        vy = -L[0]*s(q[0])*dq[0] - L[1]*s(q[0]+q[1])*(dq[0]+dq[1]) - \
            L[2]*s(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])

        return vx, vy

    def get_comvel(self, q, dq):
        L = self.L
        l = self.l
        v = [[0 for i in range(2)] for j in range(self.dof)]

        for i in range(self.dof):
            qj = 0
            dqj = 0
            for j in range(i+1):
                qj += q[j]
                dqj += dq[j]
                if j == i:
                    v[i][0] += l[j]*np.cos(qj)*dqj
                    v[i][1] += -l[j]*np.sin(qj)*dqj
                else:
                    v[i][0] += L[j]*np.cos(qj)*dqj
                    v[i][1] += -L[j]*np.sin(qj)*dqj
        
        return v

    def get_comvel2(self, q, dq):
        L = self.L
        l = self.l
        v = [[0 for i in range(2)] for j in range(self.dof)]
        
        v[0][0] = l[0]*c(q[0])*dq[0]
        v[0][1] = -l[0]*s(q[0])*dq[0]

        v[1][0] = L[0]*c(q[0])*dq[0] + l[1]*c(q[0]+q[1])*(dq[0]+dq[1])
        v[1][1] = -L[0]*s(q[0])*dq[0] - l[1]*s(q[0]+q[1])*(dq[0]+dq[1])

        v[2][0] = L[0]*c(q[0])*dq[0] + L[1]*c(q[0]+q[1])*(dq[0]+dq[1]) + \
            l[2]*c(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])
        v[2][1] = -L[0]*s(q[0])*dq[0] - L[1]*s(q[0]+q[1])*(dq[0]+dq[1]) - \
            l[2]*s(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])


        return v

    def get_compos(self, q):
        L = self.L
        l = self.l
        pos = [[0 for i in range(2)] for j in range(self.dof)]

        for i in range(self.dof):
            qj = 0
            for j in range(i+1):
                qj += q[j]
                if j==i:
                    pos[i][0] += l[j]*np.sin(qj)
                    pos[i][1] += l[j]*np.cos(qj)
                else:
                    pos[i][0] += L[j]*np.sin(qj)
                    pos[i][1] += L[j]*np.cos(qj)

        return pos
    
    def get_compos2(self, q):
        L = self.L
        l = self.l
        pos = [[0 for i in range(2)] for j in range(self.dof)]

        pos[0][0] = l[0]*s(q[0])
        pos[0][1] = l[0]*c(q[0])

        pos[1][0] = L[0]*s(q[0]) + l[1]*s(q[0]+q[1])
        pos[1][1] = L[0]*c(q[0]) + l[1]*c(q[0]+q[1])

        pos[2][0] = L[0]*s(q[0]) + L[1]*s(q[0]+q[1]) + l[2]*s(q[0]+q[1]+q[2])
        pos[2][1] = L[0]*c(q[0]) + L[1]*c(q[0]+q[1]) + l[2]*c(q[0]+q[1]+q[2])

        return pos


    def get_AngMomt(self, q, dq):
        L = self.L
        m = self.m
        I = [m[i]*L[i]**2/12 for i in range(3)]

        ri = self.get_compos(q)
        v = self.get_comvel(q, dq)

        H = m[0]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]) + I[0]*dq[0] + \
            m[1]*(ri[1][0]*v[1][1]-ri[1][1]*v[1][0]) + I[1]*(dq[0]+dq[1]) +\
            m[2]*(ri[2][0]*v[2][1]-ri[2][1]*v[2][0]) + I[2]*(dq[0]+dq[1]+dq[2])

        Hm = [m[0]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]), 
                    m[1]*(ri[1][0]*v[1][1]-ri[1][1]*v[1][0]),
                    m[2]*(ri[2][0]*v[2][1]-ri[2][1]*v[2][0])]
        HI = [I[0]*dq[0], I[1]*(dq[0]+dq[1]), I[2]*(dq[0]+dq[1]+dq[2])]

        return H, Hm, HI


def main():
    dof = 3
    Length = [0.35, 0.35, 0.55]
    lc = [0.175, 0.175, 0.3]
    q = [0.1*np.pi, 0.05*np.pi, 0.01*np.pi]
    ddq = [1.0, 1.0, 1.0]
    
    p = np.linspace(1.0, 20.0, 20)
    H_dq = []
    Hm_dq = []
    HI_dq = []
    t_dq = []
    for i in range(3):
        dq = [1.0, 1.0, 1.0]
        m = [1.0, 1.0, 0.09]
        H_tmp = []
        Hm_tmp = []
        HI_tmp = []
        t_tmp = []
        for j in range(len(p)):
            dq[i] = p[j]
            print(dq)
            Params = ParamsCal(dof, m, L=Length, l=lc)
            H, Hm, HI = Params.get_AngMomt(q, dq)
            # dynamic
            Inertia_main, Inertia_coupling = Params.inertia_force2(q, ddq)
            Corialis1, Corialis2 = Params.coriolis(q, dq)
            Gravity=Params.gravity(q)
            tau = [Inertia_main[i] + Inertia_coupling[i] + 
                Corialis1[i] + Corialis2[i] +Gravity[i] for i in range(3)]
            # print(tau)
            t_tmp.append(tau)
            H_tmp.append(H)
            Hm_tmp.append(Hm)
            HI_tmp.append(HI)

        H_dq.append(H_tmp)
        Hm_dq.append(Hm_tmp)
        HI_dq.append(HI_tmp)
        t_dq.append(t_tmp)

    H_m = []
    Hm_m = []
    HI_m = []
    t_m = []
    for i in range(2):
        dq = [1.0, 1.0, 1.0]
        m = [1.0, 1.0, 0.09]
        H_tmp = []
        Hm_tmp = []
        HI_tmp = []
        t_tmp = []
        for j in range(len(p)):
            m[i] = p[j]
            print(m)
            Params = ParamsCal(dof, m, L=Length, l=lc)
            H, Hm, HI = Params.get_AngMomt(q, dq)
            Inertia_main, Inertia_coupling = Params.inertia_force2(q, ddq)
            Corialis1, Corialis2 = Params.coriolis(q, dq)
            Gravity=Params.gravity(q)
            tau = [Inertia_main[i] + Inertia_coupling[i] + 
                Corialis1[i] + Corialis2[i] +Gravity[i] for i in range(3)]
            t_tmp.append(tau)
            H_tmp.append(H)
            Hm_tmp.append(Hm)
            HI_tmp.append(HI)

        H_m.append(H_tmp)
        Hm_m.append(Hm_tmp)
        HI_m.append(HI_tmp)
        t_m.append(t_tmp)

    t_dq = np.asarray(t_dq)
    t_m = np.asarray(t_m)
    print(t_dq.shape, t_m.shape)
    params = {
        'text.usetex': True,
        'font.size': 15,
        'axes.titlesize': 20,
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

    plt.rcParams.update(params)
    cmap = mpl.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(24/2.54, 18/2.54))
    # fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(1, 1)
    g_data = gs[0].subgridspec(2, 2, wspace=0.4, hspace=0.6)
    ax_L = [fig.add_subplot(g_data[0, i]) for i in range(dof-1)]
    ax_m = [fig.add_subplot(g_data[1, i]) for i in range(dof-1)]

    #! plot force -------------------------------------------
    titlelabel_u = ["Ang-vel", "Mass"]
    titlelabel_m = [r"$\dot{\theta}$",r"$m$"]
    anglabel = ["shoulder", "elbow", "wrist"]
    linklabel = ["Upper Arm", "Forearm"]
    index = [r"$\mathbf{A}$",r"$\mathbf{B}$"]
    axislabel = ["t (s)", "H (kg.m/s)"]
    [ax_L[0].plot(p, H_dq[i], label = anglabel[i]) for i in range(dof)]
    [ax_L[1].plot(p, H_m[i], label = linklabel[i]) for i in range(dof-1)]
    [ax_L[i].set_title(titlelabel_u[i]) for i in range(2)]
    [ax_L[i].set_xlim([0,20]) for i in range(2)]
    [ax_L[i].set_ylim([-10,0]) for i in range(2)]
    [ax_L[i].set_xlabel(titlelabel_m[i]) for i in range(2)]
    ax_L[0].set_ylabel(axislabel[1])
    [ax_L[i].text(-0.1,1.1,index[i],transform=ax_L[i].transAxes) for i in range(2)]
    [ax_L[i].legend(fontsize=15) for i in range(2)]

    index = [r"$\mathbf{C}$",r"$\mathbf{D}$"]
    axislabel = ["t (s)", r"$\tau_{Elbow} (N.m)$"]
    [ax_m[0].plot(p, t_dq[i,:,1], label = anglabel[i]) for i in range(dof)]
    [ax_m[1].plot(p, t_m[i,:,1], label = linklabel[i]) for i in range(dof-1)]
    [ax_m[i].set_title(titlelabel_u[i]) for i in range(dof-1)]
    [ax_m[i].set_xlim([0,20]) for i in range(dof-1)]
    # [ax_m[i].set_ylim([-10,0]) for i in range(dof-1)]
    [ax_m[i].set_xlabel(titlelabel_m[i]) for i in range(dof-1)]
    ax_m[0].set_ylabel(axislabel[1])
    [ax_m[i].text(-0.1,1.1,index[i],transform=ax_m[i].transAxes) for i in range(dof-1)]
    [ax_m[i].legend(fontsize=15) for i in range(2)]

    fig.savefig("./image/MomtAnalysis/Momt_w.png", dpi=600)
    plt.show()
    pass

def main2():
    dof = 3
    L = [0.35, 0.35, 0.55]
    l = [0.175, 0.175, 0.3]
    q = [0.1*np.pi, 0.05*np.pi, 0.01*np.pi]
    ddq = [1.0, 1.0, 1.0]
    
    dq = [1.0, 1.0, 1.0]
    m = [1.0, 1.0, 0.09]
    Params = ParamsCal(dof, m, L=L, l=l)
    H, Hm, HI = Params.get_AngMomt(q, dq)

    endvel1 = Params.get_compos(q)
    endvel2 = Params.get_compos2(q)
    print(endvel1, endvel2)

    # Hm1 = m[0]*(l[0]*s(q[0])*(-l[0]*s(q[0]))-l[0]*c(q[0])*(l[0]*c(q[0])))*dq[0]

    # Hm21 = m[1]*((L[0]*s(q[0]) + l[1]*s(q[0]+q[1]))*((-L[0]*s(q[0])-l[1]*s(q[0]+q[1])))-\
    #         (L[0]*c(q[0]) + l[1]*c(q[0]+q[1]))*(L[0]*c(q[0])+l[1]*c(q[0]+q[1])))*dq[0]
    # Hm22 = m[1]*((L[0]*s(q[0]) + l[1]*s(q[0]+q[1]))*(-l[1]*s(q[0]+q[1]))-\
    #         (L[0]*c(q[0]) + l[1]*c(q[0]+q[1]))*(l[1]*c(q[0]+q[1])))*dq[1]

    # Hm31 = m[2]*((L[0]*s(q[0]) + L[1]*s(q[0]+q[1]) + L[2]*s(q[0]+q[1]+q[2]))*(-L[0]*s(q[0])- L[1]*s(q[0]+q[1])-l[2]*s(q[0]+q[1]+q[2]))-\
    #         (L[0]*c(q[0]) + L[1]*c(q[0]+q[1]) + L[2]*c(q[0]+q[1]+q[2]))*(L[0]*c(q[0])+L[1]*c(q[0]+q[1])+l[2]*c(q[0]+q[1]+q[2])))*dq[0]
    # Hm32 = m[2]*((L[0]*s(q[0]) + L[1]*s(q[0]+q[1]) + L[2]*s(q[0]+q[1]+q[2]))*(- L[1]*s(q[0]+q[1])-l[2]*s(q[0]+q[1]+q[2]))-\
    #         (L[0]*c(q[0]) + L[1]*c(q[0]+q[1]) + L[2]*c(q[0]+q[1]+q[2]))*(L[1]*c(q[0]+q[1])+l[2]*c(q[0]+q[1]+q[2])))*dq[1]
    # Hm33 = m[2]*((L[0]*s(q[0]) + L[1]*s(q[0]+q[1]) + L[2]*s(q[0]+q[1]+q[2]))*(-l[2]*s(q[0]+q[1]+q[2]))-\
    #         (L[0]*c(q[0]) + L[1]*c(q[0]+q[1]) + L[2]*c(q[0]+q[1]+q[2]))*(l[2]*c(q[0]+q[1]+q[2])))*dq[2]

    # r[0][0] = l[0]*s(q[0])
    # r[0][1] = l[0]*c(q[0])

    # r[1][0] = L[0]*s(q[0]) + l[1]*s(q[0]+q[1])
    # r[1][1] = L[0]*c(q[0]) + l[1]*c(q[0]+q[1])

    # r[2][0] = L[0]*s(q[0]) + L[1]*s(q[0]+q[1]) + L[2]*s(q[0]+q[1]+q[2])
    # r[2][1] = L[0]*c(q[0]) + L[1]*c(q[0]+q[1]) + L[2]*c(q[0]+q[1]+q[2])

    # v[0][0] = l[0]*c(q[0])*dq[0]
    # v[0][1] = -l[0]*s(q[0])*dq[0]

    # v[1][0] = L[0]*c(q[0])*dq[0] + l[1]*c(q[0]+q[1])*(dq[0]+dq[1])
    # v[1][1] = -L[0]*s(q[0])*dq[0] - l[1]*s(q[0]+q[1])*(dq[0]+dq[1])

    # v[2][0] = L[0]*c(q[0])*dq[0] + L[1]*c(q[0]+q[1])*(dq[0]+dq[1]) + \
    #     l[2]*c(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])
    # v[2][1] = -L[0]*s(q[0])*dq[0] - L[1]*s(q[0]+q[1])*(dq[0]+dq[1]) - \
    #     l[2]*s(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])

    # print(H, Hm, HI)
    # print(Hm1, Hm21, Hm22, Hm31, Hm32, Hm33)
    # print(Hm1+Hm21+Hm31, Hm22+Hm32, Hm33)
    pass

if __name__ == "__main__":
    main()
    # main2()
    pass
