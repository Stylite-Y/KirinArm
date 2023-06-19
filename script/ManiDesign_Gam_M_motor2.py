'''
1. 2023.05.30:
        - 羽毛球二连杆模型: 基于角动量最大化优化机械臂参数
2. 2023.06.05:
        - 羽毛球模型修正为三连杆: 基于角动量最大化优化机械臂参数
3. 2023.06.09:
        - 加入电机库参数，优化电机需求:同参数循环载入电机参数
4. 2023.06.17:
        - 加入电机库参数，优化电机需求: 不同电机参数单独设置参数
'''

import os
import yaml
import datetime
import pickle
import casadi as ca
from casadi import sin as s
from casadi import cos as c
import numpy as np
import matplotlib.pyplot as plt
import time
from ruamel.yaml import YAML
from PostProcess import DataProcess


class Bipedal_hybrid():
    def __init__(self, cfg):
        self.opti = ca.Opti()

        # time and collection defination related parameter
        self.T = cfg['Controller']['Period']
        self.Im = cfg['Robot']['Motor']['Inertia']
        self.N = cfg['Controller']['CollectionNum']
        self.dt = self.T / self.N

        self.L = cfg['Robot']['Geometry']['length']
        self.dof = len(self.L)
        self.l = cfg['Robot']['Mass']['massCenter']
        self.H_tar = cfg['Ref']['H_tar']
        self.V_tar = cfg['Ref']['V_tar']

        # motor parameter
        self.motor_cs = cfg['Robot']['Motor']['CriticalSpeed']
        self.motor_ms = cfg['Robot']['Motor']['MaxSpeed']
        self.motor_mt = cfg['Robot']['Motor']['MaxTorque']

        # evironemnt parameter
        self.mu = cfg['Environment']['Friction_Coeff']
        self.g = cfg['Environment']['Gravity']
        self.damping = cfg['Robot']['damping']
        self.m_bd = cfg['Robot']['Mass']['mass']
        self.I_bd = cfg['Robot']['Mass']['inertia']

        ## define variable
        # mass and gamma
        self.m = self.opti.variable(2)
        self.gam = self.opti.variable(3)

        # motion
        self.q = [self.opti.variable(self.dof) for _ in range(self.N)]
        self.dq = [self.opti.variable(self.dof) for _ in range(self.N)]
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.N-1)]
        self.u = [self.opti.variable(self.dof) for _ in range(self.N)]

        self.I = [self.m[0]*self.L[0]**2/12, self.m[1]*self.L[1]**2/12,
                    self.I_bd[0]]

        # boundry
        self.m_UB = cfg['Robot']['Mass']['Upper']
        self.m_LB = cfg['Robot']['Mass']['Lower']

        self.gam_UB = cfg['Robot']['Gear']['Upper']
        self.gam_LB = cfg['Robot']['Gear']['Lower']

        self.u_LB = [-self.motor_mt * self.gam[0], -self.motor_mt * self.gam[1], 
                    -self.motor_mt * self.gam[2]]
        self.u_UB = [self.motor_mt * self.gam[0], self.motor_mt * self.gam[1], 
                    self.motor_mt * self.gam[2]]

        self.q_LB = [-np.pi/6, -np.pi, -0.3*np.pi]
        self.q_UB = [np.pi/2, 0.0, 0.3*np.pi]
        self.q_m = [self.q_UB[0], self.q_LB[1], self.q_UB[2]]

        self.dq_LB = [-self.motor_ms/self.gam[0], -self.motor_ms/self.gam[1], 
                    -self.motor_ms/self.gam[2]]

        self.dq_UB = [self.motor_ms/self.gam[0], self.motor_ms/self.gam[1], 
                    self.motor_ms/self.gam[2]]

        pass

    def MassMatrix(self, q):
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
        
        C2 = L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[0] \
            - 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[2] \
            - 2*L1*lc2*m2*s(q[2]) * dq[1]*dq[2] \
            - L1*lc2*m2*s(q[2]) * dq[2]*dq[2]

        C3 = lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[0] \
            + 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[1] \
            + L1*lc2*m2*s(q[2]) * dq[1]*dq[1]

        return [C1, C2, C3]

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
            vx += L[i]*c(qj)*dqj
            vy += -L[i]*s(qj)*dqj

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
                    v[i][0] += l[j]*c(qj)*dqj
                    v[i][1] += -l[j]*s(qj)*dqj
                else:
                    v[i][0] += L[j]*c(qj)*dqj
                    v[i][1] += -L[j]*s(qj)*dqj

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
                    pos[i][0] += l[j]*s(qj)
                    pos[i][1] += l[j]*c(qj)
                else:
                    pos[i][0] += L[j]*s(qj)
                    pos[i][1] += L[j]*c(qj)

        return pos

    @staticmethod
    def get_posture(q):
        L = [0.35, 0.35]
        l_upper_x = np.zeros(2)
        l_upper_y = np.zeros(2)
        l_forearm_x = np.zeros(2)
        l_forearm_y = np.zeros(2)
        l_upper_x[0] = 0
        l_upper_x[1] = l_upper_x[0] + L[0]*np.sin(q[0])
        l_upper_y[0] = 0
        l_upper_y[1] = l_upper_y[0] + L[0]*np.cos(q[0])

        l_forearm_x[0] = 0 + L[0]*np.sin(q[0])
        l_forearm_x[1] = l_forearm_x[0] + L[1]*np.sin(q[0]+q[1])
        l_forearm_y[0] = 0 + L[0]*np.cos(q[0])
        l_forearm_y[1] = l_forearm_y[0] + L[1]*np.cos(q[0]+q[1])
        return [l_upper_x, l_upper_y, l_forearm_x, l_forearm_y]

    @staticmethod
    def get_motor_boundary(speed, MaxTorque=36, CriticalSpeed=27, MaxSpeed=53):
        upper = MaxTorque - (speed-CriticalSpeed) / \
            (MaxSpeed-CriticalSpeed)*MaxTorque
        upper = np.clip(upper, 0, MaxTorque)
        lower = -MaxTorque + (speed+CriticalSpeed) / \
            (-MaxSpeed+CriticalSpeed)*MaxTorque
        lower = np.clip(lower, -MaxTorque, 0)
        return upper, lower

    pass


class nlp():
    def __init__(self, robot, cfg, armflag = True):
        # load parameter
        self.cfg = cfg
        self.armflag = armflag
        self.T = cfg['Controller']['Period']
        self.MomtCoeff = cfg["Optimization"]["CostCoeff"]["MomtCoeff"]
        self.VelCoeff = cfg["Optimization"]["CostCoeff"]["VelCoeff"]
        self.PowerCoeff = cfg["Optimization"]["CostCoeff"]["PowerCoeff"]
        self.ForceCoeff = cfg["Optimization"]["CostCoeff"]["ForceCoeff"]
        self.SmoothCoeff = cfg["Optimization"]["CostCoeff"]["SmoothCoeff"]
        self.PosCoeff = cfg["Optimization"]["CostCoeff"]["PosCoeff"]
        self.forceRatio = cfg["Environment"]["ForceRatio"]
        max_iter = cfg["Optimization"]["MaxLoop"]

        self.cost = self.Cost(robot)
        robot.opti.minimize(self.cost)

        self.ceq = self.getConstraints(robot)
        robot.opti.subject_to(self.ceq)

        p_opts = {"expand": True, "error_on_fail": False}
        s_opts = {"max_iter": max_iter}
        robot.opti.solver("ipopt", p_opts, s_opts)
        self.initialGuess(robot)
        pass

    def initialGuess(self, robot):
        init = robot.opti.set_initial
        # region: sol1
        init(robot.gam[0], 8.0)
        init(robot.gam[1], 8.0)
        init(robot.gam[2], 3.0)
        init(robot.m[0], 2.0)
        init(robot.m[1], 2.0)
        # endregion
        theta_r, _, _ = nlp.RefTraj(robot.N, robot.dt)
        vel_r = []
        for i in range(robot.dof):
            vel_r.append([theta_r[i][j+1] - theta_r[i][j] for j in range(robot.N-1)])
        for i in range(robot.N):
            init(robot.q[i][0], theta_r[0][i])
            init(robot.q[i][1], theta_r[1][i])
            init(robot.q[i][2], theta_r[2][i])
            for j in range(robot.dof):
                if i <robot.N-1:
                    init(robot.dq[i][j], vel_r[j][i])
                else:
                    init(robot.dq[i][j], vel_r[j][i-1])
            pass

    def Cost(self, robot):
        # region aim function of optimal control
        power = 0
        force = 0
        VelTar = 0
        PosTar = 0
        smooth = 0
        MomtTar=0

        H_m = robot.H_tar
        Ve_m = robot.V_tar
        P_tar = [0.05*np.pi, -0.01*np.pi, 0.01*np.pi]
        q_m = robot.q_m
        # q_m = [np.pi,np.pi, np.pi]
        # endregion
        
        for i in range(robot.N):
            # font vel cal
            vxi, vyi = robot.get_endvel(robot.q[i], robot.dq[i])
            vei = vxi**2 + vyi**2
            ve = ca.sqrt(vei)

            # angular momt cal
            ri = robot.get_compos(robot.q[i])
            v = robot.get_comvel(robot.q[i], robot.dq[i])

            # for j in range(robot.dof):
            #     Hi += robot.m[j]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]) + robot.I[i]*robot.dq[i][0]
            Hi = robot.m[0]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]) + robot.I[0]*robot.dq[i][0] + \
                 robot.m[1]*(ri[1][0]*v[1][1]-ri[1][1]*v[1][0]) + robot.I[1]*(robot.dq[i][0]+robot.dq[i][1]) +\
                 robot.m_bd[0]*(ri[2][0]*v[2][1]-ri[2][1]*v[2][0]) + robot.I[2]*(robot.dq[i][0]+robot.dq[i][1]+robot.dq[i][2])
            
            # state process costfun
            MomtTar += -(Hi/H_m)**2 * robot.dt
            # VelTar += (ve/Ve_m - 1)**2 * robot.dt
            # VelTar += -((vei)/(Ve_m)**2) * robot.dt
            VelTar += ((vei)/(Ve_m)**2 - 1)**2 * robot.dt

            for k in range(robot.dof):
                power += ((robot.dq[i][k]*robot.u[i][k]) / (robot.motor_ms*robot.motor_mt))**2 * robot.dt
                force += (robot.u[i][k] / robot.u_UB[k])**2 * robot.dt
                PosTar += ((robot.q[i][k] - P_tar[k])/q_m[k])**2*robot.dt

        # Final state costfun
        # VelTar += (ve/Ve_m - 1)**2
        # VelTar += -((vei)/(Ve_m)**2)
        VelTar += ((vei)/(Ve_m)**2 - 1)**2
        MomtTar += -(Hi/H_m)**2
        for i in range(robot.dof):
            # PosTar += ((robot.q[-1][i] - P_tar[i])/q_m[k])**2 * 1.2
            PosTar += ((robot.q[-1][i] - P_tar[i])/q_m[k])**2 * 1.2

        for i in range(robot.N-1):
            for k in range(robot.dof):
                smooth += ((robot.u[i+1][k]-robot.u[i][k])/10)**2
                pass
            pass

        res = 0
        res = (res + power*self.PowerCoeff) if (self.PowerCoeff > 1e-6) else res
        res = (res + VelTar*self.VelCoeff) if (self.VelCoeff > 1e-6) else res
        res = (res + MomtTar*self.MomtCoeff) if (self.MomtCoeff > 1e-6) else res
        res = (res + force*self.ForceCoeff) if (self.ForceCoeff > 1e-6) else res
        res = (res + smooth*self.SmoothCoeff) if (self.SmoothCoeff > 1e-6) else res
        res = (res + PosTar*self.PosCoeff) if (self.PosCoeff > 1e-6) else res

        return res

    def getConstraints(self, robot):
        ceq = []
        # region dynamics constraints
        # continuous dynamics
        for j in range(robot.N):
            if j < (robot.N-1):
                ceq.extend([robot.q[j+1][k]-robot.q[j][k]-robot.dt/2 *
                            (robot.dq[j+1][k]+robot.dq[j][k]) == 0 for k in range(robot.dof)])
                inertia = robot.inertia_force(
                    robot.q[j], robot.ddq[j])
                coriolis = robot.coriolis(
                    robot.q[j], robot.dq[j])
                gravity = robot.gravity(robot.q[j])
                ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                            robot.u[j][k] == 0 for k in range(robot.dof)])

        # endregion

        # ceq.extend([robot.mm2[0]-0.8*robot.mm2[1] >= 0])

        # region boundary constraint
        for temp_q in robot.q:
            ceq.extend([robot.opti.bounded(robot.q_LB[j],
                        temp_q[j], robot.q_UB[j]) for j in range(robot.dof)])
            pass
        for temp_dq in robot.dq:
            ceq.extend([robot.opti.bounded(robot.dq_LB[j],
                        temp_dq[j], robot.dq_UB[j]) for j in range(robot.dof)])
            pass
        for temp_u in robot.u:
            ceq.extend([robot.opti.bounded(robot.u_LB[j],
                        temp_u[j], robot.u_UB[j]) for j in range(robot.dof)])

        ceq.extend([robot.opti.bounded(robot.m_LB[j],
                    robot.m[j], robot.m_UB[j]) for j in range(2)])

        ceq.extend([robot.opti.bounded(robot.gam_LB[j],
                    robot.gam[j], robot.gam_UB[j]) for j in range(robot.dof)])

        # ceq.extend([robot.m[0]-0.8*robot.m[1] >= 0])
        # endregion

        # region motor external characteristic curve
        cs = robot.motor_cs
        ms = robot.motor_ms
        mt = robot.motor_mt
        for i in range(len(robot.u)):
            for j in range(robot.dof):
                motor_u = robot.u[i][j] / robot.gam[j]
                motor_dq = robot.dq[i][j] * robot.gam[j]
                ceq.extend([motor_u-(-mt)/(ms-cs)*(motor_dq-ms)<=0])
                ceq.extend([motor_u-(-mt)/(ms-cs)*(motor_dq+ms)>=0])
            # ceq.extend([robot.u[j][k]-ca.fmax(mt[k] - (robot.dq[j][k] -
            #                                             cs[k])/(ms[k]-cs[k])*mt[k], 0) <= 0 for k in range(robot.dof)])
            # ceq.extend([robot.u[j][k]-ca.fmin(-mt[k] + (robot.dq[j][k] +
            #                                                 cs[k])/(-ms[k]+cs[k])*mt[k], 0) >= 0 for k in range(robot.dof)])
            pass

        # endregion

        ceq.extend([robot.q[0][0] == -np.pi* 0.1])
        ceq.extend([robot.q[0][1] == -np.pi* 0.7])
        ceq.extend([robot.q[0][2] == -np.pi* 0.2])

        ceq.extend([robot.dq[0][0]==0])
        ceq.extend([robot.dq[0][1]==0])
        ceq.extend([robot.dq[0][2]==0])

        # region smooth constraint
        for j in range(len(robot.u)-1):
            ceq.extend([(robot.u[j][k]-robot.u
                        [j+1][k])**2 <= 50 for k in range(robot.dof)])
            pass
        # endregion

        return ceq

    def solve_and_output(self, robot, flag_save=True, StorePath="./", **params):
        # solve the nlp and stroge the solution
        q = []
        dq = []
        ddq = []
        u = []
        t = []
        try:
            sol1 = robot.opti.solve()
            gamma=sol1.value(robot.gam)
            m=sol1.value(robot.m)
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([sol1.value(robot.q[j][k]) for k in range(robot.dof)])
                dq.append([sol1.value(robot.dq[j][k])
                            for k in range(robot.dof)])
                if j < (robot.N-1):
                    ddq.append([sol1.value(robot.ddq[j][k])
                                for k in range(robot.dof)])
                    u.append([sol1.value(robot.u[j][k])
                                for k in range(robot.dof)])
                else:
                    ddq.append([sol1.value(robot.ddq[j-1][k])
                                for k in range(robot.dof)])
                    u.append([sol1.value(robot.u[j-1][k])
                                for k in range(robot.dof)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            gamma=value(robot.gam)
            m=value(robot.m)
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k])
                            for k in range(robot.dof)])
                dq.append([value(robot.dq[j][k])
                            for k in range(robot.dof)])
                if j < (robot.N-1):
                    ddq.append([value(robot.ddq[j][k])
                                for k in range(robot.dof)])
                    u.append([value(robot.u[j][k])
                                for k in range(robot.dof)])
                else:
                    ddq.append([value(robot.ddq[j-1][k])
                                for k in range(robot.dof)])
                    u.append([value(robot.u[j-1][k])
                                for k in range(robot.dof)])
                pass
            pass
        finally:
            q = np.asarray(q)
            dq = np.asarray(dq)
            ddq = np.asarray(ddq)
            u = np.asarray(u)
            m = np.asarray(m)
            gam = np.asarray(gamma)
            t = np.asarray(t).reshape([-1, 1])

            if flag_save:
                # datename initial
                date = params['date']
                dirname = "-Traj-Mtf_"+str(self.MomtCoeff)+"-Pf_"+str(self.PosCoeff)+"-Vf_"+str(self.VelCoeff)+\
                        "-Wf_"+str(self.PowerCoeff)+"-Ff_"+str(self.ForceCoeff)+"-Sf_"+str(self.SmoothCoeff)+\
                        "-T_"+str(self.T)
                save_dir = StorePath + date + dirname+ "/"

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                with open(save_dir+"config.yaml", mode='w') as file:
                    YAML().dump(self.cfg, file)
                Data = {'u': u, "q": q, "dq": dq, "ddq": ddq, "t": t, "m": m, "gam": gam}
                with open(os.path.join(save_dir, "sol.pkl"), 'wb') as f:
                    pickle.dump(Data, f)

            return q, dq, ddq, u, t, gamma, m

    @staticmethod
    def RefTraj(N, dt):
        # Cs = [5.851e5, -1.262e5, 8662, -218.2, 12.75, -0.3481]
        # Cs = [-285.1, 6.626, 11.66, -0.3542]
        Cs = [-1.886e10, 6.99e9, -1.034e9, 7.82e7, -3.215e6, 7.0e4, -697.7, 12.92, -0.3441]
        Ce = [-1.521e6, 3.045e5, -2.012e4, 703.1, 0.2709, -2.809]
        # Cs = [-7.367e7, 5.46e7, -1.616e7, 2.444e6, -2.01e5, 8751, -174.4, 6.462, -0.3441]
        # Ce = [-4.753e5, 1.903e4, -2515, 175.8, 0.1355, -2.809]
        Cw = [np.pi, -0.2*np.pi]

        theta_s = []
        theta_e = []
        theta_w = []
        dtheta_s = []
        dtheta_e = []
        dtheta_w = []
        ddtheta_s = []
        ddtheta_e = []
        ddtheta_w = []
        for i in range(N):
            tt = i * dt
            theta_sm = Cs[0]*tt**8 + Cs[1]*tt**7 + Cs[2]*tt**6 + Cs[3]*tt**5 + \
                    Cs[4]*tt**4 + Cs[5]*tt**3 + Cs[6]*tt**2 + Cs[7]*tt**1 + Cs[8]
            # theta_sm = Cs[0]*tt**3 + Cs[1]*tt**2 + Cs[2]*tt**1 + Cs[3]
            theta_em = Ce[0]*tt**5 + Ce[1]*tt**4 + Ce[2]*tt**3 + Ce[3]*tt**2 + \
                    Ce[4]*tt**1 + Ce[5]
            theta_wm = Cw[0]*tt + Cw[1]

            dtheta_sm = 8*Cs[0]*tt**7 + 7*Cs[1]*tt**6 + 6*Cs[2]*tt**5 +5*Cs[3]*tt**4 + \
                    4*Cs[4]*tt**3 + 3*Cs[5]*tt**2 + 2*Cs[6]*tt**1 + Cs[7]
            dtheta_em = 5*Ce[0]*tt**4 + 4*Ce[1]*tt**3 + 3*Ce[2]*tt**2 + 2*Ce[3]*tt**1 + \
                    Ce[4]
            dtheta_wm = Cw[0]

            ddtheta_sm = 8*7*Cs[0]*tt**6 + 7*6*Cs[1]*tt**5 + 6*5*Cs[2]*tt**4 + 5*4*Cs[3]*tt**3 + \
                    4*3*Cs[4]*tt**2 + 3*2*Cs[5]*tt**1 + 2*1*Cs[6]
            ddtheta_em = 5*4*Ce[0]*tt**3 + 4*3*Ce[1]*tt**2 + 3*2*Ce[2]*tt**1 + 2*1*Ce[3]
            ddtheta_wm = Cw[0]
            
            theta_s.append(theta_sm)
            theta_e.append(theta_em)
            theta_w.append(theta_wm)

            dtheta_s.append(dtheta_sm)
            dtheta_e.append(dtheta_em)
            dtheta_w.append(dtheta_wm)
            ddtheta_s.append(ddtheta_sm)
            ddtheta_e.append(ddtheta_em)
            ddtheta_w.append(ddtheta_wm)
            pass

        theta_ref = [theta_s, theta_e, theta_w]
        dtheta_ref = [dtheta_s, dtheta_e, dtheta_w]
        ddtheta_ref = [ddtheta_s, ddtheta_e, ddtheta_w]
        return theta_ref, dtheta_ref, ddtheta_ref
        pass

class ParamsCal():
    def __init__(self, dof, L=[0.35, 0.35], l=[0.175, 0.175]):
        self.dof = dof
        self.L = L
        self.l = l
        pass

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
    
    def get_AngMomt(self):
        pass

    def get_costfun(self, cfg, q, dq, u, m, gam, I):
        T = cfg['Controller']['Period']
        N = cfg['Controller']['CollectionNum']
        dt = T / N
        L = self.L

        # motor parameter
        motor_ms = cfg['Robot']['Motor']['MaxSpeed']
        motor_mt = cfg['Robot']['Motor']['MaxTorque']
        H_tar = cfg['Ref']['H_tar']
        V_tar = cfg['Ref']['V_tar']


        u_UB = [motor_mt * gam[0], motor_mt * gam[1], motor_mt * gam[2]]
        q_UB = [np.pi/2, np.pi, 0.3*np.pi]

        MomtCoeff = cfg["Optimization"]["CostCoeff"]["MomtCoeff"]
        VelCoeff = cfg["Optimization"]["CostCoeff"]["VelCoeff"]
        PowerCoeff = cfg["Optimization"]["CostCoeff"]["PowerCoeff"]
        ForceCoeff = cfg["Optimization"]["CostCoeff"]["ForceCoeff"]
        SmoothCoeff = cfg["Optimization"]["CostCoeff"]["SmoothCoeff"]
        PosCoeff = cfg["Optimization"]["CostCoeff"]["PosCoeff"]
        # region aim function of optimal control
        power = 0
        force = 0
        VelTar = 0
        VelTar2 = 0
        PosTar = 0
        PosTar2 = 0
        smooth = 0
        MomtTar=0
        MomtTar2=0

        H_m = H_tar
        Ve_m = V_tar
        P_tar = [0.05*np.pi, -0.01*np.pi, 0.01*np.pi]
        # endregion

        Hm = []
        HI = []
        Vxx = []
        Vyy = []
        
        Pw = []
        F = []
        Vel = []
        Pos = []
        Momt = []
        
        for i in range(N):
            # font vel cal
            vxi, vyi = self.get_endvel(q[i], dq[i])
            vei = vxi**2 + vyi**2

            Vxx.append([L[0]*np.cos(q[i][0])*dq[i][0], 
                        L[1]*np.cos(q[i][0]+q[i][1])*(dq[i][0]+dq[i][1]),
                        L[2]*np.cos(q[i][0]+q[i][1]+q[i][2])*(dq[i][0]+dq[i][1]+dq[i][2])])
            Vyy.append([-L[0]*np.sin(q[i][0])*dq[i][0], 
                        -L[1]*np.sin(q[i][0]+q[i][1])*(dq[i][0]+dq[i][1]),
                        -L[2]*np.sin(q[i][0]+q[i][1]+q[i][2])*(dq[i][0]+dq[i][1]+dq[i][2])])

            # angular momt cal
            ri = self.get_compos(q[i])
            v = self.get_comvel(q[i], dq[i])

            # for j in range(robot.dof):
            #     Hi += robot.m[j]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]) + robot.I[i]*robot.dq[i][0]
            Hi = m[0]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]) + I[0]*dq[i][0] + \
                 m[1]*(ri[1][0]*v[1][1]-ri[1][1]*v[1][0]) + I[1]*(dq[i][0]+dq[i][1]) +\
                 m[2]*(ri[2][0]*v[2][1]-ri[2][1]*v[2][0]) + I[2]*(dq[i][0]+dq[i][1]+dq[i][2])
            
            Hm.append([m[0]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]), 
                        m[1]*(ri[1][0]*v[1][1]-ri[1][1]*v[1][0]),
                        m[2]*(ri[2][0]*v[2][1]-ri[2][1]*v[2][0])])

            HI.append([I[0]*dq[i][0], I[1]*(dq[i][0]+dq[i][1]), I[2]*(dq[i][0]+dq[i][1]+dq[i][2])])
            
            # state process costfun
            MomtTar += -(Hi/H_m)**2 * dt
            # VelTar += -((vei)/(Ve_m)**2) * dt
            VelTar += ((vei)/(Ve_m)**2 - 1)**2 * dt

            for k in range(self.dof):
                power += ((dq[i][k]*u[i][k]) / (motor_ms*motor_mt))**2 * dt
                force += (u[i][k] / u_UB[k])**2 * dt
                PosTar += ((q[i][k] - P_tar[k])/q_UB[k])**2*dt

            Pw.append(power)
            F.append(force)
            Pos.append(PosTar)
            Vel.append(VelTar)
            Momt.append(MomtTar)

        # Final state costfun
        # VelTar += -((vei)/(Ve_m)**2)
        VelTar2 += ((vei)/(Ve_m)**2 - 1)**2
        VelTar += ((vei)/(Ve_m)**2 - 1)**2
        MomtTar2 += -(Hi/H_m)**2
        for i in range(self.dof):
            # PosTar2 += ((q[-1][i] - P_tar[i])/q_UB[k])**2 * 1.2
            PosTar2 += ((q[-1][i] - P_tar[i])/q_UB[k])**2 * 1.2

        Pos[-1] = PosTar2
        Vel[-1] = VelTar
        Momt[-1] = MomtTar2

        PosTar = PosTar * PosCoeff
        PosTar2 = PosTar2 * PosCoeff
        MomtTar = MomtTar * MomtCoeff
        MomtTar2 = MomtTar2 * MomtCoeff 
        VelTar = VelTar * VelCoeff
        VelTar2 = VelTar2 * VelCoeff
        power = power * PowerCoeff
        force = force * ForceCoeff

        Pw = np.asarray(Pw)
        F = np.asarray(F)
        Pos = np.asarray(Pos)
        Vel = np.asarray(Vel)
        Momt = np.asarray(Momt)

        # AllCost = Pw * PowerCoeff
        AllCost = Pw * PowerCoeff+ F * ForceCoeff+ Pos * PosCoeff+ Vel * VelCoeff+Momt * MomtCoeff

        CostF = [AllCost, Pw * PowerCoeff, F * ForceCoeff, Pos * PosCoeff, Vel * VelCoeff, Momt * MomtCoeff]

        CostFun = [PosTar, PosTar2, MomtTar, MomtTar2, VelTar, VelTar2, power, force]

        return CostFun, [Vxx, Vyy], [Hm, HI], CostF


def main():
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = True
    save_flag = True
    # endregion

    # region: filepath
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    ani_path = StorePath + "/data/animation/"
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # endregion

    # region load config file
    ParamFilePath = StorePath + "/config/ManiDesign.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)
    # endregion

    # region create robot and NLP problem
    robot = Bipedal_hybrid(cfg)
    nonlinearOptimization = nlp(robot, cfg)
    q, dq, ddq, u, t, gamma, m = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=save_dir, date = date)
    
    print("="*50)
    print("gamma:", gamma)
    print("="*50)
    print("m:", m)
    print("="*50)
    # endregion

    # region: ParamsCal
    mm = []
    I = []
    m_bd = cfg['Robot']['Mass']['mass']
    Length = cfg['Robot']['Geometry']['length']
    lc = cfg['Robot']['Mass']['massCenter']
    dof = len(Length)
    for i in range(dof):
        if i < 2:
            mm.append(m[i])
            I.append(m[i]*Length[i]**2/12)
        else:
            mm.append(m_bd[i-2])
            I.append(m_bd[i-2]*Length[i]**2/12)

    Lam = []
    EndVel = []
    Params = ParamsCal(dof, L=Length, l=lc)
    for i in range(len(t)):
        # font vel cal
        vxi, vyi = Params.get_endvel(q[i], dq[i])
        vi = np.sqrt(vxi**2 + vyi**2)

        # angular momt cal
        ri = Params.get_compos(q[i])
        v = Params.get_comvel(q[i], dq[i])

        Hi = mm[0]*(ri[0][0]*v[0][1]-ri[0][1]*v[0][0]) + I[0]*dq[i][0] + \
            mm[1]*(ri[1][0]*v[1][1]-ri[1][1]*v[1][0]) + I[1]*(dq[i][0]+dq[i][1])+\
            mm[2]*(ri[2][0]*v[2][1]-ri[2][1]*v[2][0]) + I[2]*(dq[i][0]+dq[i][1]+dq[i][2])
        
        EndVel.append(vi)
        Lam.append(Hi)
        pass
    # endregion

    # region: costfun cal
    CostFun, Vend, Hmt, CostF = Params.get_costfun(cfg, q, dq, u, mm, gamma, I)
    print("CostFun Component")
    print("-"*50)
    print("Position process cost: ", CostFun[0])
    print("Position final-state cost: ", CostFun[1])
    print("Momtentum process cost: ", CostFun[2])
    print("Momtentum final-state cost: ", CostFun[3])
    print("Velocity process cost: ", CostFun[4])
    print("Velocity final-state cost: ", CostFun[5])
    print("Power process cost: ", CostFun[6])
    print("Force process cost: ", CostFun[7])
    print("="*50)
    # print(Vend[0][300][2], Hmt[0][200][1])
    # endregion

    # region: acc cal
    theta_r, dtheta_r, ddtheta_r = nlp.RefTraj(robot.N, robot.dt)
    # endregion
    

    # region: visulization
    if vis_flag:
        params = {
            'text.usetex': True,
            'font.size': 15,
            'axes.titlesize': 15,
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
        }

        plt.rcParams.update(params)
        vis = DataProcess(cfg, q, dq, ddq, u, t, gamma, m, save_dir, save_flag, date = date)
        fig1 = vis.TrajPlot()
        fig2 = vis.ParamsCalAndPlot(Lam=Lam, EndVel=EndVel)
        fig3 = vis.VelAndMomtAnlysis(Vend=Vend, H=Hmt)
        # fig4 = vis.AccRefCmp(dq_r=dtheta_r, ddq_r=ddtheta_r)
        fig5 = vis.ForceComponentAnalysis(m=mm, gam=gamma)
        fig6 = vis.CostFun(CostFun = CostF)
        ani = vis.animation()

        # plt.show()
        pass
    # endregion

if __name__ == "__main__":
    main()
    pass
