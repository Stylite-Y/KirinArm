import sympy
from sympy import sin
from sympy import cos
from sympy import symbols
from sympy import Symbol as sym
from sympy import Function as Fun
from sympy import init_printing, pprint
import os

def ThreeLink():
    
    t = sym('t')

    # ================= variable and parameter defination ==================
    # define state variable
    theta0 = Fun('theta0', real=True)(t)
    theta1 = Fun('theta1', real=True)(t)
    theta2 = Fun('theta2', real=True)(t)

    dtheta0 = theta0.diff(t)
    dtheta1 = theta1.diff(t)
    dtheta2 = theta2.diff(t)
    
    # define geometry and mass parameter
    # body, thigh, shank, uparm, forearm
    m = [sym('m'+str(i)) for i in range(3)]
    I = [sym('I'+str(i)) for i in range(3)]
    L = [sym('L'+str(i)) for i in range(3)]
    l = [sym('l'+str(i)) for i in range(3)]
    g = sym('g')

    # ==================== geometry constraint ===================
    # position relationship
    htheta0 = theta0
    htheta1 = theta0 + theta1
    htheta2 = theta0 + theta1 + theta2

    x0 = l[0]* sin(theta0)       # link 2 x mcp
    z0 = l[0]* cos(theta0)       # link 2 y mcp
    x1 = L[0]* sin(theta0) + l[1]* sin(htheta1)       # link 2 x mcp
    z1 = L[0]* cos(theta0) + l[1]* cos(htheta1)       # link 2 y mcp
    x2 = L[0]* sin(theta0) + L[1]* sin(htheta1) + l[2]* sin(htheta2)       # link 2 x mcp
    z2 = L[0]* cos(theta0) + L[1]* cos(htheta1) + l[2]* cos(htheta2)       # link 2 y mcp

    # velocity relationship
    dhtheta0 = dtheta0
    dhtheta1 = dtheta0 + dtheta1
    dhtheta2 = dtheta0 + dtheta1 + dtheta2

    dx0 = l[0]* cos(theta0)*dtheta0       # link 2 x mcp
    dz0 = - l[0]* sin(theta0)*dtheta0       # link 2 y mcp
    dx1 = L[0]* cos(theta0)*dtheta0 + l[1]* cos(htheta1)*(dhtheta1)       # link 2 x mcp
    dz1 = - L[0]* sin(theta0)*(dtheta0) - l[1]* sin(htheta1)*(dhtheta1)      # link 2 y mcp
    dx2 = L[0]* cos(theta0)*dtheta0 + L[1]* cos(htheta1)*(dhtheta1) + \
          l[2]* cos(htheta2)*(dhtheta2)       # link 2 x mcp
    dz2 = - L[0]* sin(theta0)*dtheta0 - L[1]* sin(htheta1)*(dhtheta1) - \
          l[2]* sin(htheta2)*(dhtheta2)       # link 2 y mcp
 
    # ==================== kinematic and potential energy ===================
    # 动能计算： 刚体动能 = 刚体质心平移动能 + 绕质心转动动能 = 1 / 2 * m * vc ** 2 + 1 / 2 * Ic * dtheta ** 2
    T0 = 0.5 * m[0] * (dx0**2 + dz0**2) + 0.5 * I[0] * (dhtheta0**2)
    T1 = 0.5 * m[1] * (dx1**2 + dz1**2) + 0.5 * I[1] * (dhtheta1**2)
    T2 = 0.5 * m[2] * (dx2**2 + dz2**2) + 0.5 * I[2] * (dhtheta2**2)

    T = T0 + T1 + T2

    # potential energy
    V = m[0] * g * z0 + \
        m[1] * g * z1 + m[2] * g * z2 

    # Lagrange function
    L = T - V

    os.system('cls' if os.name == 'nt' else 'clear')
    init_printing()

    print("HumanModelHalf3")

    eq3 = L.diff(dtheta0).diff(t) - L.diff(theta0)
    eq4 = L.diff(dtheta1).diff(t) - L.diff(theta1)
    eq5 = L.diff(dtheta2).diff(t) - L.diff(theta2)
    # print(eq7)

    eq = [eq3, eq4, eq5]
    

    def get_inertia_term(f):
        print("Inertia term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)

            try:
                pprint(sympy.trigsimp(temp[s]), use_unicode=True)
                # pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(len(f)):
            print("="*50)
            print("Inertia term wrt. joint ", i)
            try_print(f[i], theta0.diff(t).diff(t))
            try_print(f[i], theta1.diff(t).diff(t))
            try_print(f[i], theta2.diff(t).diff(t))
        pass

    def get_gravity_term(f):
        print("Gravity term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)
            try:
                pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(len(f)):
            print("="*50)
            print("gravity term wrt. joint ", i)
            try_print(f[i], g)
        pass

    def get_coriolis_term(f):
        print("Coriolis term")
        s = [dtheta0, dtheta1, dtheta2]
        ss = [sym('O0\''), sym('O1\''), sym('O2\'')]

        for i in range(len(s)):
            f = f.replace(s[i], ss[i])
            pass
        # print(f)
        sss = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sss.append(ss[i]*ss[j])
                pass
            pass
        # print(sss)
        # s = [Xb.diff(t), Yb.diff(t), Ob.diff(t), O1many  negative sign[0].diff(t),
        #      O2[0].diff(t), O3[0].diff(t)]
        # temp = sympy.collect(
        #     f.expand(), sss, evaluate=False)
        # pprint(temp)
        cor = None
        for i in range(len(s)):
            for j in range(i, len(s)):
                print("-"*50)
                temp= sympy.collect(f.expand(), ss[i]*ss[j],  evaluate=False)
                print(i, j)
                
                try:
                    tttt = temp[ss[i]*ss[j]]*s[i]*s[j]
                    # cor = cor + tttt if cor else tttt
                    # print(cor)
                    # print(tttt)
                    cor = sympy.simplify(tttt)
                    pprint(cor, use_unicode=True)

                except:
                    pass
                pass
            # print("-"*50)
            pass
        print("-"*50)

        # print(cor)
        # cor = sympy.simplify(cor)
        # # cor = sympy.factor(cor)
        # pprint(cor, use_unicode=True)
        pass

    get_inertia_term(eq)
    print("\n"*5)
    get_gravity_term(eq)
    print("\n"*5)
    for i in range(3):
        print("="*50)
        print("Inertia term wrt. joint ", i)
        get_coriolis_term(eq[i])
    pass


if __name__=="__main__":
    ThreeLink()