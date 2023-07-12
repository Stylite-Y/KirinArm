"""
Author: hyyuan
Description: This code is used to calculate the parameters of the Cycloid Drive according to the reduce ratio and Diameter
"""

import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import ttk

# 摆线减速器参数计算: gam 减速比;, Dp 针轮中心园直径, dsw柱销直径
def ParamsCal(gam, Dp, dsw):
    try:
        if gam % 2 == 0:
            raise ValueError(('invalid value: gam should be odd number!'))
        if Dp <= 0:
            raise ValueError(('invalid value: Dp should be a positive number!'))
        if dsw<=0:
            raise ValueError(('invalid value: dsw should be a positive number!'))
    except ValueError as e:
        print("Error: ",repr(e))
    else:
        # Zp: 针轮齿数, Zc: 摆线轮齿数
        Zp = gam + 1
        Zc = gam

        # region: K1 变幅系数; K2 针径系数
        if Zc <= 11:
            K1 = 0.42 + (0.55-0.42)*Zc/11
        elif Zc<=23 and Zc>13:
            K1 = 0.48 + (0.74-0.48)*(Zc-13)/(23-13)
        elif Zc<=59 and Zc>23:
            K1 = 0.65 + (0.9-0.65)*(Zc-25)/(59-25)
        elif Zc <=87 and Zc>61:
            K1 = 0.7 + (0.9-0.7)*(Zc-61)/(87-61)

        if Zp < 12:
            K2 = 3.85 - (3.85-2.85)*Zp/12
        elif Zp <24 and Zp>=12:
            K2 = 2.8 - (2.8-2.0)*(Zp-12)/(24-12)
        elif Zp <36 and Zp>=24:
            K2 = 2.0 - (2.0-1.25)*(Zp-24)/(36-24)
        elif Zp <60 and Zp>=36:
            K2 = 1.6 - (1.6-1.0)*(Zp-36)/(60-36)
        elif Zp <88 and Zp>=60:
            K2 = 1.5 - (1.5-0.99)*(Zp-60)/(88-60)
        # endregion

        # region: 柱销数量
        if Dp<100:
            Zw = 6
        elif Dp<200 and Dp>=100:
            Zw = 8
        elif Dp<300 and Dp>=200:
            Zw = 10
        else:
            Zw = 12
        # endregion

        D1 = 0.4*Dp         # 摆线轮中心孔直径
        B = 0.15*Dp/2       # 摆线轮厚度
        a = K1*Dp/(2*Zp)    # 偏心距
        D_rp = Dp/K2*np.sin(180/Zp)         # 针径销钉套直径
        D_fc = Dp - 2*a - D_rp              # 摆线轮齿根圆直径
        Dw = (D_fc + D1)/2                  # 柱销中心园直径
        drw = 1.4*dsw                       # 柱销套直径
        dw = drw + 2*a                      # 摆线轮销孔直径 

        # region: params print
        print("="*50)
        print("过程参数:")
        print("*"*20)
        print("变幅系数 K1:            ", K1)
        print("针齿系数 K2:            ", K2)
        print("*"*50)
        print("设计参数:")
        print("*"*20)
        print("减速比/摆线轮齿数:       ", Zc)
        print("针轮齿数:               ", Zp)
        print("偏心距:                 ", a)
        print("-"*20)
        print("针齿中心圆直径:          ", Dp)
        print("针齿销钉直径:            ", D_rp)
        print("针齿销钉套直径:          ", D_rp+4)
        print("-"*20)
        print("摆线轮中心孔直径:        ", D1)
        print("柱销中心圆直径:          ", Dw)
        print("摆线轮厚度:             ", B)
        print("-"*20)
        print("摆线轮销孔直径:          ", dw)
        print("柱销数量:               ", Zw)
        print("柱销直径:               ", dsw)
        print("柱销套直径:             ", drw)
        # endregion
    pass

def ParamsCalGUI():
    # 第1步，实例化object，建立窗口window
    window = tk.Tk()
    window.config(bg='white')
    
    # 第2步，给窗口的可视化起名字
    window.title('Cycliod Params Calculate')
    
    # 第3步，设定窗口的大小(长 * 宽)
    window.geometry('1400x950')  # 这里的乘是小x
    w1 = tk.PanedWindow(window, relief='raised', width=500)
    w1.grid(row=0, column=0)
    w1.config(bg='white')
    w2 = tk.PanedWindow(window, relief='raised', width=500)
    w2.grid(row=0, column=1)
    w2.config(bg='white')
    canvas = tk.Canvas(w1,bg='white',bd=0,height=340,width=600)
    canvas.config(highlightthickness=0)
    canvas.grid(row=0, column=0, columnspan=2)
    imagepath = '/home/hyyuan/Documents/Master/Manipulator/KirinArm/script/Hardware/fig/cv2.png'
    img = Image.open(imagepath)
    resized_img = img.resize((550, 330))
    tkimg = ImageTk.PhotoImage(resized_img)
    # image_file = tk.PhotoImage(file='/home/hyyuan/Documents/Master/Manipulator/KirinArm/script/Hardware/fig/cv2.png')  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）
    image = canvas.create_image(0, 0, anchor='nw',image=tkimg) 
    
    # region: 第4步，在图形界面上设定标签
    linfo1 = tk.Label(w1, text='输入参数(必须)', bg='slategrey',fg='black',font=('Arial', 15), width=60, height=2)
    linfo1.grid(row=1, column=0, rowspan=1, columnspan=2)
    # 减速比E
    l1 = tk.Label(w1, text='减速比(gam/Zc)', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l1.grid(row=2, column=0)
    e1 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e1.grid(row=2, column=1)
    # 针齿中心圆直径
    l2 = tk.Label(w1, text='针齿中心圆直径(Dp): ', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l2.grid(row=3, column=0)
    e2 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e2.grid(row=3, column=1)  
    # 柱销直径
    l3 = tk.Label(w1, text='径向间距(dr，齿阔线修形): ', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l3.grid(row=4, column=0)
    e4 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e4.grid(row=4, column=1)  

    ## 可选参数
    linfo11 = tk.Label(w1, text='可变参数', bg='slategrey',fg='black',font=('Arial', 15), width=30, height=2)
    linfo11.grid(row=5, column=0, rowspan=1, columnspan=1)
    # 柱销直径
    l3 = tk.Label(w1, text='柱销直径(dsw): ', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l3.grid(row=6, column=0)
    e3 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e3.grid(row=6, column=1)  
    # 变幅系数K1
    l11 = tk.Label(w1, text='变幅系数(K1)', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l11.grid(row=7, column=0)
    e11 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e11.grid(row=7, column=1)
    # K1表格
    cols = ("Zc", "<=11", "13~23", "25~59", "61~87")
    tree = ttk.Treeview(w1, height=1, show="headings", columns=cols)
    # tree["columns"] = ("zc", "<=11", "13~23", "25~59", "61~87")
    # tree.column("Zc", width=130)
    # tree.column("<=11", width=130)
    # tree.column("13~23", width=130)
    # tree.column("25~59", width=130)
    # tree.column("61~87", width=130)
    tree.column("Zc", width=80)
    tree.column("<=11", width=80)
    tree.column("13~23", width=80)
    tree.column("25~59", width=80)
    tree.column("61~87", width=80)
    tree.heading("Zc", text=cols[0], anchor=tk.W)
    tree.heading("<=11", text=cols[1], anchor=tk.W)
    tree.heading("13~23", text=cols[2], anchor=tk.W)
    tree.heading("25~59", text=cols[3], anchor=tk.W)
    tree.heading("61~87", text=cols[4], anchor=tk.W)
    tree.insert("", 0, values=("K1", "0.42~0.55", "0.48~0.74", "0.65~0.9", "0.75~0.9"))
    tree.grid(row=8, column=0, columnspan=2)
    # 针轮系数K2
    l21 = tk.Label(w1, text='针轮系数(K2)', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l21.grid(row=9, column=0)
    e21 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e21.grid(row=9, column=1)  
    cols = ("Zp", "<=12", "12~24", "24~36", "36~60", "60~88")
    tree2 = ttk.Treeview(w1, height=1, show="headings", columns=cols)
    # tree["columns"] = ("zc", "<=11", "13~23", "25~59", "61~87")
    # tree2.column("Zp", width=100)
    # tree2.column("<=12", width=110)
    # tree2.column("12~24", width=110)
    # tree2.column("24~36", width=110)
    # tree2.column("36~60", width=110)
    # tree2.column("60~88", width=110)
    tree2.column("Zp", width=50)
    tree2.column("<=12", width=80)
    tree2.column("12~24", width=70)
    tree2.column("24~36", width=70)
    tree2.column("36~60", width=70)
    tree2.column("60~88", width=70)
    tree2.heading("Zp", text=cols[0], anchor=tk.W)
    tree2.heading("<=12", text=cols[1], anchor=tk.W)
    tree2.heading("12~24", text=cols[2], anchor=tk.W)
    tree2.heading("24~36", text=cols[3], anchor=tk.W)
    tree2.heading("36~60", text=cols[4], anchor=tk.W)
    tree2.heading("60~88", text=cols[5], anchor=tk.W)
    tree2.insert("", 0, values=("K2", "3.85~2.85", "2.8~2.0", "2.0~1.25", "1.6~1.0", "1.5~0.99"))
    tree2.grid(row=10, column=0, columnspan=2)
    # 摆线轮中心孔直径
    l31 = tk.Label(w1, text='摆线轮中心孔直径(D1=(0.4~0.5)Dp): ', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l31.grid(row=11, column=0)
    e31 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e31.grid(row=11, column=1, rowspan=2)  
    # 摆线轮厚度
    l41 = tk.Label(w1, text='摆线轮厚度(B=(0.05~0.1)Dp): ', bg='white',fg='black', font=('Arial', 12), width=30, height=2, anchor=tk.W)
    l41.grid(row=13, column=0)
    e41 = tk.Entry(w1, show=None, font=('Arial', 14), width=30)  # 显示成明文形式
    e41.grid(row=13, column=1, rowspan=2)
    # 放置lable的方法有：1）l.pack(); 2)l.place();

    # 输出结果文本
    linfo2 = tk.Label(w2, text='输出参数', bg='slategrey',fg='black',font=('Arial', 15), width=60, height=2)
    linfo2.grid(row=6, column=0, rowspan=2, columnspan=2)
    # 减速比/摆线轮齿数
    var1 = tk.StringVar()
    lo1 = tk.Label(w2, text='减速比/摆线轮齿数(Zc): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t1 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo1.grid(row=8, column=0)
    t1.grid(row=8, column=1)
    # 针轮齿数
    var2 = tk.StringVar()
    lo2 = tk.Label(w2, text='针轮齿数(Zp): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t2 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo2.grid(row=9, column=0)
    t2.grid(row=9, column=1)
    # 偏心距
    lo3 = tk.Label(w2, text='偏心距(a): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t3 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo3.grid(row=10, column=0)
    t3.grid(row=10, column=1)

    ## 针轮参数
    # 针齿中心圆直径
    lo4 = tk.Label(w2, text='针齿中心圆直径(Dp): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t4 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo4.grid(row=11, column=0)
    t4.grid(row=11, column=1)
    # 针齿销钉直径
    lo5 = tk.Label(w2, text='针齿销钉直径(drp): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t5 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo5.grid(row=12, column=0)
    t5.grid(row=12, column=1)
    # 针齿销钉套直径
    lo6 = tk.Label(w2, text='针齿销钉套直径(drp\'): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t6 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo6.grid(row=13, column=0)
    t6.grid(row=13, column=1)

    ## 摆线轮参数
    # 摆线轮中心孔直径
    lo7 = tk.Label(w2, text='摆线轮中心孔直径(D1): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t7 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo7.grid(row=14, column=0)
    t7.grid(row=14, column=1)
    # 柱销中心圆直径
    lo8 = tk.Label(w2, text='柱销中心圆直径(Dw): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t8 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo8.grid(row=15, column=0)
    t8.grid(row=15, column=1)
    # 摆线轮厚度
    lo9 = tk.Label(w2, text='摆线轮厚度(B): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t9 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo9.grid(row=16, column=0)
    t9.grid(row=16, column=1)

    ## 柱销参数
    # 摆线轮销孔直径
    lo10 = tk.Label(w2, text='摆线轮销孔直径(dw): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t10 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo10.grid(row=17, column=0)
    t10.grid(row=17, column=1)
    # 柱销数量
    lo11 = tk.Label(w2, text='柱销数量(Zw): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t11 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo11.grid(row=18, column=0)
    t11.grid(row=18, column=1)
    # 柱销直径
    lo12 = tk.Label(w2, text='柱销直径(dsw): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t12 = tk.Text(w2, width=30, height=1, bg='white',font=('Arial', 14))
    lo12.grid(row=19, column=0)
    t12.grid(row=19, column=1)
    # 柱销套直径
    lo13 = tk.Label(w2, text='柱销套直径(drw): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t13 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo13.grid(row=20, column=0)
    t13.grid(row=20, column=1)
    # 移据修型
    lo14 = tk.Label(w2, text='齿阔修型-移距修形量(dr_p): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t14 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo14.grid(row=21, column=0)
    t14.grid(row=21, column=1)
    # 等据修型
    lo15 = tk.Label(w2, text='齿阔修型-等距修形量(dr_rp): ', bg='white',fg='black',font=('Arial', 12), width=30, height=2, anchor=tk.W)
    t15 = tk.Text(w2, width=30, height=1, font=('Arial', 14))
    lo15.grid(row=22, column=0)
    t15.grid(row=22, column=1)
    # 空白行
    # 注释
    t16 = tk.Label(w2, text=' ', bg='white',fg='black',font=('Arial', 12), width=30, height=5, anchor=tk.W)
    t16.grid(row=23, column=0, columnspan=3)
    # 注释
    t17 = tk.Text(w2, width=90, height=5, font=('Arial', 11))
    t17.grid(row=24, column=0, columnspan=2)
    text1 = '注释: '
    text2 = '1. 柱销直径dsw也可以根据手册中的强度要求进行计算，但是太过复杂。'
    text3 = '2. 针齿销钉套直径(Drp + 2*delta)，delta指针齿套壁厚，一般取2~5 mm。'
    text4 = '3. dr_p, 负为直径变大，正为减小'
    text5 = '4. dr_rp, 负为直径变小，正为变大'
    t17.insert('insert', text1)
    t17.insert(tk.INSERT, '\n')
    t17.insert('insert', text2)
    t17.insert(tk.INSERT, '\n')
    t17.insert('insert', text3)
    t17.insert(tk.INSERT, '\n')
    t17.insert('insert', text4)
    t17.insert(tk.INSERT, '\n')
    t17.insert('insert', text5)
    # endregion

    def calculate():
        gam = int(e1.get())
        Dp = float(e2.get())
        dr = float(e4.get())
        # dsw = float(e3.get())
        K1 = e11.get()
        K2 = e21.get()
        B = e31.get()
        print(type(gam))

        try:
            if gam % 2 == 0:
                raise ValueError(('invalid value: gam should be odd number!'))
            if Dp <= 0:
                raise ValueError(('invalid value: Dp should be a positive number!'))
            # if dsw<=0:
            #     raise ValueError(('invalid value: dsw should be a positive number!'))
        except ValueError as e:
            print("Error: ",repr(e))
        else:
            # Zp: 针轮齿数, Zc: 摆线轮齿数
            Zp = gam + 1
            Zc = gam

            # region: K1 变幅系数; K2 针径系数
            if K1 == '':
                if Zc <= 11:
                    K1 = 0.42 + (0.55-0.42)*Zc/11
                elif Zc<=23 and Zc>13:
                    K1 = 0.48 + (0.74-0.48)*(Zc-13)/(23-13)
                elif Zc<=59 and Zc>23:
                    K1 = 0.65 + (0.9-0.65)*(Zc-25)/(59-25)
                elif Zc <=87 and Zc>61:
                    K1 = 0.7 + (0.9-0.7)*(Zc-61)/(87-61)
            else:
                K1 = float(K1)
                print(K1)
            if K2 == '':
                if Zp < 12:
                    K2 = 3.85 - (3.85-2.85)*Zp/12
                elif Zp <24 and Zp>=12:
                    K2 = 2.8 - (2.8-2.0)*(Zp-12)/(24-12)
                elif Zp <36 and Zp>=24:
                    K2 = 2.0 - (2.0-1.25)*(Zp-24)/(36-24)
                elif Zp <60 and Zp>=36:
                    K2 = 1.6 - (1.6-1.0)*(Zp-36)/(60-36)
                elif Zp <88 and Zp>=60:
                    K2 = 1.5 - (1.5-0.99)*(Zp-60)/(88-60)
            else:
                K2 = float(K2)
                print(K2)
            # endregion

            # region: 柱销数量
            if Dp<100:
                Zw = 6
            elif Dp<200 and Dp>=100:
                Zw = 8
            elif Dp<300 and Dp>=200:
                Zw = 10
            else:
                Zw = 12
            # endregion

            if Dp<=550:
                delta = 0.15
            else:
                delta = 0.25

            if B == '':
                B = 0.15*Dp/2
            else:
                B = float(B)

            D1 = 0.4*Dp         # 摆线轮中心孔直径
            a = K1*Dp/(2*Zp)    # 偏心距
            D_rp = Dp/K2*np.sin(np.pi/Zp)         # 针径销钉套直径
            D_fc = Dp - 2*a - D_rp              # 摆线轮齿根圆直径
            Dw = (D_fc + D1)/2                  # 柱销中心园直径
            # drw = 1.4*dsw                       # 柱销套直径
            # dw = drw + 2*a+delta                      # 摆线轮销孔直径 

            dw1 = Dw - D1 - 2*0.03*Dp
            dw2 = Dw*np.sin(np.pi/Zw) - 0.03*Dp
            dw =min(dw1, dw2)
            drw = dw - 2*a - delta
            dsw = drw - 4

            # 齿阔线修型
            # dr = 0.02       # 径向间隙
            dr_rp = dr/(1-np.sqrt(1-K1**2))
            dr_p = -dr*np.sqrt(1-K1**2)/(1-np.sqrt(1-K1**2))

            # t1.insert('insert', gam)
            # t2.insert('insert', Zp)
            # t3.insert('insert', a)
            # t4.insert('insert', Dp)
            # t5.insert('insert', D_rp)
            # t6.insert('insert', D_rp+4)
            # t7.insert('insert', D1)
            # t8.insert('insert', Dw)
            # t9.insert('insert', B)
            # t10.insert('insert', dw)
            # t11.insert('insert', Zw)
            # t12.insert('insert', dsw)
            # t13.insert('insert', drw)
            tt = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,  t14, t15]
            params = [gam, Zp, a, Dp, D_rp-4, D_rp, D1, Dw, B, dw, Zw, dsw, drw,dr_p,dr_rp]


            for i in range(len(tt)):
                    tt[i].delete('1.0', 'end')
            for i in range(len(tt)):
                    tt[i].insert('insert', round(params[i],2))
            
        pass
    b1 = tk.Button(w1, text='计算', width=15, height=2, command=calculate)
    b1.grid(row=15, column=0, columnspan=2)
    # 第6步，主窗口循环显示
    window.mainloop()
    pass


if __name__ == "__main__":
    # ParamsCal(29, 100, 6)
    ParamsCalGUI()