import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=200)

def print_state(time_value, cH, cOH, cK, cNO3):
    print(f"t = {time_value:.6f} s")
    print("cH =", cH)
    print("cOH =", cOH)
    print("cK =", cK)
    print("cNO3 =", cNO3)
    print()

F=96485.33212
R=8.314462618
Temp=298.15
I = 20e-6 
A_liquid = 7e-6
kb=1e-8
U_solution=0.2
L=0.01
N=100
dx=L/N

x_face=np.linspace(0,L,N+1)
x_center= x_face[:-1]+dx/2

T=600
dt=T/6000
t=np.linspace(0,T,6000+1)

cH=np.full(N,1e-4)
cOH=np.full(N,1e-4)
cK=np.full(N,10.0)
cNO3=np.full(N,10.0)

D_H = 9.31e-9
D_OH = 5.27e-9
D_K = 1.96e-9
D_NO3 = 1.90e-9

c=np.array([cH,cOH,cK,cNO3])
pH=-np.log10(cH/1000)
J_H=np.zeros(N+1)
J_OH=np.zeros(N+1)
J_K=np.zeros(N+1)
J_NO3=np.zeros(N+1)

J_source=I/(F*A_liquid)

weight=np.full(N-1,dx)
weight[0]+=dx/2         
weight[-1]+=dx/2

steps=len(t) - 1
print_state(t[0], cH, cOH, cK, cNO3)
for step in range(steps):
    
    cH_mean=(cH[1:]+cH[:-1])/2
    cOH_mean=(cOH[1:]+cOH[:-1])/2
    cK_mean=(cK[1:]+cK[:-1])/2
    cNO3_mean=(cNO3[1:]+cNO3[:-1])/2
    keppa=(F)**2*(D_H*cH_mean+D_OH*cOH_mean+D_K*cK_mean+D_NO3*cNO3_mean)/(R*Temp)
    
    cH_diff=(cH[1:]-cH[:-1])/dx
    cOH_diff=(cOH[1:]-cOH[:-1])/dx
    cK_diff=(cK[1:]-cK[:-1])/dx
    cNO3_diff=(cNO3[1:]-cNO3[:-1])/dx
    j_diff=-F*(D_H*cH_diff-D_OH*cOH_diff+D_K*cK_diff-D_NO3*cNO3_diff)

    integral_diff=np.sum(weight*j_diff/keppa)
    integral_j=np.sum(weight/keppa)

    j=(U_solution+integral_diff)/integral_j
    E=(j-j_diff)/keppa
    I=j*A_liquid

    
    J_H[1:-1]=-D_H*cH_diff+D_H*(F)*cH_mean*E/(R*Temp)
    J_OH[1:-1]=-D_OH*cOH_diff-D_OH*(F)*cOH_mean*E/(R*Temp)
    J_K[1:-1]=-D_K*cK_diff+D_K*(F)*cK_mean*E/(R*Temp)
    J_NO3[1:-1]=-D_NO3*cNO3_diff-D_NO3*(F)*cNO3_mean*E/(R*Temp)

    J_source=j/F
    J_H[0] = J_source
    J_OH[-1] = -J_source

    cH_new=cH+dt*(J_H[:-1]-J_H[1:])/dx
    cOH_new=cOH+dt*(J_OH[:-1]-J_OH[1:])/dx
    cK_new=cK+dt*(J_K[:-1]-J_K[1:])/dx
    cNO3_new=cNO3+dt*(J_NO3[:-1]-J_NO3[1:])/dx

    if (np.any(cH_new<=0) or np.any(cOH_new<=0) or np.any(cK_new<=0) or np.any(cNO3_new<=0)):
        break
    b=cH_new-cOH_new
    c_1=(np.abs(b)+np.sqrt(b**2+4*kb))/(2)
    c_2=kb/c_1

    cH=np.where(b>=0,c_1,c_2)
    cOH=np.where(b>=0,c_2,c_1)
    cK=cK_new
    cNO3=cNO3_new
    print_state(t[step + 1], cH, cOH, cK, cNO3)
