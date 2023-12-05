#
import vtk
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
#
def dcol_auglag(xk0,xk1,pnt0,pnt1,c_a,c_l,sol_flg):
#
#   transform the points (about 0,0,0)
#
    tmp = np.array([xk0[1],xk0[2],xk0[3]])
    if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
    else: tmp=tmp*0.
    rot=R.from_rotvec((c_a*xk0[0])*tmp).as_matrix().T
    tve0=np.dot(pnt0,rot)
#
    tmp = np.array([xk1[1],xk1[2],xk1[3]])
    if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
    else: tmp=tmp*0.
    rot=R.from_rotvec((c_a*xk1[0])*tmp).as_matrix().T
    tve1=np.dot(pnt1,rot)
#
    nut=len(tve0)+len(tve1)
    nuts=[0,len(tve0),nut]
#
    ck0=xk0[4:7]
    ck1=xk1[4:7]
#
    xt=np.zeros(nut)
    scl=np.linalg.norm(c_l*ck0-c_l*ck1)**2.
#
    bds=[[0.,1.] for i in range(nut)]; tup_bds=tuple(bds)
    c_p2=1.#np.array([1.,1.])
    c_p1=np.array([0.,0.])
#
    c_p1_old=np.ones_like(c_p1)
    xtold=np.ones_like(xt)
    g0=1
    g1=1
    while np.linalg.norm(xt-xtold) > 1e-3: #or max(g0,g1)>1e-6 or c_p2 < 1e2:
#
        xtold[:]=xt
        sol=minimize(func,xt,args=(ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl),\
#           bounds=tup_bds,jac=grad,method='TNC')#,options={'ftol':1e-16,'gtol':1e-16})
            bounds=tup_bds,jac=grad,method='L-BFGS-B',options={'ftol':1e-16,'gtol':1e-16})
#
        xt=sol.x
        xt0=xt[nuts[0]:nuts[1]]
        xt1=xt[nuts[1]:nuts[2]]
        g0=(np.sum(xt0)-1.)
        g1=(np.sum(xt1)-1.)
#
#       augmented lagrangian see Rao 461-462
#
        c_p1_old[:]=c_p1
        c_p1[0]=c_p1[0]+2*c_p2*g0
        c_p1[1]=c_p1[1]+2*c_p2*g1
        c_p2=c_p2*2.
#
    xt0=xt[nuts[0]:nuts[1]]
    xt1=xt[nuts[1]:nuts[2]]
    pos0=np.dot(xt0,tve0)+c_l*ck0
    pos1=np.dot(xt1,tve1)+c_l*ck1
#
    return [np.sqrt(max(sol.fun,0.)*scl), sol.x, pos0, pos1]
#
def grad(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl):
#
    g=np.zeros_like(xt)
#   dposdt=np.zeros((len(xt),3))
#
#   xk0_c=xk[7*pi[0]+4:7*pi[0]+7]
#   xk1_c=xk[7*pi[1]+4:7*pi[1]+7]
#   pnts0=pnts[pi[0]]
#   pnts1=pnts[pi[1]]
#
    xt0=xt[nuts[0]:nuts[1]]
    xt1=xt[nuts[1]:nuts[2]]
    pos0=np.dot(xt0,tve0)+c_l*ck0
    pos1=np.dot(xt1,tve1)+c_l*ck1
#
#   dposdt[nuts[0]:nuts[1]]=pnts[col[0]]#np.dot(pnt,rot)
#   dposdt[nuts[1]:nuts[2]]=pnts[col[1]]#np.dot(pnt,rot)
#
    dis=pos0-pos1
    g[nuts[0]:nuts[1]]=2.*np.dot(tve0,dis)/scl
    g[nuts[1]:nuts[2]]=-2.*np.dot(tve1,dis)/scl
#
    g0=(np.sum(xt0)-1.)
    g1=(np.sum(xt1)-1.)
#   if g0 + eqf*1e8 > 0:
    if sol_flg == 1:
        g0 = max(g0, -c_p1[0]/2./c_p2)
        g1 = max(g1, -c_p1[1]/2./c_p2)
#
    g[nuts[0]:nuts[1]]=g[nuts[0]:nuts[1]]+2.*g0*c_p2+c_p1[0]
#   if g1 + eqf*1e8 > 0:
    g[nuts[1]:nuts[2]]=g[nuts[1]:nuts[2]]+2.*g1*c_p2+c_p1[1]
#
    return g
#
def func(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl):
#
    xt0=xt[nuts[0]:nuts[1]]
    xt1=xt[nuts[1]:nuts[2]]
#   xk0_c=xk[7*pi[0]+4:7*pi[0]+7]
#   xk1_c=xk[7*pi[1]+4:7*pi[1]+7]
#   pnts0=pnts[pi[0]]
#   pnts1=pnts[pi[1]]
#
#   pos0=np.dot(xt0,pnts0)+c_l*xk0_c#[7*pi[0]+4:7*pi[0]+7]
#   pos1=np.dot(xt1,pnts1)+c_l*xk1_c#[7*pi[1]+4:7*pi[1]+7]
    pos0=np.dot(xt0,tve0)+c_l*ck0
    pos1=np.dot(xt1,tve1)+c_l*ck1
#
#   for later
#   make each t ---> t_i*( t_t0 + t_t1 + t_t2 )**3.  
#   ---> I think it remains convexish. so, then, if triangle does not sum to 1, 
#   then the t does nothing--> 0. if it does, then the t counts.
#   else we do convex decompositions
#
    tmp=0.
    g0=(np.sum(xt0)-1.)
    g1=(np.sum(xt1)-1.)
#
    if sol_flg == 1:
        g0 = max(g0, -c_p1[0]/2./c_p2)
        g1 = max(g1, -c_p1[1]/2./c_p2)
#
    dis=pos0-pos1
    f=np.dot(dis,dis.T)/scl+c_p1[0]*g0+c_p1[1]*g1 + c_p2*g0**2. + c_p2*g1**2.
#
    return f
