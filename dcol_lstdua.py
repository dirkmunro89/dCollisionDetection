#
import vtk
import osqp
import numpy as np
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
#
def dual_func(x_d,nut,nuts,tve0,tve1,ck0,ck1,c_l,scl):
#
    x=dual_uptd(x_d,nut,nuts,tve0,tve1,ck0,ck1,c_l,scl)
#
#   half time derivative; we start at zeros everywhere
#
    W=1e0*np.dot(x[-3:],x[-3:])+1e-3*np.dot(x[:nut],x[:nut]) #+1e-6*np.sum(x[:nut])
#
    W=W+x_d[0]*(np.dot(np.append(tve0[:,0]/scl,-tve1[:,0]/scl),x[:nut]))
    W=W+x_d[0]*(-x[-3]+ck0[0]/scl*c_l[0]-ck1[0]/scl*c_l[0])
#
    W=W+x_d[1]*(np.dot(np.append(tve0[:,1]/scl,-tve1[:,1]/scl),x[:nut]))
    W=W+x_d[1]*(-x[-2]+ck0[1]/scl*c_l[1]-ck1[1]/scl*c_l[1])
#
    W=W+x_d[2]*(np.dot(np.append(tve0[:,2]/scl,-tve1[:,2]/scl),x[:nut]))
    W=W+x_d[2]*(-x[-1]+ck0[2]/scl*c_l[2]-ck1[2]/scl*c_l[2])
#
    W=W+x_d[3]*(np.sum(x[nuts[0]:nuts[1]])-1.)#/nut
    W=W+x_d[4]*(np.sum(x[nuts[1]:nuts[2]])-1.)#/nut
#
    return -W
#
def dual_grad(x_d,nut,nuts,tve0,tve1,ck0,ck1,c_l,scl):
#
    x=dual_uptd(x_d,nut,nuts,tve0,tve1,ck0,ck1,c_l,scl)
#
    dW=np.zeros(5)
#
    dW[0]=np.dot(np.append(tve0[:,0]/scl,-tve1[:,0]/scl),x[:nut])-x[-3]+(ck0[0]-ck1[0])/scl*c_l[0]
    dW[1]=np.dot(np.append(tve0[:,1]/scl,-tve1[:,1]/scl),x[:nut])-x[-2]+(ck0[1]-ck1[1])/scl*c_l[1]
    dW[2]=np.dot(np.append(tve0[:,2]/scl,-tve1[:,2]/scl),x[:nut])-x[-1]+(ck0[2]-ck1[2])/scl*c_l[2]
    dW[3]=(np.sum(x[nuts[0]:nuts[1]])-1.)#/nut
    dW[4]=(np.sum(x[nuts[1]:nuts[2]])-1.)#/nut
#
    return -dW
#
def dual_uptd(x_d,nut,nuts,tve0,tve1,ck0,ck1,c_l,scl):
#
    x=np.zeros(nut+3)
#
    x[-3]=x_d[0].copy()/1e0
    x[-2]=x_d[1].copy()/1e0
    x[-1]=x_d[2].copy()/1e0
#
    x[:nut]=-x_d[0]*np.append(tve0[:,0]/scl,-tve1[:,0]/scl)
    x[:nut]=x[:nut]-x_d[1]*np.append(tve0[:,1]/scl,-tve1[:,1]/scl)
    x[:nut]=x[:nut]-x_d[2]*np.append(tve0[:,2]/scl,-tve1[:,2]/scl)
#
    x[:nut]=x[:nut]-np.append(x_d[3]*np.ones(nuts[1]-nuts[0]),x_d[4]*np.ones(nuts[2]-nuts[1]))#/nut
#
    x[:nut]=np.minimum(np.maximum(-np.ones(nut)*0.+(x[:nut])/1e-3,np.zeros(nut)),1.)
#
    return x
#
def dcol_lstdua(xk0,xk1,pnt0,pnt1,c_a,c_l,sol_flg):
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
    c_p1=None
    c_p2=None
#
    bds=[[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e4,1e4],[-1e4,1e4]]; tup_bds=tuple(bds)
    print(bds)
    sol=minimize(dual_func,np.array([0e0,0e0,0e0,0e0,0e0]),args=(nut,nuts,tve0,tve1,ck0,ck1,c_l,scl), \
        jac=None,method='L-BFGS-B',bounds=tup_bds, \
        options={'disp':True,'gtol':1e-32,'ftol':1e-32,'maxls':1000000})
#       options={'disp':True,'maxcor':1,'gtol':1e-32,'ftol':1e-32,'maxls':1000})
#   sol=minimize(dual_func,sol.x,args=(nut,nuts,tve0,tve1,ck0,ck1,c_l,scl), \
#       jac=dual_grad,method='L-BFGS-B',bounds=tup_bds, \
#       options={'disp':True,'gtol':1e-32,'ftol':1e-32,'maxls':1000})
#
    print(sol.x)
    x=dual_uptd(sol.x,nut,nuts,tve0,tve1,ck0,ck1,c_l,scl)
    xt0=x[nuts[0]:nuts[1]]
    xt1=x[nuts[1]:nuts[2]]
    print(sum(xt0))
    print(sum(xt1))
    print(x)
#
    xt=x[:nut]
    g=func(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#
    xt0=xt[nuts[0]:nuts[1]]
    xt1=xt[nuts[1]:nuts[2]]
#
    pos0=np.dot(xt0,tve0)+c_l*ck0
    pos1=np.dot(xt1,tve1)+c_l*ck1
#
    return [np.sqrt(max(g[0],0.)*scl), xt, pos0, pos1]
#
def hess(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl):
#
    ddf=np.zeros((len(xt),len(xt)))
#
    ddf[nuts[0]:nuts[1],nuts[0]:nuts[1]]=2*np.dot(tve0,tve0.T)
    ddf[nuts[1]:nuts[2],nuts[1]:nuts[2]]=2*np.dot(tve1,tve1.T)
    ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]]=-2*np.dot(tve1,tve0.T)
    ddf[nuts[0]:nuts[1],nuts[1]:nuts[2]]=ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]].T
#
    return ddf/scl
#
def grad(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl):
#
    df=np.zeros((3,len(xt)))
#
#   xt0=xt[nuts[0]:nuts[1]]
#   xt1=xt[nuts[1]:nuts[2]]
#   pos0=np.dot(xt0,tve0)+c_l*ck0
#   pos1=np.dot(xt1,tve1)+c_l*ck1
#
#   dposdt[nuts[0]:nuts[1]]=pnts[col[0]]#np.dot(pnt,rot)
#   dposdt[nuts[1]:nuts[2]]=pnts[col[1]]#np.dot(pnt,rot)
#
#   dis=pos0-pos1
#   df[0][nuts[0]:nuts[1]]=2.*np.dot(tve0,dis)/scl
#   df[0][nuts[1]:nuts[2]]=-2.*np.dot(tve1,dis)/scl
#
    df[1][nuts[0]:nuts[1]]=np.ones(nuts[1]-nuts[0])
    df[2][nuts[1]:nuts[2]]=np.ones(nuts[2]-nuts[1])
#
    return df 
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
    f=np.zeros(3)
    tmp=0.
    f[1]=(np.sum(xt0)-1.)
    f[2]=(np.sum(xt1)-1.)
#
#   if sol_flg == 1:
#       g0 = max(g0, -c_p1[0]/2./c_p2)
#       g1 = max(g1, -c_p1[1]/2./c_p2)
#
    dis=pos0-pos1
    f[0]=np.dot(dis,dis.T)/scl#+c_p1[0]*g0+c_p1[1]*g1 + c_p2*g0**2. + c_p2*g1**2.
#
    return f
