#
import vtk
import osqp
import numpy as np
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from rota import rota, drota
#
def dcol_lstosq(xk0,xk1,pnt0,pnt1,c_a,c_l,sol_flg):
#
#   transform the points (about 0,0,0)
#    
#   tmp = np.array([xk0[1],xk0[2],xk0[3]])
#   if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
#   else: tmp=tmp*0.
#
#   rot=R.from_rotvec((c_a*xk0[0])*tmp).as_matrix().T
#   tve0=np.dot(pnt0,rot)
#
#   rot=R.from_rotvec((c_a*xk0[0])*tmp).as_quat()#.as_matrix().T
#   rot_tmp=rot.copy()
#   rot_tmp[-1]=rot[0]
#   rot_tmp[0]=rot[-1]
#   rot_tmp[1]=rot[0]
#   rot_tmp[2]=rot[1]
#   rot_tmp[3]=rot[2]
#
    qua0=xk0[:4]#/np.linalg.norm(xk0[:4])
    tve0=rota(qua0,pnt0)
    qua1=xk1[:4]#/np.linalg.norm(xk1[:4])
    tve1=rota(qua1,pnt1)
#
    [dtve0dr,dtve0di,dtve0dj,dtve0dk]=drota(qua0,pnt0)
    [dtve1dr,dtve1di,dtve1dj,dtve1dk]=drota(qua1,pnt1)
#
#   tmp = np.array([xk1[1],xk1[2],xk1[3]])
#   if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
#   else: tmp=tmp*0.
#   rot=R.from_rotvec((c_a*xk1[0])*tmp).as_matrix().T
#   tve1=np.dot(pnt1,rot)
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
    dg=grad(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#
    obj = osqp.OSQP()
#
    H=sparse.block_diag(  [sparse.csc_matrix((nut,nut)), 2.*sparse.eye(3) ], format='csc')
#
    q=np.zeros(nut+3) # evaluated at zero
#
    Ad0 = sparse.csc_matrix(np.append(tve0[:,0]/scl,-tve1[:,0]/scl))
    Ad1 = sparse.csc_matrix(np.append(tve0[:,1]/scl,-tve1[:,1]/scl))
    Ad2 = sparse.csc_matrix(np.append(tve0[:,2]/scl,-tve1[:,2]/scl))
#
    A = sparse.bmat([[  Ad0    ,-np.array([1.,0.,0.])],
                     [  Ad1    ,-np.array([0.,1.,0.])],
                     [  Ad2    ,-np.array([0.,0.,1.])],
                     [  dg[1]  ,None],
                     [  dg[2]  ,None],
                     [sparse.eye(nut), None]],format='csc')
#
    b = c_l*np.array([ck1[0] - ck0[0], ck1[1]-ck0[1], ck1[2]-ck0[2]])/scl
#
    l = np.hstack([b, 0., 0., np.zeros(nut)])
    u = np.hstack([b, 1., 1., np.ones(nut)])
#
    obj.setup(H,q,A,l,u,verbose=False,eps_abs=1e-12,eps_rel=1e-12)#,max_iter=int(1e6),polish=True)
#
    sol=obj.solve()
#
    xt=sol.x[:nut]
    g=func(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#
    xt0=xt[nuts[0]:nuts[1]]
    xt1=xt[nuts[1]:nuts[2]]
#
    pos0=np.dot(xt0,tve0)+c_l*ck0
    pos1=np.dot(xt1,tve1)+c_l*ck1
#
    dc=c_l[0]/scl*sol.y[0]
#
#   print(sol.y[:3])
#   print(np.dot(xt0,dtve0dr[:,0])*scl)
#   print(np.dot(xt0,dtve0dr[:,1])*scl)
#   print(np.dot(xt0,dtve0dr[:,2])*scl)
#   stop
    dr=sol.y[0]*np.dot(xt0,dtve0dr[:,0])/scl
    dr=dr+sol.y[1]*np.dot(xt0,dtve0dr[:,1])/scl
    dr=dr+sol.y[2]*np.dot(xt0,dtve0dr[:,2])/scl
#
    return g[0]*scl, xt, pos0, pos1, dr, dc
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
