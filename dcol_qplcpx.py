#
import vtk
import osqp
import numpy as np
import cplex
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
#
def dcol_qplcpx(xk0,xk1,pnt0,pnt1,c_a,c_l,sol_flg):
#
#   transform the points (about 0,0,0)
#
#   tmp = np.array([xk0[1],xk0[2],xk0[3]])
#   if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
#   else: tmp=tmp*0.
#   rot=R.from_rotvec((c_a*xk0[0])*tmp).as_matrix().T
#   tve0=np.dot(pnt0,rot)
#
#   tmp = np.array([xk1[1],xk1[2],xk1[3]])
#   if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
#   else: tmp=tmp*0.
#   rot=R.from_rotvec((c_a*xk1[0])*tmp).as_matrix().T
#   tve1=np.dot(pnt1,rot)
#
    qua=xk0[:4]#/np.linalg.norm(xk0[:4])
    tve0=rota(qua,pnt0)
    qua=xk1[:4]#/np.linalg.norm(xk1[:4])
    tve1=rota(qua,pnt1)
#
    [dtve0dr,dtve0di,dtve0dj,dtve0dk]=drota(qua,pnt0)
    [dtve1dr,dtve1di,dtve1dj,dtve1dk]=drota(qua,pnt1)
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
    dx_l=np.zeros(nut)-xt
    dx_u=np.ones(nut)-xt
#
    g=func(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
    dg=grad(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
    ddg=hess(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#
    G_tmp=np.zeros((nut,nut),dtype=np.float64); np.fill_diagonal(G_tmp,-1e0)
    u=-g[1:]; l=0.*np.ones(2,dtype=float)
#
#
    tmp=np.zeros((nut,nut),dtype=np.float64); np.fill_diagonal(tmp,1e0)
    A=sparse.csc_matrix(np.append(dg[1:],tmp,axis=0))
    u=-g[1:]; l=-np.inf*np.ones(2,dtype=float)
    l=np.append(l,dx_l); u=np.append(u,dx_u)
#
#   A=sparse.csc_matrix(dg[1:])
#   u=-g[1:]; l=np.zeros(2,dtype=float)
#   try to do A as proper sparse matrix.
#   do not need upper bounds, but dont think it matters
#   can we get rid of all the bounds...? lets try
#
    prb=cplex.Cplex()
#
    prb.variables.add(obj=dg[0][:], lb=dx_l)
    ind = range(nut)#[i for i in range(n)]
    lin_expr=[]
    for j in range(2): lin_expr.append( cplex.SparsePair( ind = ind, val=dg[j+1] ))
    rhs=-g[1:]
#
    prb.linear_constraints.add(lin_expr=lin_expr,rhs=rhs, senses=['L']*2)
#
    prb.objective.set_quadratic(ddg)
#
    prb.set_results_stream(None)
    prb.set_log_stream(None)
#
    prb.solve()
    xt[:]=prb.solution.get_values()
#
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
    nut=len(xt)
#
    ddf[nuts[0]:nuts[1],nuts[0]:nuts[1]]=2*np.dot(tve0,tve0.T)/scl
    ddf[nuts[1]:nuts[2],nuts[1]:nuts[2]]=2*np.dot(tve1,tve1.T)/scl
    ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]]=-2*np.dot(tve1,tve0.T)/scl
    ddf[nuts[0]:nuts[1],nuts[1]:nuts[2]]=ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]].T
#
    tmp=sparse.lil_matrix(ddf)
    ddf_lst=list(zip(tmp.rows,tmp.data))
#
    return ddf_lst
#
def grad(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl):
#
    df=np.zeros((3,len(xt)))
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
    df[0][nuts[0]:nuts[1]]=2.*np.dot(tve0,dis)/scl
    df[0][nuts[1]:nuts[2]]=-2.*np.dot(tve1,dis)/scl
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
