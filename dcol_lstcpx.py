#
import vtk
import cplex
import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
#
from rota import rota,drota
#
def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP
#
def dcol_lstcpx(xk0,xk1,pnt0,pnt1,c_a,c_l,sol_flg):
#
#   transform the points (about 0,0,0)
#
    tmp = xk0[:4]
#   if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
#   else: tmp=tmp*0.
    qua=xk0[:4]#/np.linalg.norm(xk0[:4])
    tve0=rota(qua,pnt0)
#   rot=R.from_rotvec((c_a*xk0[0])*tmp).as_matrix().T
#   tve0=np.dot(pnt0,rot)
    dPdr = drota(qua,pnt0)
#
    tmp = xk1[:4]
#   if np.linalg.norm(tmp): tmp = tmp/np.linalg.norm(tmp)
#   else: tmp=tmp*0.
#   rot=R.from_rotvec((c_a*xk1[0])*tmp).as_matrix().T
#   tve1=np.dot(pnt1,rot)
    qua=xk1[:4]#/np.linalg.norm(xk1[:4])
    tve1=rota(qua,pnt1)
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
#   g=func(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
    dg=grad(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#   ddg=hess(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#
    G = sparse.bmat([ [-np.eye(nut,nut) , np.zeros((nut,3))],
                      [   dg[1]         , None            ],
                      [   dg[2]         , None            ]])
    h = np.hstack( (-dx_l,1.,1.))
#
    Ad0 = sparse.csc_matrix(np.append(tve0[:,0]/scl,-tve1[:,0]/scl))
    Ad1 = sparse.csc_matrix(np.append(tve0[:,1]/scl,-tve1[:,1]/scl))
    Ad2 = sparse.csc_matrix(np.append(tve0[:,2]/scl,-tve1[:,2]/scl))
    A = sparse.bmat([[  Ad0    ,-np.array([1.,0.,0.])],
                     [  Ad1    ,-np.array([0.,1.,0.])],
                     [  Ad2    ,-np.array([0.,0.,1.])]],format='csc')
#
#
#   H=sparse.block_diag(  [sparse.csc_matrix((nut,nut)), 2.*sparse.eye(3) ], format='csc')
    H=sparse.block_diag(  [0e0*sparse.eye(nut), 2.*sparse.eye(3) ], format='csc')
    q=np.zeros(nut+3) 
    solvers.options['show_progress']=True
    solvers.options['refinement']=True
    solvers.options['abstol']=1e-32
    solvers.options['maxiters']=int(1e6)
#
#   H=scipy_sparse_to_spmatrix(H)
#   G=scipy_sparse_to_spmatrix(G)
#   q=matrix(q,tc='d')
#   h=matrix(h,tc='d')
#   A=scipy_sparse_to_spmatrix(A)
#   b=matrix(b,tc='d')
#
    lb=np.hstack((dx_l, -np.inf, -np.inf, -np.inf ))
#
    prb=cplex.Cplex()
    prb.variables.add(q,lb=lb)
#
    lin_expr=[]
#
    ind0=list(range(nut)) + [nut]
    val0=np.append(tve0[:,0]/scl,-tve1[:,0]/scl)
    val0=np.append(val0,-1)
    lin_expr.append( cplex.SparsePair(  ind=ind0, val=val0  )   )
#
    ind1=list(range(nut)) + [nut+1]
    val1=np.append(tve0[:,1]/scl,-tve1[:,1]/scl)
    val1=np.append(val1,-1)
    lin_expr.append( cplex.SparsePair(  ind=ind1, val=val1  )   )
#
    ind2=list(range(nut)) + [nut+2]
    val2=np.append(tve0[:,2]/scl,-tve1[:,2]/scl)
    val2=np.append(val2,-1)
    lin_expr.append( cplex.SparsePair(  ind=ind2, val=val2  )   )
#
    ind3=list(range(nuts[0],nuts[1]))
    val3=dg[1][nuts[0]:nuts[1]]#np.append(tve0[:,2]/scl,-tve1[:,2]/scl)
    lin_expr.append( cplex.SparsePair(  ind=ind3, val=val3  )   )
#
    ind4=list(range(nuts[1],nuts[2]))
    val4=dg[2][nuts[1]:nuts[2]]#np.append(tve0[:,2]/scl,-tve1[:,2]/scl)
    lin_expr.append( cplex.SparsePair(  ind=ind4, val=val4  )   )
#
    b = c_l*np.array([ck1[0] - ck0[0], ck1[1]-ck0[1], ck1[2]-ck0[2]])/scl
#
    b=np.hstack((b,1.,1.))
#
    prb.linear_constraints.add(lin_expr=lin_expr,rhs=b,senses=['E','E','E','E','E'])
#
    H=np.append(1e-6*np.ones(nut),2.*np.ones(3))
#
    prb.objective.set_quadratic(H)
#
    prb.set_results_stream(None)
    prb.set_log_stream(None)
#
#   print(prb.parameters.qpmethod.help())
#   prb.parameters.qpmethod.set(prb.parameters.qpmethod.values.barrier)
    prb.parameters.barrier.convergetol.set(1e-12)
#   stop
    prb.solve()
#
    dr = 1.
    dc=-c_l/scl*prb.solution.get_dual_values()[:3]
#
    xt=np.array(prb.solution.get_values())[:nut]
    g=func(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl)
#
    xt0=xt[nuts[0]:nuts[1]]
    xt1=xt[nuts[1]:nuts[2]]
#
    pos0=np.dot(xt0,tve0)+c_l*ck0
    pos1=np.dot(xt1,tve1)+c_l*ck1
#
    return [g[0]*scl, xt, pos0, pos1,dr,dc]
#
def hess(xt,ck0,ck1,tve0,tve1,nuts,c_a,c_l,c_p1,c_p2,sol_flg,scl):
#
    ddf=np.zeros((len(xt),len(xt)))
#
    ddf[nuts[0]:nuts[1],nuts[0]:nuts[1]]=2*np.dot(tve0,tve0.T)
    ddf[nuts[1]:nuts[2],nuts[1]:nuts[2]]=2*np.dot(tve1,tve1.T)
    ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]]=-2*np.dot(tve1,tve0.T)
#
    return ddf/scl
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
