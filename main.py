#
import os
import sys
import vtk
import time
import numpy as np
import logging as log
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
#
import matplotlib.pyplot as plt
#
from dist import dist
from dcol_auglag import dcol_auglag
from dcol_qplcvx import dcol_qplcvx
from dcol_qplcpx import dcol_qplcpx
from dcol_qplosq import dcol_qplosq
from dcol_lstcvx import dcol_lstcvx
from dcol_lstosq import dcol_lstosq
from dcol_lstcpx import dcol_lstcpx
from dcol_lstdua import dcol_lstdua
#
from init import init#, pretfms6, pretfms24
#
from util import tfmx, tran, appdata, woutfle
#
if __name__ == "__main__":
#
#   parameters
#
    c_l=np.array([300.,300.,300.]) # for in box
    c_s=1.01
    c_a=np.pi 
    c_e=1000
#
    t0=0
    out='./out_%d/'%t0
    if not os.path.isdir(out):
        os.makedirs(out)
#
    level=log.INFO
    format   = '%(message)s'
    handlers=[log.FileHandler('history_%d.log'%t0), log.StreamHandler()]
    log.basicConfig(level=level, format=format, handlers=handlers)
#
    t0=time.time()
#
#   hard code system arguments
#
    log.info('='*87)
    sol_flg=int(sys.argv[1])
#
    log.info('sol_flg: %d'%sol_flg)
#
    sys.argv=['main.py', 'objall', '0', '2', 'stl/Fone.stl']
#
    tmp=" ".join(sys.argv)
    c=0
    while True:
        try: tmp[c+87]
        except: break
        log.info(tmp[c:c+87])
        c=c+87
#
    log.info(tmp[c:])
    log.info('-'*87)
    log.info('Writing output to:\n%s'%out)
    log.info('='*87)
#
    opt_str=sys.argv[1]
    vis_flg=int(sys.argv[2])
#
    c=0
    nums=[]; flns=[]
    while True:
        try: sys.argv[c+3]
        except: break
        nums.append(int(sys.argv[c+3]))
        flns.append(sys.argv[c+4])
        c=c+2
#
    nobj=int(c/2) # number of unique parts (objects)
#
    objs=[]; tfms=[]; maps=[]
#
    n=0 
    c_v_0=0.
    c_v_1=0.
#
    for i in range(nobj):
#
        log.info('='*87)
        log.info('Pack %2d of %10s: '%(nums[i],flns[i]))
        log.info('-'*87)
#
#       make the object from the input file
#
        obj=init(i,flns[i],c_e,c_s,log,0)
#
#       append to a list of the unique objects in the build
#
        objs.append(obj)
#
#       make a transform for each instance of the object in the build
#       and a list which maps back to the object id
#
        for j in range(nums[i]):
#
            tfm=vtk.vtkTransform()
            tfm.PostMultiply()
            tfm.Translate(0., 0., 0.)
            tfm.Update()
            tfms.append(tfm)
#
            maps.append(i)
#
            c_v_0=c_v_0+obj.bbv
            c_v_1=c_v_1+obj.vol
            n=n+1
#
    log.info('='*87)
    log.info('Total volume of AABB volumes : %7.3e'%(c_v_0))
    log.info('Total volume                 : %7.3e'%(c_v_1))
    log.info('Efficiency                   : %7.3e'%(c_v_1/c_v_0))
    log.info('='*87)
#
    log.info('='*87)
#
    pnts = [obj.pts for obj in objs]
    vtps = [obj.vtp for obj in objs]
#
    c_r=[]
#
    log.info('%6s%16s%16s%16s%16s%16s'%('k','DistT','DistP','Error','Viol.','Time (s)'))
    log.info('-'*87)
#
####################################################
#
    pi=[0,1] # global part indices
#
#   the transforms
#
    tt=0.
    ae=0.
    me=0.
    ei=0.
    errs=[]
    xk_keep=None
    for k in range(100):
#
        xk=np.array([0 for i in range(7*n)])
        xk=2.*np.random.rand(7*n)/1.-1./1.
#
#       xk=np.loadtxt('xk_kep.log')
#
#       some prep
#
        xk0=xk[7*pi[0]:7*pi[0]+7]
        xk1=xk[7*pi[1]:7*pi[1]+7]
        pnt0=pnts[maps[pi[0]]]
        pnt1=pnts[maps[pi[1]]]
#
#       calc distance
#
        t0=time.time()
#
        if sol_flg == 0:
#
#           augmented lagrangian formulation, equality constraints
#
            [dis,xt,pos0,pos1]=dcol_auglag(xk0,xk1,pnt0,pnt1,c_a,c_l,0)
#
        elif sol_flg == 1:
#
#           augmented lagrangian formulation, inequality constraints
#
            [dis,xt,pos0,pos1]=dcol_auglag(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        elif sol_flg == 2:
#
#           QP cvxopt
#
            [dis,xt,pos0,pos1]=dcol_qplcvx(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        elif sol_flg == 3:
#
#           QP osqp
#
            [dis,xt,pos0,pos1]=dcol_qplosq(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
#           QP cplex
#
        elif sol_flg == 4:
#
            [dis,xt,pos0,pos1]=dcol_qplcpx(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        elif sol_flg == 5:
#
#           least squares transform cvxopt
#
            [dis,xt,pos0,pos1]=dcol_lstcvx(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        elif sol_flg == 6:
#
#           least squares transform osqp
#
            [dis,xt,pos0,pos1]=dcol_lstosq(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        elif sol_flg == 7:
#
#           least squares transform cplex
#
            [dis,xt,pos0,pos1]=dcol_lstcpx(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        elif sol_flg == 8:
#
#           least squares transform dual
#
            [dis,xt,pos0,pos1]=dcol_lstdua(xk0,xk1,pnt0,pnt1,c_a,c_l,1)
#
        t1=time.time()
#
#   output
#
        nut=len(pnt0)+len(pnt1)
        nuts=[0,len(pnt0),nut]
#
        tees=[]
        for i in range(n):
            tees.append(xt[nuts[i]:nuts[i+1]])
#
        app=appdata(xk,n,nums,maps,vtps,tees,c_l,c_a,c_r,0,0,1)
        woutfle(out,app.GetOutput(),'objec',k)
#
        points=vtk.vtkPoints()
        points.InsertNextPoint(pos0)
        points.InsertNextPoint(pos1)
#
        line=vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0,0)
        line.GetPointIds().SetId(1,1)
        cell=vtk.vtkCellArray()
        cell.InsertNextCell(line)
        ply=vtk.vtkPolyData()
        ply.SetPoints(points)
        ply.SetLines(cell)
        woutfle(out,ply,'mdc',k)
#
        vtps_1=[]
        for i in range(n):
#
            tfm=tfmx(xk,i,c_l,c_a,c_r,None,99,0)
            vtp=tran(vtps[maps[i]],tfm) 
            vtps_1.append(vtp)
#
        [dis_ref,p0_ref,p1_ref]=dist(vtps_1[0],vtps_1[1])
#
        points=vtk.vtkPoints()
        points.InsertNextPoint(p0_ref)
        points.InsertNextPoint(p1_ref)
        line=vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0,0)
        line.GetPointIds().SetId(1,1)
        cell=vtk.vtkCellArray()
        cell.InsertNextCell(line)
        ply=vtk.vtkPolyData()
        ply.SetPoints(points)
        ply.SetLines(cell)
        woutfle(out,ply,'mdr',k)
#
        g1=np.sum(xt[nuts[0]:nuts[1]])-1
        g2=np.sum(xt[nuts[1]:nuts[2]])-1
#
        err=abs(dis - dis_ref)
        rer=err
        if dis_ref > 1e-6:
            rer=rer/dis_ref
            # if no violation and distance is less then the point to surface distance 
            # then it is not a reference value
            if rer > me and dis_ref < dis and max(max(g1,g2),0.) > 1e-12: 
                ei=k
                ae=err
                xk_keep=xk.copy()
                me=max(me,rer)
        tt=tt+t1-t0
        log.info('%6d%16.7e%16.7e%16.7e%16.7e%16.7e'%(k,dis,dis_ref,rer,max(max(g1,g2),0.),t1-t0))
#
    log.info('Maximum Rel. error (%d): %14.7e'%(ei,me))
    log.info('Absolute distance error: %14.7e'%(ae))
    log.info('Total time (s): %14.7e'%tt)
#
#   errs=np.array(errs)
#   mask = errs > 1e-12
#   errs_clip=errs[mask]
#
#   log.info('mean %14.7e and std dev %14.7e'%((np.mean(errs_clip)),(np.std(errs_clip))))
#
#   log.info('Distance: %14.7e'%dis_ref)
#   log.info('Violation: %14.7e'%max(max(g1,g2),0.))
#
#   log.info('Rel. Error if non-zero: %14.7e'%(err))
#   log.info('Time (s): %14.7e'%(t1-t0))
#
#
