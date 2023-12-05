#
import vtk
import numpy as np
#
# gorilla routine for checking minimium distances
#
def dist(vtp0,vtp1):
#
    pnt0=vtp0.GetPoints()
    pnt1=vtp1.GetPoints()
#
    flt=vtk.vtkCellLocator()
    flt.SetDataSet(vtp1)
    flt.BuildLocator()
    flt.Update()
    min_dis=1e8
    min_p0=None
    min_p1=None
    for pid in range(pnt0.GetNumberOfPoints()):
        p=pnt0.GetPoint(pid)
        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        flt.FindClosestPoint(p,c,cellId,subId,d)
        flt.Update()
        d=np.linalg.norm(np.array(p)-np.array(c))
        if d < min_dis:
            min_p1=c#.copy()
            min_p0=p#.copy()
        min_dis=min(min_dis,d)
    flt=vtk.vtkCellLocator()
    flt.SetDataSet(vtp0)
    flt.BuildLocator()
    flt.Update()
    min_p=None
    for pid in range(pnt1.GetNumberOfPoints()):
        p=pnt1.GetPoint(pid)
        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        flt.FindClosestPoint(p,c,cellId,subId,d)
        flt.Update()
        d=np.linalg.norm(np.array(p)-np.array(c))
        if d < min_dis:
            min_p1=c#.copy()
            min_p0=p#.copy()
        min_dis=min(min_dis,d)
#
#   check if anything is inside
#
    c0=0
    enc0=vtk.vtkSelectEnclosedPoints()
    enc0.CheckSurfaceOff()
    enc0.Initialize(vtp0)
    for pid in range(pnt1.GetNumberOfPoints()):
        p=pnt1.GetPoint(pid)
        c0=enc0.IsInsideSurface(p)
        if c0:
            break
    if c0==0:
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(vtp1)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
        p=flt.GetCenter()
        c0=enc0.IsInsideSurface(p)
    if c0:
        min_dis=0.
        min_p0=np.array(min_p0)*0.
        min_p1=np.array(min_p1)*0.
#
    c1=0
    enc1=vtk.vtkSelectEnclosedPoints()
    enc1.CheckSurfaceOff()
    enc1.Initialize(vtp1)
    for pid in range(pnt0.GetNumberOfPoints()):
        p=pnt0.GetPoint(pid)
        c1=enc1.IsInsideSurface(p)
        if c1:
            break
    if c1==0:
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(vtp0)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
        p=flt.GetCenter()
        c1=enc1.IsInsideSurface(p)
    if c1:
        min_dis=0.
        min_p0=np.array(min_p0)*0.
        min_p1=np.array(min_p1)*0.
#
    if c0 == 0 and c1 == 0: # do collision check
#
        tfm=vtk.vtkTransform()
        tfm.PostMultiply()
        tfm.Translate(0., 0., 0.)
        tfm.Update()
        col=vtk.vtkCollisionDetectionFilter()
        col.SetCollisionModeToAllContacts()
        col.SetGenerateScalars(0)
        col.SetInputData(0,vtp0)
        col.SetTransform(0,tfm)
        col.SetInputData(1,vtp1)
        col.SetTransform(1,tfm)
        col.Update()
        c=col.GetNumberOfContacts()
#
        if c:
            min_dis=0.
            min_p0=np.array(min_p0)*0.
            min_p1=np.array(min_p1)*0.
#
    return [min_dis,min_p0,min_p1]
#
