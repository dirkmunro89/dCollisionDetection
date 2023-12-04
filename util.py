#
import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
#
def tfmx(x,i,c_l,c_a,c_r,tfm,int_flg,rev_flg):
#
    ret_flg=0
    if tfm is None:
        ret_flg=1
        tfm=vtk.vtkTransform()
    tfm.PostMultiply()
#
    c=i
#
    if int_flg == 1:
#
        tmp=x[c*4]*3.#abs(x[c*4])%7 - 3.5
#
        if tmp >= 0-3.5 and tmp < 1-3.5:
            r=R.from_matrix(c_r[0].T).as_rotvec()
        elif tmp >= 1-3.5 and tmp < 2-3.5:
            r=R.from_matrix(c_r[1].T).as_rotvec()
        elif tmp >= 2-3.5 and tmp < 3-3.5:
            r=R.from_matrix(c_r[2].T).as_rotvec()
        elif tmp >= 3-3.5 and tmp < 4-3.5:
            r=R.from_matrix(c_r[3].T).as_rotvec()
        elif tmp >= 4-3.5 and tmp < 5-3.5:
            r=R.from_matrix(c_r[4].T).as_rotvec()
        elif tmp >= 5-3.5 and tmp < 6-3.5:
            r=R.from_matrix(c_r[5].T).as_rotvec()
        elif tmp >= 6-3.5 and tmp < 7-3.5:
            r=R.from_matrix(c_r[6].T).as_rotvec()
        else:
            print(x[i*4],tmp)
            print('error util')
            exit()
#
        tmp=max(np.linalg.norm(r),1e-9)
        if rev_flg:
            tfm.Translate(-c_l[0]*x[c*4+1], -c_l[1]*x[c*4+2], -c_l[2]*x[c*4+3])
#           tfm.Scale(1./c_s,1./c_s,1./c_s)
            tfm.RotateWXYZ(-np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
        else:
            tfm.RotateWXYZ(np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
#           tfm.Scale(c_s,c_s,c_s)
            tfm.Translate(c_l[0]*x[c*4+1], c_l[1]*x[c*4+2], c_l[2]*x[c*4+3])
#
    elif int_flg == 24:
#
        tmp=x[c*4]*12.#abs(x[c*4])%7 - 3.5
#
        for i in range(25):
            if tmp >= i-12.5 and tmp < i+1-12.5:
                r=R.from_matrix(c_r[i].T).as_rotvec()
                break
#
        tmp=max(np.linalg.norm(r),1e-9)
        if rev_flg:
            tfm.Translate(-c_l[0]*x[c*4+1], -c_l[1]*x[c*4+2], -c_l[2]*x[c*4+3])
#           tfm.Scale(1./c_s,1./c_s,1./c_s)
            tfm.RotateWXYZ(-np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
        else:
            tfm.RotateWXYZ(np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
#           tfm.Scale(c_s,c_s,c_s)
            tfm.Translate(c_l[0]*x[c*4+1], c_l[1]*x[c*4+2], c_l[2]*x[c*4+3])
#
    elif int_flg == 2:
#
        tmp=x[c*7]*3.#abs(x[c*4])%7 - 3.5
#
        if tmp >= 0-3.5 and tmp < 1-3.5:
            r=R.from_matrix(c_r[0].T).as_rotvec()
        elif tmp >= 1-3.5 and tmp < 2-3.5:
            r=R.from_matrix(c_r[1].T).as_rotvec()
        elif tmp >= 2-3.5 and tmp < 3-3.5:
            r=R.from_matrix(c_r[2].T).as_rotvec()
        elif tmp >= 3-3.5 and tmp < 4-3.5:
            r=R.from_matrix(c_r[3].T).as_rotvec()
        elif tmp >= 4-3.5 and tmp < 5-3.5:
            r=R.from_matrix(c_r[4].T).as_rotvec()
        elif tmp >= 5-3.5 and tmp < 6-3.5:
            r=R.from_matrix(c_r[5].T).as_rotvec()
        elif tmp >= 6-3.5 and tmp < 7-3.5:
            r=R.from_matrix(c_r[6].T).as_rotvec()
        else:
            print(x[i*4],tmp)
            print('error util')
            exit()
#
        tmp=max(np.linalg.norm(r),1e-9)
        if rev_flg:
            tfm.Translate(-c_l[0]*x[c*7+4], -c_l[1]*x[c*7+5], -c_l[2]*x[c*7+6])
            tfm.Scale(1/(2.+x[c*7+1]),1/(2.+x[c*7+2]),1/(2.+x[c*7+3]))
            tfm.RotateWXYZ(-np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
        else:
            tfm.RotateWXYZ(np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
            tfm.Scale(2.+x[c*7+1],2.+x[c*7+2],2.+x[c*7+3])
            tfm.Translate(c_l[0]*x[c*7+4], c_l[1]*x[c*7+5], c_l[2]*x[c*7+6])
#    
    else:
#
#       checked: it is normalised internally anyway; good to show it here
#
        if rev_flg:
            tfm.Translate(-c_l[0]*x[c*7+4], -c_l[1]*x[c*7+5], -c_l[2]*x[c*7+6])
#           tfm.Scale(1./c_s,1./c_s,1./c_s)
            tmp=x[c*7+1:c*7+4]#/np.linalg.norm(x[c*7+1:c*7+4])
            tfm.RotateWXYZ(np.rad2deg(-c_a*x[c*7]), tmp[0], tmp[1], tmp[2])
        else:
            tmp=x[c*7+1:c*7+4]#/np.linalg.norm(x[c*7+1:c*7+4])
            tfm.RotateWXYZ(np.rad2deg(c_a*x[c*7]), tmp[0], tmp[1], tmp[2])
#           tfm.Scale(c_s,c_s,c_s)
            tfm.Translate(c_l[0]*x[c*7+4], c_l[1]*x[c*7+5], c_l[2]*x[c*7+6])
#
    tfm.Update()
#
    if ret_flg:
        return tfm
#
def appdata(x,n,nums,maps,vtis,tees,c_l,c_a,c_r,int_flg,str_flg,col_flg):
#
    t = np.linspace(-510, 510, n)                                              
    rgbs=np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)
#
    c=0
    app=vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(len(nums)):
#
        if str_flg:
           red=vtk.vtkXMLPolyDataReader()
           red.ReadFromInputStringOn()
           red.SetInputString(stis[i])
           red.Update()
           vtp=red.GetOutput()
        else:
            vtp = vtis[i]
#
        for j in range(nums[i]):
#
            tfm=tfmx(x,c,c_l,c_a,c_r,None,int_flg,0)
#
            tmp=tran(vtp,tfm)
#
            if col_flg:
                color=vtk.vtkUnsignedCharArray() 
                color.SetName("Colors") 
                color.SetNumberOfComponents(3) 
                color.SetNumberOfTuples(tmp.GetNumberOfCells())
                for k in range(tmp.GetNumberOfCells()):
                    color_tup=list(rgbs[c])
                    color.SetTuple(k,color_tup)
                tmp.GetCellData().SetScalars(color)
#
                array=vtk.vtkFloatArray()
                array.SetName("Tees")
                array.SetNumberOfComponents(1)
                array.SetNumberOfTuples(tmp.GetNumberOfPoints())
                for k in range(tmp.GetNumberOfPoints()):
                    array.SetTuple1(k,tees[c][k])
                tmp.GetPointData().SetScalars(array)
#
            app.AddInputData(tmp)
#
            c=c+1
#
    app.Update()
#
    return app
#
def tran(vtp,tfm):
#
    tfm_flt =  vtk.vtkTransformPolyDataFilter()
    tfm_flt.SetInputData(vtp)
    tfm_flt.SetTransform(tfm)
    tfm_flt.Update()
    tmp=tfm_flt.GetOutput()
#
    return tmp
#
def woutfle(out,vtp,fln,k):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    if k < 0:
        writer.SetFileName(out+fln+'_%s.vtp'%(chr(ord('`')+(-k))))
    elif k == 0:
        writer.SetFileName(out+fln+'.vtp')
    else:
        writer.SetFileName(out+fln+'_%d.vtp'%k)
    writer.Update()
#
def woutstr(vtp):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
#   writer.SetCompressorTypeToZLib()
    writer.SetWriteToOutputString(True)
    writer.Update()
    return writer.GetOutputString()
#
