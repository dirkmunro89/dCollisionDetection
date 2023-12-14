#
import numpy as np
#
def rota(x,vec):
    qr=x[0]; qi=x[1]; qj=x[2]; qk=x[3]
    s=1.#np.linalg.norm(x)**-2.
    R = np.array([[1.-2.*s*(qj**2.+qk**2.),2.*s*(qi*qj-qk*qr),2.*s*(qi*qk+qj*qr)],\
        [2.*s*(qi*qj+qk*qr),1.-2.*s*(qi**2.+qk**2.),2.*s*(qj*qk-qi*qr)],\
        [2.*s*(qi*qk-qj*qr), 2.*s*(qj*qk+qi*qr),1.-2.*s*(qi**2.+qj**2.)]])
    return np.dot(vec,R.T)
#
def drota(x,vec):
#
    qr=x[0]; qi=x[1]; qj=x[2]; qk=x[3]
    s=1.#np.linalg.norm(x)**-2.
    dRdr = np.array([[0.,2.*s*(-qk),2.*s*qj],\
        [2.*s*qk,0.,2.*s*(-qi)],\
        [2.*s*(-qj), 2.*s*qi, 0.]])
    dRdi = np.array([[0.,2.*s*qj,2.*s*qk],\
        [2.*s*(qj),-4.*s*qi,-2.*s*qr],\
        [2.*s*(qk), 2.*s*qr,-4.*s*qi]])
    dRdj = np.array([[-4.*s*qj,2.*s*qi,2.*s*qr],\
        [2.*s*qi,0.,2.*s*qk],\
        [-2.*s*qr, 2.*s*qk,-4.*s*qj]])
    dRdk = np.array([[-4.*s*qk,-2.*s*qr,2.*s*qi],\
        [2.*s*qr,-4.*s*qk,2.*s*qj],\
        [2.*s*qi, 2.*s*qj,0.]])
#
    return np.dot(vec,dRdr.T),np.dot(vec,dRdi.T),np.dot(vec,dRdj.T),np.dot(vec,dRdk.T)
#

