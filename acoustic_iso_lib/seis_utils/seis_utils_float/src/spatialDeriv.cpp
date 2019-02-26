#include <float2DReg.h>
#include "spatialDeriv.h"
#include <vector>
#include <omp.h>

/////////////////////////// Gradient in z-direction ////////////////////////////
zGrad::zGrad(int fat){
    _fat = fat;
}

void zGrad::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

    if (!add) data->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
    float dzInv=model->getHyper()->getAxis(1).d;
    dzInv=1.0/dzInv;

    #pragma omp parallel for collapse(2)
    for (int ix=_fat; ix<nx-_fat; ix++){
        for (int iz=_fat; iz<nz-_fat-1; iz++){
            (*data->_mat)[ix][iz] += ((*model->_mat)[ix][iz+1]-(*model->_mat)[ix][iz])*dzInv;
        }
    }
    #pragma omp parallel for
    for (int ix=_fat; ix<nx-_fat; ix++){
        (*data->_mat)[ix][nz-_fat-1] += -1.0*(*model->_mat)[ix][nz-_fat-1]*dzInv;
    }
}
void zGrad::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
    float dzInv=model->getHyper()->getAxis(1).d;
    dzInv=1.0/dzInv;

    #pragma omp parallel for
    for (int ix=_fat; ix<nx-_fat; ix++){
        (*model->_mat)[ix][_fat] += -1.0*(*data->_mat)[ix][_fat]*dzInv;
    }
    #pragma omp parallel for collapse(2)
    for (int ix=_fat; ix<nx-_fat; ix++){
        for (int iz=_fat+1; iz<nz-_fat; iz++){
            (*model->_mat)[ix][iz] += ((*data->_mat)[ix][iz-1]-(*data->_mat)[ix][iz])*dzInv;
        }
    }
}

/////////////////////////// Gradient in x-direction ////////////////////////////
xGrad::xGrad(int fat){
    _fat = fat;
}

void xGrad::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

    if (!add) data->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
    float dxInv=model->getHyper()->getAxis(2).d;
    dxInv=1.0/dxInv;

    #pragma omp parallel for collapse(2)
    for (int ix=_fat; ix<nx-_fat-1; ix++){
        for (int iz=_fat; iz<nz-_fat; iz++){
            (*data->_mat)[ix][iz] += ((*model->_mat)[ix+1][iz]-(*model->_mat)[ix][iz])*dxInv;
        }
    }
    #pragma omp parallel for
    for (int iz=_fat; iz<nz-_fat; iz++){
        (*data->_mat)[nx-_fat-1][iz] += -1.0*(*model->_mat)[nx-_fat-1][iz]*dxInv;
    }
}

void xGrad::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
    float dxInv=model->getHyper()->getAxis(2).d;
    dxInv=1.0/dxInv;

    #pragma omp parallel for
    for (int iz=_fat; iz<nz-_fat; iz++){
        (*model->_mat)[_fat][iz] += -1.0*(*data->_mat)[_fat][iz]*dxInv;
    }
    #pragma omp parallel for collapse(2)
    for (int ix=_fat+1; ix<nx-_fat; ix++){
        for (int iz=_fat; iz<nz-_fat; iz++){
            (*model->_mat)[ix][iz] += ((*data->_mat)[ix-1][iz]-(*data->_mat)[ix][iz])*dxInv;
        }
    }
}

// /////////////////////// Sum of gradient in both directions /////////////////////
zxGrad::zxGrad(int fat){
    _fat = fat;
}

void zxGrad::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

    if (!add) data->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
    float dzInv=model->getHyper()->getAxis(1).d;
    float dxInv=model->getHyper()->getAxis(2).d;
    dzInv=1.0/dzInv;
    dxInv=1.0/dxInv;

    // z-direction
    #pragma omp parallel for collapse(2)
    for (int ix=_fat; ix<nx-_fat; ix++){
        for (int iz=_fat; iz<nz-_fat-1; iz++){
            (*data->_mat)[ix][iz] += ((*model->_mat)[ix][iz+1]-(*model->_mat)[ix][iz])*dzInv;
        }
    }
    #pragma omp parallel for
    for (int ix=_fat; ix<nx-_fat; ix++){
        (*data->_mat)[ix][nz-_fat-1] += -1.0*(*model->_mat)[ix][nz-_fat-1]*dzInv;
    }

    // x-direction
    #pragma omp parallel for collapse(2)
    for (int ix=_fat; ix<nx-_fat-1; ix++){
        for (int iz=_fat; iz<nz-_fat; iz++){
            (*data->_mat)[ix][iz] += ((*model->_mat)[ix+1][iz]-(*model->_mat)[ix][iz])*dxInv;
        }
    }
    #pragma omp parallel for
    for (int iz=_fat; iz<nz-_fat; iz++){
        (*data->_mat)[nx-_fat-1][iz] += -1.0*(*model->_mat)[nx-_fat-1][iz]*dxInv;
    }
}

void zxGrad::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
    float dzInv=model->getHyper()->getAxis(1).d;
    float dxInv=model->getHyper()->getAxis(2).d;
    dzInv=1.0/dzInv;
    dxInv=1.0/dxInv;

    // z-direction
    #pragma omp parallel for
    for (int ix=_fat; ix<nx-_fat; ix++){
        (*model->_mat)[ix][_fat] += -1.0*(*data->_mat)[ix][_fat]*dzInv;
    }
    #pragma omp parallel for collapse(2)
    for (int ix=_fat; ix<nx-_fat; ix++){
        for (int iz=_fat+1; iz<nz-_fat; iz++){
            (*model->_mat)[ix][iz] += ((*data->_mat)[ix][iz-1]-(*data->_mat)[ix][iz])*dzInv;
        }
    }
    // x-direction
    #pragma omp parallel for
    for (int iz=_fat; iz<nz-_fat; iz++){
        (*model->_mat)[_fat][iz] += -1.0*(*data->_mat)[_fat][iz]*dxInv;
    }
    #pragma omp parallel for collapse(2)
    for (int ix=_fat+1; ix<nx-_fat; ix++){
        for (int iz=_fat; iz<nz-_fat; iz++){
            (*model->_mat)[ix][iz] += ((*data->_mat)[ix-1][iz]-(*data->_mat)[ix][iz])*dxInv;
        }
    }
}
