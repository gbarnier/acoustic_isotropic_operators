#include <float3DReg.h>
#include "dsoGpu.h"
#include <vector>
#include <omp.h>

using namespace SEP;

dsoGpu::dsoGpu(int nz, int nx, int nExt, int fat, float zeroShift){
    _nz = nz;
    _nx = nx;
    _nExt = nExt;
	if (_nExt % 2 == 0) {std::cout << "**** ERROR: Length of extended axis must be an uneven number ****" << std::endl; assert(1==2); }
    _hExt = (_nExt-1) / 2;
    _zeroShift = zeroShift;
    _fat = fat;
}

void dsoGpu::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {

	if (!add) data->scale(0.0);

    for (int iExt=0; iExt<_nExt; iExt++){
        float weight = 1.0*(iExt-_hExt) + _zeroShift; // Compute weight for this extended point
        #pragma omp parallel for collapse(2)
        for (int ix=_fat; ix<_nx-_fat; ix++){
            for (int iz=_fat; iz<_nz-_fat; iz++){
                (*data->_mat)[iExt][ix][iz] += weight * (*model->_mat)[iExt][ix][iz];
            }
        }
    }
}

void dsoGpu::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const {

	if (!add) model->scale(0.0);

    for (int iExt=0; iExt<_nExt; iExt++){
        float weight = 1.0*(iExt-_hExt) + _zeroShift; // Compute weight for this extended point
        #pragma omp parallel for collapse(2)
        for (int ix=_fat; ix<_nx-_fat; ix++){
            for (int iz=_fat; iz<_nz-_fat; iz++){
                (*model->_mat)[iExt][ix][iz] += weight * (*data->_mat)[iExt][ix][iz];
            }
        }
    }
}
