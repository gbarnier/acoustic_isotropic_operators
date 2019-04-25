#include <float3DReg.h>
#include "SymesZGrad.h"
#include <vector>
#include <omp.h>

////////////////// Gradient in z-direction shifted by one sample ///////////////
SymesZGrad::SymesZGrad(int fat){
    _fat = fat;
}

void SymesZGrad::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {

    if (!add) data->scale(0.0);

    int nz=model->getHyper()->getAxis(1).n;
    int nx=model->getHyper()->getAxis(2).n;
	int nExt=model->getHyper()->getAxis(3).n;

    float dz=model->getHyper()->getAxis(1).d;
    float dzInv=1.0/(2.0*dz);
	// float dzInv1=3.0/(4.0*dz);
	// float dzInv2=-3.0/(20.0*dz);
	// float dzInv3=1.0/(60.0*dz);

	// #pragma omp parallel for collapse(3)
	// for (int iExt=0; iExt<nExt; iExt++){
	// 	for (int ix=_fat; ix<nx-_fat; ix++){
    //     	for (int iz=_fat; iz<nz-_fat; iz++){
    //         	(*data->_mat)[iExt][ix][iz] += ((*model->_mat)[iExt][ix][iz+2]-(*model->_mat)[iExt][ix][iz])*dzInv1 +
	// 										   ((*model->_mat)[iExt][ix][iz+3]-(*model->_mat)[iExt][ix][iz-1])*dzInv2 +
	// 										   ((*model->_mat)[iExt][ix][iz+4]-(*model->_mat)[iExt][ix][iz-2])*dzInv3;
	// 		}
	// 	}
	// }

	// #pragma omp parallel for collapse(3)
	// for (int iExt=0; iExt<nExt; iExt++){
	// 	for (int ix=_fat; ix<nx-_fat; ix++){
    //     	for (int iz=_fat; iz<nz-_fat; iz++){
    //         	(*data->_mat)[iExt][ix][iz] += ((*model->_mat)[iExt][ix][iz+1]-(*model->_mat)[iExt][ix][iz-1])*dzInv1 +
	// 										   ((*model->_mat)[iExt][ix][iz+2]-(*model->_mat)[iExt][ix][iz-2])*dzInv2 +
	// 										   ((*model->_mat)[iExt][ix][iz+3]-(*model->_mat)[iExt][ix][iz-3])*dzInv3;
	// 		}
	// 	}
	// }

	for (int iExt=0; iExt<nExt; iExt++){
		for (int ix=_fat; ix<nx-_fat; ix++){
        	for (int iz=_fat; iz<nz-_fat; iz++){
            	(*data->_mat)[iExt][ix][iz] += ((*model->_mat)[iExt][ix][iz+1]-(*model->_mat)[iExt][ix][iz-1])*dzInv;
			}
		}
	}

}

void SymesZGrad::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const {

	if (!add) model->scale(0.0);
	std::cout << "Adjoint not implemented for this operator" << std::endl;
}
