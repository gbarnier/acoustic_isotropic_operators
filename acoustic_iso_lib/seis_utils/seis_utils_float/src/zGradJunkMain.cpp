#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include "spatialDeriv.h"
#include <vector>
#include <string>

using namespace SEP;

int main(int argc, char **argv) {

	/*************************** Main IO bullshit******************************/
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	/* Model and data declaration */
	std::shared_ptr<float3DReg> data;

	/****************************** Model**************************************/
	std::shared_ptr<SEP::genericRegFile> modelFile = io->getRegFile("model",usageIn);
	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();
	std::shared_ptr<SEP::float3DReg> model(new SEP::float3DReg(modelHyper));
	modelFile->readFloatStream(model);

	/******************************* Data *************************************/
	std::shared_ptr<SEP::genericRegFile> dataFile = io->getRegFile("data",usageOut);
	dataFile->setHyper(modelHyper);
	dataFile->writeDescription();
	data = model->clone();
	data->set(0.0);

	/****************************** Dimensions ********************************/
	int nz = model->getHyper()->getAxis(1).n;
	float oz = model->getHyper()->getAxis(1).o;
	float dz = model->getHyper()->getAxis(1).d;
	int nx = model->getHyper()->getAxis(2).n;
	float ox = model->getHyper()->getAxis(2).o;
	float dx = model->getHyper()->getAxis(2).d;
	int nExt = model->getHyper()->getAxis(3).n;
	float oExt = model->getHyper()->getAxis(3).o;
	float dExt = model->getHyper()->getAxis(3).d;
	float dzInv=1.0/(2.0*dz);
	float dzInv1=3.0/(4.0*dz);
	float dzInv2=-3.0/(20.0*dz);
	float dzInv3=1.0/(60.0*dz);
	int _fat = par->getInt("fat");

	#pragma omp parallel for collapse(3)
	for (int iExt=0; iExt<nExt; iExt++){
		for (int ix=_fat; ix<nx-_fat; ix++){
        	for (int iz=_fat; iz<nz-_fat; iz++){
            	(*data->_mat)[iExt][ix][iz] += ((*model->_mat)[iExt][ix][iz+2]-(*model->_mat)[iExt][ix][iz])*dzInv1 +
											   ((*model->_mat)[iExt][ix][iz+3]-(*model->_mat)[iExt][ix][iz-1])*dzInv2 +
											   ((*model->_mat)[iExt][ix][iz+4]-(*model->_mat)[iExt][ix][iz-2])*dzInv3;
			}
		}
	}

    // #pragma omp parallel for collapse(3)
	// for (int iExt=0; iExt<nExt; iExt++){
	// 	for (int ix=_fat; ix<nx-_fat; ix++){
    //     	for (int iz=_fat; iz<nz-_fat; iz++){
    //         	(*data->_mat)[iExt][ix][iz] += ((*model->_mat)[iExt][ix][iz+1]-(*model->_mat)[iExt][ix][iz-1])*dzInv;
	// 		}
	// 	}
	// }

	// #pragma omp parallel for
	// for (int iExt=0; iExt<nExt; iExt++){
	// 	for (int ix=_fat; ix<nx-_fat; ix++){
    // 		(*data->_mat)[iExt][ix][nz-_fat-1] += -1.0*(*model->_mat)[iExt][ix][nz-_fat-1]*dzInv;
    // 	}
	// }

	dataFile->writeFloatStream(data);

	return 0;

}
