#include <iostream>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"

using namespace SEP;

int main(int argc, char **argv) {

	// IO bullshit
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	// Model
	std::shared_ptr <genericRegFile> modelFile = io->getRegFile("model",usageIn);
 	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();
	if (modelHyper->getNdim() == 2){
		axis extAxis(1, 0.0, 1.0);
		modelHyper->addAxis(extAxis);
	}
 	std::shared_ptr<SEP::float3DReg> model(new SEP::float3DReg(modelHyper));
	modelFile->readFloatStream(model);

	// Model parameters
	long long nz = model->getHyper()->getAxis(1).n;
	long long nx = model->getHyper()->getAxis(2).n;
	long long nExt = model->getHyper()->getAxis(3).n;

	// Parfile
	int zPad = par->getInt("zPad");
	int xPad = par->getInt("xPad");
	int fat = par->getInt("fat", 5);
	int blockSize = par->getInt("blockSize", 16);
	int freeSurface = par->getInt("freeSurface", 0);

	// Compute size of zPadPlus
	int zPadPlus, nzNew, nzNewTotal;
	if (freeSurface == 1){
		long long nzTotal = zPad + nz;
		float ratioz = float(nzTotal) / float(blockSize);
		ratioz = ceilf(ratioz);
		long long nbBlockz = ratioz;
		zPadPlus = nbBlockz * blockSize - nz;
		nzNew = zPadPlus + nz;
		nzNewTotal = nzNew + 2*fat;
		zPad = 0;
	} else {
		long long nzTotal = zPad * 2 + nz;
		float ratioz = float(nzTotal) / float(blockSize);
		ratioz = ceilf(ratioz);
		long long nbBlockz = ratioz;
		zPadPlus = nbBlockz * blockSize - nz - zPad;
		nzNew = zPad + zPadPlus + nz;
		nzNewTotal = nzNew + 2*fat;
	}

	// Compute size of xPadPlus
	int xPadPlus;
	long long nxTotal = xPad * 2 + nx;
	float ratiox = float(nxTotal) / float(blockSize);
	ratiox = ceilf(ratiox);
	long long nbBlockx = ratiox;
	xPadPlus = nbBlockx * blockSize - nx - xPad;
	long long nxNew = xPad + xPadPlus + nx;
	long long nxNewTotal = nxNew + 2*fat;

	// Compute parameters
	float dz = modelHyper->getAxis(1).d;
	float oz = modelHyper->getAxis(1).o - (fat + zPad) * dz;
	float dx = modelHyper->getAxis(2).d;
	float ox = modelHyper->getAxis(2).o - (fat + xPad) * dx;

	// Data
	axis zAxis = axis(nzNewTotal, oz, dz);
	axis xAxis = axis(nxNewTotal, ox, dx);
	axis extAxis = 	axis(nExt, model->getHyper()->getAxis(3).o, model->getHyper()->getAxis(3).d);
 	std::shared_ptr<SEP::hypercube> dataHyper(new hypercube(zAxis, xAxis, extAxis));
 	std::shared_ptr<SEP::float3DReg> data(new SEP::float3DReg(dataHyper));
	std::shared_ptr <genericRegFile> dataFile = io->getRegFile("data",usageOut);
	dataFile->setHyper(dataHyper);
	dataFile->writeDescription();
	data->scale(0.0);

	for (int iExt=0; iExt<nExt; iExt++) {

		// Copy central part
		for (long long ix=0; ix<nx; ix++){
			for (long long iz=0; iz<nz; iz++){
				(*data->_mat)[iExt][ix+fat+xPad][iz+fat+zPad] = (*model->_mat)[iExt][ix][iz];
			}
		}

		for (long long ix=0; ix<nx; ix++){
			// Top central part
			for (long long iz=0; iz<zPad; iz++){
				(*data->_mat)[iExt][ix+fat+xPad][iz+fat] = (*model->_mat)[iExt][ix][0];
			}
			// Bottom central part
			for (long long iz=0; iz<zPadPlus; iz++){
			(*data->_mat)[iExt][ix+fat+xPad][iz+fat+zPad+nz] = (*model->_mat)[iExt][ix][nz-1];
			}
		}

		// Left part
		for (long long ix=0; ix<xPad; ix++){
			for (long long iz=0; iz<nzNew; iz++) {
				(*data->_mat)[iExt][ix+fat][iz+fat] = (*data->_mat)[iExt][xPad+fat][iz+fat];
			}
		}

		// Right part
		for (long long ix=0; ix<xPadPlus; ix++){
			for (long long iz=0; iz<nzNew; iz++){
				(*data->_mat)[iExt][ix+fat+nx+xPad][iz+fat] = (*data->_mat)[iExt][fat+xPad+nx-1][iz+fat];
			}
		}
	}

	// Write model
	dataFile->writeFloatStream(data);

	// Display info
	std::cout << " " << std::endl;
	std::cout << "------------------------ Model padding program --------------------" << std::endl;
	std::cout << "Original nz = " << nz << " [samples]" << std::endl;
	std::cout << "Original nx = " << nx << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	if (freeSurface == 1){
		std::cout << "zPadMinus = " << zPad << " [samples] => Model designed with a free surface at the top" << std::endl;
	} else {
		std::cout << "zPadMinus = " << zPad << " [samples]" << std::endl;
	}
	std::cout << "zPadPlus = " << zPadPlus << " [samples]" << std::endl;
	std::cout << "xPadMinus = " << xPad << " [samples]" << std::endl;
	std::cout << "xPadPlus = " << xPadPlus << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "blockSize = " << blockSize << " [samples]" << std::endl;
	std::cout << "FAT = " << fat << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "New nz = " << nzNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "New nx = " << nxNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "-------------------------------------------------------------------" << std::endl;
	std::cout << " " << std::endl;
	return 0;

}
