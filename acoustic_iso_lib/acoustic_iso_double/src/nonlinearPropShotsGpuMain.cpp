#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double1DReg.h"
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include "deviceGpu.h"
#include "fdParam.h"
#include "nonlinearPropShotsGpu.h"
#include <vector>

using namespace SEP;

int main(int argc, char **argv) {

	/************************************** Main IO *************************************/
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	/* General parameters */
	int adj = par->getInt("adj", 0);
	int saveWavefield = par->getInt("saveWavefield", 0);
	int dotProd = par->getInt("dotProd", 0);
	int dipole = par->getInt("dipole", 0);
	int zDipoleShift = par->getInt("zDipoleShift", 1);
	int xDipoleShift = par->getInt("xDipoleShift", 0);
	int nShot = par->getInt("nShot");
	axis shotAxis = axis(nShot, 1.0, 1.0);

	if (adj == 0 && dotProd == 0 ){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "----------------------- Running nonlinear forward -----------------" << std::endl;
		std::cout << "----------------------- Single precision c++ code -----------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	if (adj == 1 && dotProd == 0 ){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "----------------------- Running nonlinear adjoint -----------------" << std::endl;
		std::cout << "----------------------- Single precision c++ code -----------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}
	if (dotProd == 1){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------------ Running dot product test -----------------" << std::endl;
		std::cout << "------------------------ Single precision c++ code ----------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	/* Model and data declaration */
	std::shared_ptr<double3DReg> model1Float, data1Float;
	std::shared_ptr<double3DReg> wavefield1Float;
	std::shared_ptr <genericRegFile> model1File, data1File, wavefield1File, dampFile;

	/* Read time parameters */
	int nts = par->getInt("nts");
	double dts = par->getFloat("dts", 0.0);
	int sub = par->getInt("sub");
	axis timeAxisCoarse = axis(nts, 0.0, dts);
	int ntw = (nts - 1) * sub + 1;
	double dtw = dts / double(sub);
	axis timeAxisFine = axis(ntw, 0.0, dtw);

	/* Read padding parameters */
	int zPadMinus = par->getInt("zPadMinus");
	int zPadPlus = par->getInt("zPadPlus");
	int xPadMinus = par->getInt("xPadMinus");
	int xPadPlus = par->getInt("xPadPlus");
	int fat = par->getInt("fat");

	/************************************** Velocity model ******************************/
	/* Read velocity (includes the padding + FAT) */
	std::shared_ptr<SEP::genericRegFile> velFile = io->getRegFile("vel",usageIn);
	std::shared_ptr<SEP::hypercube> velHyper = velFile->getHyper();
	std::shared_ptr<SEP::double2DReg> velFloat(new SEP::double2DReg(velHyper));
	velFile->readFloatStream(velFloat);
	int nz = velFloat->getHyper()->getAxis(1).n;
	double oz = velFloat->getHyper()->getAxis(1).o;
	double dz = velFloat->getHyper()->getAxis(1).d;
	int nx = velFloat->getHyper()->getAxis(2).n;
	double ox = velFloat->getHyper()->getAxis(2).o;
	double dx = velFloat->getHyper()->getAxis(2).d;

	/********************************* Create sources vector ****************************/
	/* Create source device vector */
	int nzSource = 1;
	int ozSource = par->getInt("zSource") - 1 + zPadMinus + fat;
	int dzSource = 1;
	int nxSource = 1;
	int oxSource = par->getInt("xSource") - 1 + xPadMinus + fat;
	int dxSource = 1;
	int spacingShots = par->getInt("spacingShots", spacingShots);
	axis sourceAxis(nShot, ox+oxSource*dx, spacingShots*dx);
	std::vector<std::shared_ptr<deviceGpu>> sourcesVector;
	for (int iShot; iShot<nShot; iShot++){
		std::shared_ptr<deviceGpu> sourceDevice(new deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velFloat, nts, dipole, zDipoleShift, xDipoleShift));
		sourcesVector.push_back(sourceDevice);
		oxSource = oxSource + spacingShots;
	}

	/********************************* Create receivers vector **************************/
	int nzReceiver = 1;
	int ozReceiver = par->getInt("depthReceiver") - 1 + zPadMinus + fat;
	int dzReceiver = 1;
	int nxReceiver = par->getInt("nReceiver");
	int oxReceiver = par->getInt("oReceiver") - 1 + xPadMinus + fat;
	int dxReceiver = par->getInt("dReceiver");
	axis receiverAxis(nxReceiver, ox+oxReceiver*dx, dxReceiver*dx);
	std::vector<std::shared_ptr<deviceGpu>> receiversVector;
	int nRecGeom = 1; // Constant receivers' geometry
	for (int iRec; iRec<nRecGeom; iRec++){
		std::shared_ptr<deviceGpu> recDevice(new deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velFloat, nts, dipole, zDipoleShift, xDipoleShift));
		receiversVector.push_back(recDevice);
	}

	/*********************************** Allocation *************************************/
	/* Forward propagation */
	if (adj == 0) {

		/* Allocate and read model */
		// Provide one wavelet for all shots
		model1File = io->getRegFile(std::string("model"),usageIn);
		std::shared_ptr <hypercube> model1Hyper = model1File->getHyper();
		if (model1Hyper->getNdim() == 1){
			axis a(1);
			model1Hyper->addAxis(a);
			model1Hyper->addAxis(a);
		}
		model1Float = std::make_shared<double3DReg>(model1Hyper);
		model1File->readFloatStream(model1Float);

		/* Data double allocation */
		std::shared_ptr<hypercube> data1Hyper(new hypercube(model1Hyper->getAxis(1), receiverAxis, sourceAxis));
		data1Float = std::make_shared<double3DReg>(data1Hyper);

		/* Files shits */
		data1File = io->getRegFile(std::string("data"), usageOut);
		data1File->setHyper(data1Hyper);
		data1File->writeDescription();
	}

	if (adj == 1) {

		/* Allocate and read data */
		data1File = io->getRegFile(std::string("data"),usageIn);

		std::shared_ptr <hypercube> data1Hyper = data1File->getHyper();
		if (data1Hyper->getNdim() == 2) {
			axis a(1);
			data1Hyper->addAxis(a);
		}

		data1Float = std::make_shared<double3DReg>(data1Hyper);
		data1File->readFloatStream(data1Float);

		/* Allocate model */
		axis a(1); // Dummy axis
		std::shared_ptr <hypercube> model1Hyper(new hypercube(timeAxisCoarse, a, a));
		model1Float = std::make_shared<double3DReg>(model1Hyper);

		/* Files shits */
		model1File = io->getRegFile(std::string("model"),usageOut);
		model1File->setHyper(model1Hyper);
		model1File->writeDescription();
	}

	if (saveWavefield == 1){
		// The wavefield(s) allocation is done inside the nonlinearPropShotsGpu object -> no need to allocate outside
		std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(velFloat->getHyper()->getAxis(1), velFloat->getHyper()->getAxis(2), timeAxisCoarse));
		wavefield1File = io->getRegFile(std::string("wavefield"), usageOut);
		wavefield1File->setHyper(wavefield1Hyper);
		wavefield1File->writeDescription();
	}

	/************************************************************************************/
	/******************************** SIMULATIONS ***************************************/
	/************************************************************************************/

	/* Create nonlinear propagation object */
	std::shared_ptr<nonlinearPropShotsGpu> object1(new nonlinearPropShotsGpu(velFloat, par, sourcesVector, receiversVector));

	/********************************** FORWARD *****************************************/
	if (adj == 0 && dotProd == 0) {

		/* Apply forward */
		if (saveWavefield == 1){
			object1->forwardWavefield(false, model1Float, data1Float);
		} else {
			object1->forward(false, model1Float, data1Float);
		}
		/* Output data */
		data1File->writeFloatStream(data1Float);

		/* Wavefield */
		if (saveWavefield == 1){
			wavefield1Float = object1->getWavefield();
			wavefield1File->writeFloatStream(wavefield1Float);
		}
	}

	/********************************** ADJOINT *****************************************/
	if (adj == 1 && dotProd == 0){

		/* Apply adjoint */
		if (saveWavefield == 1){
			object1->adjointWavefield(false, model1Float, data1Float);
		} else {
			object1->adjoint(false, model1Float, data1Float);
		}
		model1File->writeFloatStream(model1Float);

		/* Wavefield */
		if (saveWavefield == 1){
			wavefield1Float = object1->getWavefield();
			wavefield1File->writeFloatStream(wavefield1Float);
		}
	}

	/****************************** DOT PRODUCT TEST ************************************/
	if (dotProd == 1){
		object1->setDomainRange(model1Float, data1Float);
		bool dotprod;
		dotprod = object1->dotTest(true);
	}

	std::cout << " " << std::endl;
	std::cout << "-------------------------------------------------------------------" << std::endl;
	std::cout << "------------------------------ ALL DONE ---------------------------" << std::endl;
	std::cout << "-------------------------------------------------------------------" << std::endl;
	std::cout << " " << std::endl;

	return 0;

}
