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
#include "deviceGpu.h"
#include "BornExtShotsGpu.h"
#include <vector>
#include <string>

using namespace SEP;

int main(int argc, char **argv) {

	/************************************** Main IO *************************************/
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();
	int adj = par->getInt("adj", 0);
	int saveWavefield = par->getInt("saveWavefield", 0);
	int dotProd = par->getInt("dotProd", 0);
	int dipole = par->getInt("dipole", 0);
	int zDipoleShift = par->getInt("zDipoleShift", 1);
	int xDipoleShift = par->getInt("xDipoleShift", 0);
	int nShot = par->getInt("nShot");
	axis shotAxis = axis(nShot, 1.0, 1.0);

	if (adj == 0 && dotProd == 0){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------ Running extended Born forward ------------------" << std::endl;
		std::cout << "--------------------- Single precision c++ code -------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	if (adj == 1 && dotProd == 0){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------ Running extended Born adjoint ------------------" << std::endl;
		std::cout << "-------------------- Single precision c++ code --------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}
	if (dotProd == 1){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "--------------------- Running dot product test --------------------" << std::endl;
		std::cout << "--------------------- Single precision c++ code -------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}


	/* Model and data declaration */
	std::shared_ptr<float2DReg> sourcesSignalTempFloat;
	std::shared_ptr<float3DReg> model1Float, model2Float;
	std::shared_ptr<float3DReg> data1Float, data2Float;
	std::shared_ptr <genericRegFile> sourcesFile, model1File, model2File, data1File, data2File, sourcesSignalsFile;
	std::shared_ptr <hypercube> model1Hyper, data1Hyper;

	/* Read time parameters */
	int nts = par->getInt("nts");
	float dts = par->getFloat("dts", 0.0);
	int sub = par->getInt("sub");
	axis timeAxisCoarse = axis(nts, 0.0, dts);
	int ntw = (nts - 1) * sub + 1;
	float dtw = dts / float(sub);
	axis timeAxisFine = axis(ntw, 0.0, dtw);

	/* Read extension parameters */
	axis extAxis;
	std::string extension = par->getString("extension", "none");
	int nExt = par->getInt("nExt", 1);
	if (nExt%2 == 0){std::cout << "Length of extension axis must be an uneven number" << std::endl; throw std::runtime_error("");}
	int hExt = (nExt-1)/2;
	if (extension == "time"){
		float dExt = par->getFloat("dExt", dts);
		float oExt = par->getFloat("oExt", -dExt*hExt);
		extAxis = axis(nExt, oExt, dExt);
	} else {
		float dExt = par->getFloat("dExt", par->getFloat("dx", -1.0));
		float oExt = par->getFloat("oExt", -dExt*hExt);
		extAxis = axis(nExt, oExt, dExt);
	}

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
	std::shared_ptr<SEP::float2DReg> velFloat(new SEP::float2DReg(velHyper));
	velFile->readFloatStream(velFloat);
	int nz = velFloat->getHyper()->getAxis(1).n;
	float oz = velFloat->getHyper()->getAxis(1).o;
	float dz = velFloat->getHyper()->getAxis(1).d;
	int nx = velFloat->getHyper()->getAxis(2).n;
	float ox = velFloat->getHyper()->getAxis(2).o;
	float dx = velFloat->getHyper()->getAxis(2).d;

	/********************************* Create sources vector ****************************/
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

	/******************************* Create sources signals vector **********************/
	// Read sources signals file - we use one identical wavelet for all shots
	sourcesSignalsFile = io->getRegFile(std::string("sources"),usageIn);
	std::shared_ptr <hypercube> sourcesSignalsHyper = sourcesSignalsFile->getHyper();
	std::vector<std::shared_ptr<float2DReg>> sourcesSignalsVector;
	if (sourcesSignalsHyper->getNdim() == 1){
		axis a(1);
		sourcesSignalsHyper->addAxis(a);
	}
	sourcesSignalTempFloat = std::make_shared<float2DReg>(sourcesSignalsHyper);
	sourcesSignalsFile->readFloatStream(sourcesSignalTempFloat);
	sourcesSignalsVector.push_back(sourcesSignalTempFloat);

	/*********************************** Allocation *************************************/

	/* Forward propagation */
	if (adj == 0) {

		/* Allocate and read model */
		model1File = io->getRegFile(std::string("model"),usageIn);
		std::shared_ptr <hypercube> model1Hyper = model1File->getHyper();
		if (model1Hyper->getNdim() == 2){
			axis a(1);
			model1Hyper->addAxis(a);
		}

		model1Float = std::make_shared<float3DReg>(model1Hyper);
		model1File->readFloatStream(model1Float);

		/* Data allocation */
		std::shared_ptr<hypercube> data1Hyper(new hypercube(sourcesSignalsHyper->getAxis(1), receiverAxis, sourceAxis));
		data1Float = std::make_shared<float3DReg>(data1Hyper);

		/* Files shits */
		data1File = io->getRegFile(std::string("data"), usageOut);
		data1File->setHyper(data1Hyper);
		data1File->writeDescription();

	}

	/* Adjoint propagation */
	if (adj == 1) {

		/* Allocate and read data */
		data1File = io->getRegFile(std::string("data"),usageIn);
		std::shared_ptr <hypercube> data1Hyper = data1File->getHyper();

		// Case where only one receiver and 1 shot
		if (data1Hyper->getNdim() == 1){
			axis a(1);
			data1Hyper->addAxis(a);
			data1Hyper->addAxis(a);
		}

		// Case where only multiple receivers and 1 shot
		if (data1Hyper->getNdim() == 2){
			axis a(1);
			data1Hyper->addAxis(a);
		}

		data1Float = std::make_shared<float3DReg>(data1Hyper);
		data1File->readFloatStream(data1Float);

		/* Allocate and read model */
		std::shared_ptr<hypercube> model1Hyper = velHyper;
		model1Hyper->addAxis(extAxis);
		model1Float = std::make_shared<float3DReg>(model1Hyper);

		/* Stupid files shits */
		model1File = io->getRegFile(std::string("model"),usageOut);
		model1File->setHyper(model1Hyper);
		model1File->writeDescription();

	}

	/* Wavefields */
	std::shared_ptr<float3DReg> srcWavefield1Float, secWavefield1Float;
	std::shared_ptr<genericRegFile> srcWavefield1File = io->getRegFile(std::string("srcWavefield"), usageOut);
	std::shared_ptr<genericRegFile> secWavefield1File = io->getRegFile(std::string("secWavefield"), usageOut);

	/************************************************************************************/
	/******************************** SIMULATIONS ***************************************/
	/************************************************************************************/

	/* Create Born extended object */
	std::shared_ptr<BornExtShotsGpu> object1(new BornExtShotsGpu(velFloat, par, sourcesVector, sourcesSignalsVector, receiversVector));

	/********************************** FORWARD *****************************************/
	if (adj == 0 && dotProd ==0) {
		if (saveWavefield == 1){
			object1->forwardWavefield(false, model1Float, data1Float);
		} else {
			object1->forward(false, model1Float, data1Float);
		}
		data1File->writeFloatStream(data1Float);

		/* Wavefield */
		if (saveWavefield == 1){

			// Wavefield hypercube
			std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(velFloat->getHyper()->getAxis(1), velFloat->getHyper()->getAxis(2), timeAxisCoarse));

			// Source wavefield
			srcWavefield1Float = object1->getSrcWavefield();
			srcWavefield1File->setHyper(wavefield1Hyper);
			srcWavefield1File->writeDescription();
			srcWavefield1File->writeFloatStream(srcWavefield1Float);

			// Scattered wavefield
			secWavefield1Float = object1->getSecWavefield();
			secWavefield1File->setHyper(wavefield1Hyper);
			secWavefield1File->writeDescription();
			secWavefield1File->writeFloatStream(secWavefield1Float);
		}
	}

	/********************************** ADJOINT *****************************************/
	if (adj == 1 && dotProd ==0) {

		if (saveWavefield == 1){
			object1->adjointWavefield(false, model1Float, data1Float);
		} else {
			object1->adjoint(false, model1Float, data1Float);
		}
		model1File->writeFloatStream(model1Float);

		/* Wavefield */
		if (saveWavefield == 1){

			// Wavefield hypercube
			std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(velFloat->getHyper()->getAxis(1), velFloat->getHyper()->getAxis(2), timeAxisCoarse));

			// Source wavefield
			srcWavefield1File->setHyper(wavefield1Hyper);
			srcWavefield1File->writeDescription();
			srcWavefield1File->writeFloatStream(srcWavefield1Float);
			srcWavefield1Float = object1->getSrcWavefield();

			// Receiver wavefield
			secWavefield1Float = object1->getSecWavefield();
			secWavefield1File->setHyper(wavefield1Hyper);
			secWavefield1File->writeDescription();
			secWavefield1File->writeFloatStream(secWavefield1Float);

		}
	}

	/* Dot product test */
	if (dotProd == 1) {
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
