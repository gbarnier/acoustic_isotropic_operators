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
#include "wemvaExtShotsGpu.h"
#include <vector>

using namespace SEP;

int main(int argc, char **argv) {

	/************************************** Main IO *************************************/
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();
	int adj = par->getInt("adj", 0);
	int saveWavefield = par->getInt("saveWavefield", 0);
	int dotProd = par->getInt("dotProd", 0);

	if (adj == 0 && dotProd == 0){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------ Running extended Wemva forward -----------------" << std::endl;
		std::cout << "--------------------- Single precision c++ code -------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	if (adj == 1 && dotProd == 0){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------ Running extended Wemva adjoint -----------------" << std::endl;
		std::cout << "--------------------- Single precision c++ code -------------------" << std::endl;
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
    // Sources signals
    std::shared_ptr<float2DReg> sourcesSignalTempFloat;

    // Wemva data
    std::shared_ptr<float3DReg> receiversSignalsTempFloat;
	std::shared_ptr<float2DReg> receiversSignalsSliceFloat;

    // Model
	std::shared_ptr<float2DReg> model1Float, model2Float;

    // Image
	std::shared_ptr<float3DReg> data1Float, data2Float;

    // Files
    std::shared_ptr <genericRegFile> model1File, model2File, data1File, data2File, sourcesSignalsFile, receiversSignalsFile;
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
	if (nExt%2 == 0){std::cout << "**** ERROR: Length of extended axis must be an uneven number ****" << std::endl; assert(1==2);}
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

	/********************************* Velocity model ***********************************/
	/* Read velocity (includes the padding + FAT) */
	std::shared_ptr<SEP::genericRegFile> velFile = io->getRegFile("vel",usageIn);
	std::shared_ptr<SEP::hypercube> velHyper = velFile->getHyper();
	std::shared_ptr<SEP::float2DReg> velFloat(new SEP::float2DReg(velHyper));
	velFile->readFloatStream(velFloat);
	int nz = velFloat->getHyper()->getAxis(1).n;
	int nx = velFloat->getHyper()->getAxis(2).n;

    /********************************* Create sources vector ****************************/
	int nzSource = 1;
	int ozSource = par->getInt("zSource") - 1 + zPadMinus + fat;
	int dzSource = 1;
	int nxSource = 1;
	int oxSource = par->getInt("xSource") - 1 + xPadMinus + fat;
	int dxSource = 1;
	int spacingShots = par->getInt("spacingShots", spacingShots);
	axis sourceAxis(nxSource, oxSource, dxSource);
	std::vector<std::shared_ptr<deviceGpu>> sourcesVector;
	int nShot = par->getInt("nShot");
	for (int iShot; iShot<nShot; iShot++){
		std::shared_ptr<deviceGpu> sourceDevice(new deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velFloat, nts));
		sourcesVector.push_back(sourceDevice);
		oxSource = oxSource + spacingShots;
	}
	axis shotAxis = axis(nShot, oxSource, dxSource);

	/********************************* Create receivers vector **************************/
	int nzReceiver = 1;
	int ozReceiver = par->getInt("depthReceiver") - 1 + zPadMinus + fat;
	int dzReceiver = 1;
	int nxReceiver = par->getInt("nReceiver");
	int oxReceiver = par->getInt("oReceiver") - 1 + xPadMinus + fat;
	int dxReceiver = par->getInt("dReceiver");
	axis receiverAxis(nxReceiver, oxReceiver, dxReceiver);
	std::vector<std::shared_ptr<deviceGpu>> receiversVector;
	int nRecGeom = 1; // Constant receivers' geometry
	for (int iRec; iRec<nRecGeom; iRec++){
		std::shared_ptr<deviceGpu> recDevice(new deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velFloat, nts));
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

    /***************************** Create receivers signals vector **********************/
	// Read Wemva data
	receiversSignalsFile = io->getRegFile(std::string("wemvaData"),usageIn);
	std::shared_ptr <hypercube> receiversSignalsHyper = receiversSignalsFile->getHyper();
	std::vector<std::shared_ptr<float2DReg>> receiversSignalsVector;

	// Case where you only have one shot, one receiver
	if (receiversSignalsHyper->getNdim() == 1){
		axis a(1);
		receiversSignalsHyper->addAxis(a);
		receiversSignalsHyper->addAxis(a);
	}

	// Case where you only have one shot
	if (receiversSignalsHyper->getNdim() == 2){
		axis a(1);
		receiversSignalsHyper->addAxis(a);
	}

	receiversSignalsTempFloat = std::make_shared<float3DReg>(receiversSignalsHyper);
	receiversSignalsSliceFloat = std::make_shared<float2DReg>(receiversSignalsHyper->getAxis(1), receiversSignalsHyper->getAxis(2));
	receiversSignalsFile->readFloatStream(receiversSignalsTempFloat);

	for (int iShot=0; iShot<receiversSignalsHyper->getAxis(3).n; iShot++){
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<receiversSignalsHyper->getAxis(2).n; iReceiver++){
			for (int its=0; its<nts; its++){
				(*receiversSignalsSliceFloat->_mat)[iReceiver][its] = (*receiversSignalsTempFloat->_mat)[iShot][iReceiver][its];
			}
		}
		receiversSignalsVector.push_back(receiversSignalsSliceFloat);
	}

	/*********************************** Allocation *************************************/

	/* Forward Wemva */
	if (adj == 0) {

		/* Allocate and read model */
		model1File = io->getRegFile(std::string("model"),usageIn);
		std::shared_ptr <hypercube> model1Hyper = model1File->getHyper();
		model1Float = std::make_shared<float2DReg>(model1Hyper);
		model1File->readFloatStream(model1Float);

		/* Data allocation */
		std::shared_ptr<hypercube> data1Hyper(new hypercube(model1Float->getHyper()->getAxis(1), model1Float->getHyper()->getAxis(2), extAxis));
		data1Float = std::make_shared<float3DReg>(data1Hyper);

		/* Files shits */
		data1File = io->getRegFile(std::string("data"), usageOut);
		data1File->setHyper(data1Hyper);
		data1File->writeDescription();

	}

	/* Adjoint Wemva */
	if (adj == 1) {

		/* Allocate and read data */
		data1File = io->getRegFile(std::string("data"),usageIn);
		std::shared_ptr <hypercube> data1Hyper = data1File->getHyper();

		// Case where extension size is 1
		if (data1Hyper->getNdim() == 2){
			data1Hyper->addAxis(extAxis);
		}

		data1Float = std::make_shared<float3DReg>(data1Hyper);
		data1File->readFloatStream(data1Float);

		/* Allocate and read model */
		std::shared_ptr<hypercube> model1Hyper = velHyper;
		model1Float = std::make_shared<float2DReg>(model1Hyper);

		/* Stupid files shits */
		model1File = io->getRegFile(std::string("model"),usageOut);
		model1File->setHyper(model1Hyper);
		model1File->writeDescription();

	}

	/* Wavefields */
	std::shared_ptr<float3DReg> srcWavefield1Float, srcWavefield2Float, secWavefield1Float, secWavefield2Float;
	std::shared_ptr<genericRegFile> srcWavefield1File = io->getRegFile(std::string("srcWavefield"), usageOut);
	std::shared_ptr<genericRegFile> secWavefield1File = io->getRegFile(std::string("secWavefield1"), usageOut);
	std::shared_ptr<genericRegFile> secWavefield2File = io->getRegFile(std::string("secWavefield2"), usageOut);

	/************************************************************************************/
	/******************************** SIMULATIONS ***************************************/
	/************************************************************************************/

	/* Create tomo extended object */
	std::shared_ptr<wemvaExtShotsGpu> object1(new wemvaExtShotsGpu(velFloat, par, sourcesVector, sourcesSignalsVector, receiversVector, receiversSignalsVector));

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
			std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(velFloat->getHyper()->getAxis(1), velFloat->getHyper()->getAxis(2), timeAxisCoarse));

			// Write source wavefield
			srcWavefield1Float = object1->getSrcWavefield();
			srcWavefield1File->setHyper(wavefield1Hyper);
			srcWavefield1File->writeDescription();
			srcWavefield1File->writeFloatStream(srcWavefield1Float);

			// Write scattered wavefield #1
			secWavefield1Float = object1->getSecWavefield1();
			secWavefield1File->setHyper(wavefield1Hyper);
			secWavefield1File->writeDescription();
			secWavefield1File->writeFloatStream(secWavefield1Float);

			// Write scattered wavefield #2
			secWavefield2Float = object1->getSecWavefield2();
			secWavefield2File->setHyper(wavefield1Hyper);
			secWavefield2File->writeDescription();
			secWavefield2File->writeFloatStream(secWavefield2Float);

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
			std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(velFloat->getHyper()->getAxis(1), velFloat->getHyper()->getAxis(2), timeAxisCoarse));

			// Write source wavefield
			srcWavefield1Float = object1->getSrcWavefield();
			srcWavefield1File->setHyper(wavefield1Hyper);
			srcWavefield1File->writeDescription();
			srcWavefield1File->writeFloatStream(srcWavefield1Float);

			// Write scattered wavefield #1
			secWavefield1Float = object1->getSecWavefield1();
			secWavefield1File->setHyper(wavefield1Hyper);
			secWavefield1File->writeDescription();
			secWavefield1File->writeFloatStream(secWavefield1Float);

			// Write scattered wavefield #2
			secWavefield2Float = object1->getSecWavefield2();
			secWavefield2File->setHyper(wavefield1Hyper);
			secWavefield2File->writeDescription();
			secWavefield2File->writeFloatStream(secWavefield2Float);
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
