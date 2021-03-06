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
#include "double1DReg.h"
#include "double2DReg.h"
#include "double3DReg.h"
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
	int dipole = par->getInt("dipole", 0);
	int zDipoleShift = par->getInt("zDipoleShift", 1);
	int xDipoleShift = par->getInt("xDipoleShift", 0);
	int dotProd = par->getInt("dotProd", 0);

	if (adj == 0 && dotProd == 0){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------ Running extended Wemva forward -----------------" << std::endl;
		std::cout << "--------------------- Double precision c++ code -------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	if (adj == 1 && dotProd == 0){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------ Running extended Wemva adjoint -----------------" << std::endl;
		std::cout << "--------------------- Double precision c++ code -------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}
	if (dotProd == 1){
		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "------------------------ Running dot product test -----------------" << std::endl;
		std::cout << "--------------------- Double precision c++ code -------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	/* Model and data declaration */
    // Sources signals
    std::shared_ptr<double2DReg> sourcesSignalTempdouble;
	std::shared_ptr<double2DReg> sourcesSignalTempDouble;

    // Wemva data
    std::shared_ptr<double3DReg> receiversSignalsTempdouble;
	std::shared_ptr<double2DReg> receiversSignalsSliceDouble;

    // Model
	std::shared_ptr<double2DReg> model1Double, model2Double;
	std::shared_ptr<double2DReg> model1double, model2double;

    // Image
	std::shared_ptr<double3DReg> data1Double, data2Double;
	std::shared_ptr<double3DReg> data1double, data2double;

    // Files
    std::shared_ptr <genericRegFile> model1File, model2File, data1File, data2File, sourcesSignalsFile, receiversSignalsFile;
	std::shared_ptr <hypercube> model1Hyper, data1Hyper;

	/* Read time parameters */
	int nts = par->getInt("nts");
	double dts = par->getdouble("dts", 0.0);
	int sub = par->getInt("sub");
	axis timeAxisCoarse = axis(nts, 0.0, dts);
	int ntw = (nts - 1) * sub + 1;
	double dtw = dts / double(sub);
	axis timeAxisFine = axis(ntw, 0.0, dtw);

	/* Read extension parameters */
	axis extAxis;
	std::string extension = par->getString("extension", "none");
	int nExt = par->getInt("nExt", 1);
	if (nExt%2 == 0){std::cout << "**** ERROR: Length of extended axis must be an uneven number ****" << std::endl; throw std::runtime_error("");}
	int hExt = (nExt-1)/2;
	if (extension == "time"){
		double dExt = par->getdouble("dExt", dts);
		double oExt = par->getdouble("oExt", -dExt*hExt);
		extAxis = axis(nExt, oExt, dExt);
	} else {
		double dExt = par->getdouble("dExt", par->getdouble("dx", -1.0));
		double oExt = par->getdouble("oExt", -dExt*hExt);
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
	std::shared_ptr<SEP::double2DReg> veldouble(new SEP::double2DReg(velHyper));
	std::shared_ptr<SEP::double2DReg> velDouble(new SEP::double2DReg(velHyper));
	velFile->readFloatStream(veldouble);
	int nz = veldouble->getHyper()->getAxis(1).n;
	int nx = veldouble->getHyper()->getAxis(2).n;
	for (int ix = 0; ix < nx; ix++) {
		for (int iz = 0; iz < nz; iz++) {
			(*velDouble->_mat)[ix][iz] = (*veldouble->_mat)[ix][iz];
		}
	}

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
		std::shared_ptr<deviceGpu> sourceDevice(new deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velDouble, nts, dipole, zDipoleShift, xDipoleShift));
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
		std::shared_ptr<deviceGpu> recDevice(new deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velDouble, nts, dipole, zDipoleShift, xDipoleShift));
		receiversVector.push_back(recDevice);
	}

	/******************************* Create sources signals vector **********************/
	// Read sources signals file - we use one identical wavelet for all shots
	sourcesSignalsFile = io->getRegFile(std::string("sources"),usageIn);
	std::shared_ptr <hypercube> sourcesSignalsHyper = sourcesSignalsFile->getHyper();
	std::vector<std::shared_ptr<double2DReg>> sourcesSignalsVector;

	if (sourcesSignalsHyper->getNdim() == 1){
		axis a(1);
		sourcesSignalsHyper->addAxis(a);
	}
	sourcesSignalTempdouble = std::make_shared<double2DReg>(sourcesSignalsHyper);
	sourcesSignalTempDouble = std::make_shared<double2DReg>(sourcesSignalsHyper);
	sourcesSignalsFile->readFloatStream(sourcesSignalTempdouble);
	for (int its=0; its<nts; its++){
		(*sourcesSignalTempDouble->_mat)[0][its] = (*sourcesSignalTempdouble->_mat)[0][its];
	}
	sourcesSignalsVector.push_back(sourcesSignalTempDouble);

    /***************************** Create receivers signals vector **********************/
	// Read Wemva data
	receiversSignalsFile = io->getRegFile(std::string("wemvaData"),usageIn);
	std::shared_ptr <hypercube> receiversSignalsHyper = receiversSignalsFile->getHyper();
	std::vector<std::shared_ptr<double2DReg>> receiversSignalsVector;

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

	receiversSignalsTempdouble = std::make_shared<double3DReg>(receiversSignalsHyper);
	receiversSignalsFile->readFloatStream(receiversSignalsTempdouble);

	for (int iShot=0; iShot<receiversSignalsHyper->getAxis(3).n; iShot++){
		receiversSignalsSliceDouble = std::make_shared<double2DReg>(receiversSignalsHyper->getAxis(1), receiversSignalsHyper->getAxis(2));
		for (int iReceiver=0; iReceiver<receiversSignalsHyper->getAxis(2).n; iReceiver++){
			for (int its=0; its<nts; its++){
				(*receiversSignalsSliceDouble->_mat)[iReceiver][its] = (*receiversSignalsTempdouble->_mat)[iShot][iReceiver][its];
			}
		}
		receiversSignalsVector.push_back(receiversSignalsSliceDouble);
	}

	/*********************************** Allocation *************************************/

	/* Forward Wemva */
	if (adj == 0) {

		/* Allocate and read model */
		model1File = io->getRegFile(std::string("model"),usageIn);
		std::shared_ptr <hypercube> model1Hyper = model1File->getHyper();
		model1double = std::make_shared<double2DReg>(model1Hyper);
		model1Double = std::make_shared<double2DReg>(model1Hyper);
		model1File->readFloatStream(model1double);
		for (int ix=0; ix<model1Hyper->getAxis(2).n; ix++) {
			for (int iz=0; iz<model1Hyper->getAxis(1).n; iz++) {
				(*model1Double->_mat)[ix][iz] = (*model1double->_mat)[ix][iz];
			}
		}

		/* Data allocation */
		std::shared_ptr<hypercube> data1Hyper(new hypercube(model1Double->getHyper()->getAxis(1), model1Double->getHyper()->getAxis(2), extAxis));
		data1Double = std::make_shared<double3DReg>(data1Hyper);
		data1double = std::make_shared<double3DReg>(data1Hyper);

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

		data1double = std::make_shared<double3DReg>(data1Hyper);
		data1Double = std::make_shared<double3DReg>(data1Hyper);
		data1File->readFloatStream(data1double);
		for (int iExt=0; iExt<nExt; iExt++){
			for (int ix=0; ix<data1Hyper->getAxis(2).n; ix++) {
				for (int iz=0; iz<data1Hyper->getAxis(1).n; iz++) {
					(*data1Double->_mat)[iExt][ix][iz] = (*data1double->_mat)[iExt][ix][iz];
				}
			}
		}

		/* Allocate and read model */
		std::shared_ptr<hypercube> model1Hyper = velHyper;
		model1double = std::make_shared<double2DReg>(model1Hyper);
		model1Double = std::make_shared<double2DReg>(model1Hyper);

		/* Stupid files shits */
		model1File = io->getRegFile(std::string("model"),usageOut);
		model1File->setHyper(model1Hyper);
		model1File->writeDescription();

	}

	/* Wavefields */
	std::shared_ptr<double3DReg> srcWavefield1Double, srcWavefield2Double, secWavefield1Double, secWavefield2Double;
	std::shared_ptr<double3DReg> srcWavefield1double, srcWavefield2double, secWavefield1double, secWavefield2double;
	std::shared_ptr<genericRegFile> srcWavefield1File = io->getRegFile(std::string("srcWavefield"), usageOut);
	std::shared_ptr<genericRegFile> secWavefield1File = io->getRegFile(std::string("secWavefield1"), usageOut);
	std::shared_ptr<genericRegFile> secWavefield2File = io->getRegFile(std::string("secWavefield2"), usageOut);

	/************************************************************************************/
	/******************************** SIMULATIONS ***************************************/
	/************************************************************************************/

	/* Create tomo extended object */
	std::shared_ptr<wemvaExtShotsGpu> object1(new wemvaExtShotsGpu(velDouble, par, sourcesVector, sourcesSignalsVector, receiversVector, receiversSignalsVector));

	/********************************** FORWARD *****************************************/
	if (adj == 0 && dotProd ==0) {

		if (saveWavefield == 1){
			object1->forwardWavefield(false, model1Double, data1Double);
		} else {
			object1->forward(false, model1Double, data1Double);
		}

		// Data
		for (int iExt=0; iExt<nExt; iExt++){
			for (int ix=0; ix<data1Double->getHyper()->getAxis(2).n; ix++) {
				for (int iz=0; iz<data1Double->getHyper()->getAxis(1).n; iz++) {
					(*data1double->_mat)[iExt][ix][iz] = (*data1Double->_mat)[iExt][ix][iz];
				}
			}
		}
		data1File->writeFloatStream(data1double);

		/* Wavefield */
		if (saveWavefield == 1){
			std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(veldouble->getHyper()->getAxis(1), veldouble->getHyper()->getAxis(2), timeAxisCoarse));
			srcWavefield1Double = object1->getSrcWavefield();
			secWavefield1Double = object1->getSecWavefield1();
			secWavefield2Double = object1->getSecWavefield2();
			srcWavefield1double = std::make_shared<double3DReg>(srcWavefield1Double->getHyper());
			secWavefield1double = std::make_shared<double3DReg>(srcWavefield1Double->getHyper());
			secWavefield2double = std::make_shared<double3DReg>(srcWavefield1Double->getHyper());

			#pragma omp parallel for
			for (int its = 0; its < nts; its++){
				for (int ix = 0; ix < nx; ix++){
					for (int iz = 0; iz < nz; iz++){
						(*srcWavefield1double->_mat)[its][ix][iz] = (*srcWavefield1Double->_mat)[its][ix][iz];
						(*secWavefield1double->_mat)[its][ix][iz] = (*secWavefield1Double->_mat)[its][ix][iz];
						(*secWavefield2double->_mat)[its][ix][iz] = (*secWavefield2Double->_mat)[its][ix][iz];
					}
				}
			}

			// Write source wavefield
			srcWavefield1File->setHyper(wavefield1Hyper);
			srcWavefield1File->writeDescription();
			srcWavefield1File->writeFloatStream(srcWavefield1double);

			// Write scattered wavefield #1
			secWavefield1File->setHyper(wavefield1Hyper);
			secWavefield1File->writeDescription();
			secWavefield1File->writeFloatStream(secWavefield1double);

			// Write scattered wavefield #2
			secWavefield2File->setHyper(wavefield1Hyper);
			secWavefield2File->writeDescription();
			secWavefield2File->writeFloatStream(secWavefield2double);

		}
	}

	/********************************** ADJOINT *****************************************/
	if (adj == 1 && dotProd ==0) {

		if (saveWavefield == 1){
			object1->adjointWavefield(false, model1Double, data1Double);
		} else {
			object1->adjoint(false, model1Double, data1Double);
		}

		// Model
		for (int ix=0; ix<model1Double->getHyper()->getAxis(2).n; ix++) {
			for (int iz=0; iz<model1Double->getHyper()->getAxis(1).n; iz++) {
				(*model1double->_mat)[ix][iz] = (*model1Double->_mat)[ix][iz];
			}
		}
		model1File->writeFloatStream(model1double);

		/* Wavefield */
		if (saveWavefield == 1){
			std::shared_ptr<hypercube> wavefield1Hyper(new hypercube(veldouble->getHyper()->getAxis(1), veldouble->getHyper()->getAxis(2), timeAxisCoarse));
			srcWavefield1Double = object1->getSrcWavefield();
			secWavefield1Double = object1->getSecWavefield1();
			secWavefield2Double = object1->getSecWavefield2();
			srcWavefield1double = std::make_shared<double3DReg>(srcWavefield1Double->getHyper());
			secWavefield1double = std::make_shared<double3DReg>(srcWavefield1Double->getHyper());
			secWavefield2double = std::make_shared<double3DReg>(srcWavefield1Double->getHyper());
			for (int its = 0; its < nts; its++){
				for (int ix = 0; ix < nx; ix++){
					for (int iz = 0; iz < nz; iz++){
						(*srcWavefield1double->_mat)[its][ix][iz] = (*srcWavefield1Double->_mat)[its][ix][iz];
						(*secWavefield1double->_mat)[its][ix][iz] = (*secWavefield1Double->_mat)[its][ix][iz];
						(*secWavefield2double->_mat)[its][ix][iz] = (*secWavefield2Double->_mat)[its][ix][iz];
					}
				}
			}
			srcWavefield1File->setHyper(wavefield1Hyper);
			srcWavefield1File->writeDescription();
			srcWavefield1File->writeFloatStream(srcWavefield1double);
			secWavefield1File->setHyper(wavefield1Hyper);
			secWavefield1File->writeDescription();
			secWavefield1File->writeFloatStream(secWavefield1double);
			secWavefield2File->setHyper(wavefield1Hyper);
			secWavefield2File->writeDescription();
			secWavefield2File->writeFloatStream(secWavefield2double);
		}

	}

	/* Dot product test */
	if (dotProd == 1) {
		object1->setDomainRange(model1Double, data1Double);
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
