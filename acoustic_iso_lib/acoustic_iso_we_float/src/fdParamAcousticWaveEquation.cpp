#include <string>
#include <float2DReg.h>
#include "fdParamAcousticWaveEquation.h"
#include <math.h>
#include <iomanip>
#include <iostream>
#include <cstring>
using namespace SEP;

fdParamAcousticWaveEquation::fdParamAcousticWaveEquation(const std::shared_ptr<float2DReg> slsq, const std::shared_ptr<paramObj> par) {

	_slsq = slsq; 
	_par = par;

	/***** Coarse time-sampling *****/
	_nts = _par->getInt("nts");
	_dts = _par->getFloat("dts",0.0);
	_ots = _par->getFloat("ots", 0.0);
	_timeAxisCoarse = axis(_nts, _ots, _dts);

	/***** Vertical axis *****/
	_nz = _par->getInt("nz");
	_zPadPlus = _par->getInt("zPadPlus",0);
	_zPadMinus = _par->getInt("zPadMinus",0);
	_zPad = std::min(_zPadMinus, _zPadPlus);
	_dz = _par->getFloat("dz",-1.0);
	_oz = _slsq->getHyper()->getAxis(1).o;
	_zAxis = axis(_nz, _oz, _dz);

	/***** Horizontal axis *****/
	_nx = _par->getInt("nx");
	_xPadPlus = _par->getInt("xPadPlus",0);
	_xPadMinus = _par->getInt("xPadMinus",0);
	_xPad = std::min(_xPadMinus, _xPadPlus);
	_dx = _par->getFloat("dx",-1.0);
	_ox = _slsq->getHyper()->getAxis(2).o;
	_xAxis = axis(_nx, _ox, _dx);

	/***** Other parameters *****/
	_fMax = _par->getFloat("fMax",1000.0);
	_blockSize = _par->getInt("blockSize");
	_fat = _par->getInt("fat");
	_minPad = std::min(_zPad, _xPad);
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_alphaCos = par->getFloat("alphaCos", 0.99);
	_errorTolerance = par->getFloat("errorTolerance", 0.000001);

	_minVp = 10000;
	_maxVp = -1;
	//#pragma omp for collapse(2)
	for (int ix = _fat; ix < _nx-2*_fat; ix++){
		for (int iz = _fat; iz < _nz-2*_fat; iz++){
			float slsqTemp = (*_slsq->_mat)[ix][iz];
			float vpTemp = std::pow(1/(slsqTemp),0.5);

			if (vpTemp < _minVp) _minVp = vpTemp;
			if (vpTemp > _maxVp) _maxVp = vpTemp;
		}
	}

	/***** QC *****/
	assert(checkParfileConsistencySpace(_slsq)); // Parfile - velocity file consistency
	// assert(checkFdStability());
	// assert(checkFdDispersion());
	assert(checkModelSize());

	/***** Scaling for propagation *****/
	_slsqDt2 = new float[_nz * _nx * sizeof(float)]; 
	_cosDamp = new float[_nz * _nx * sizeof(float)]; 

	//initialize 2d slices
	_slsqDt2Reg = std::make_shared<float2DReg>(_slsq->getHyper()->getAxis(1), _slsq->getHyper()->getAxis(2));
	_cosDampReg = std::make_shared<float2DReg>(_slsq->getHyper()->getAxis(1), _slsq->getHyper()->getAxis(2));
	_cosDampReg->set(1.0);

	//scaling factor 
	//#pragma omp for collapse(2)
	for (int ix = 0; ix < _nx; ix++){
		for (int iz = 0; iz < _nz; iz++) {
			(*_slsqDt2Reg->_mat)[ix][iz] = (*_slsq->_mat)[ix][iz]/(_dts*_dts);
			float zscale=1;
			float arg=0;
			float xscale=1;
			if(ix<_fat) {
				xscale=0;
			}
			else if(ix<_xPadMinus+_fat){
				arg=M_PI / (1.0 * _xPadMinus) * 1.0 * (_xPadMinus-ix+_fat);
				 xscale= _alphaCos + (1-_alphaCos) * cos(arg);
			}
			if(ix>_nx-_fat-1){
				 xscale=0; 
			}
			else if(ix>_nx-_xPadPlus-_fat-1){
				arg=M_PI / (1.0 * _xPadPlus) * 1.0 * (_nx-_xPadPlus-ix-_fat-1);
				 xscale= _alphaCos + (1-_alphaCos) * cos(arg);
			}
			if(iz<_fat){
				 zscale=0; 
			}
			else if(iz<_zPadMinus+_fat){
				arg=M_PI / (1.0 * _zPadMinus) * 1.0 * (_zPadMinus-iz+_fat);
				 zscale= _alphaCos + (1-_alphaCos) * cos(arg);
			}
			if(iz>_nz-_fat-1){
				 zscale=0; 
			}	
			else if(iz>_nz-_zPadPlus-_fat-1){
				arg=M_PI / (1.0 * _zPadPlus) * 1.0 * (_nz-_zPadPlus-iz-_fat-1);
				 zscale= _alphaCos + (1-_alphaCos) * cos(arg);
			}
			(*_cosDampReg->_mat)[ix][iz] = zscale*xscale;
		}
	}

	// //get pointer to float array holding values. This is later passed to the device.
	_slsqDt2 = _slsqDt2Reg->getVals();
	_cosDamp= _cosDampReg->getVals();

}

void fdParamAcousticWaveEquation::getInfo(){

		std::cerr << " " << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << "************************ FD PARAMETERS INFO ***********************" << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;

		// Coarse time sampling
		std::cerr << "------------------------ Coarse time sampling ---------------------" << std::endl;
		std::cerr << std::fixed;
		std::cerr << std::setprecision(3);
		std::cerr << "nts = " << _nts << " [samples], dts = " << _dts << " [s], ots = " << _ots << " [s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Nyquist frequency = " << 1.0/(2.0*_dts) << " [Hz]" << std::endl;
		std::cerr << "Maximum frequency from seismic source = " << _fMax << " [Hz]" << std::endl;
		std::cerr << "Samples within minimum period = " << 1.0/(2.0*_fmax)/_dts << std::endl;
		std::cerr << std::setprecision(6);
		std::cerr << "Total recording time = " << (_nts-1) * _dts << " [s]" << std::endl;
		std::cerr << " " << std::endl;


		// Vertical spatial sampling
		std::cerr << "-------------------- Vertical spatial sampling --------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nz = " << _nz-2*_fat-_zPadMinus-_zPadPlus << " [samples], dz = " << _dz << "[km], oz = " << _oz+(_fat+_zPadMinus)*_dz << " [km]" << std::endl;
		std::cerr << "Model depth = " << _oz+(_nz-2*_fat-_zPadMinus-_zPadPlus-1)*_dz << " [km]" << std::endl;
		std::cerr << "Top padding = " << _zPadMinus << " [samples], bottom padding = " << _zPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Horizontal spatial sampling
		std::cerr << "-------------------- Horizontal spatial sampling ------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nx = " << _nx << " [samples], dx = " << _dx << " [km], ox = " << _ox+(_fat+_xPadMinus)*_dx << " [km]" << std::endl;
		std::cerr << "Model width = " << _ox+(_nx-2*_fat-_xPadMinus-_xPadPlus-1)*_dx << " [km]" << std::endl;
		std::cerr << "Left padding = " << _xPadMinus << " [samples], right padding = " << _xPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;
		// GPU FD parameters
		std::cerr << "---------------------- GPU kernels parameters ---------------------" << std::endl;
		std::cerr << "Block size in z-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Block size in x-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Halo size for staggered 8th-order derivative [FAT] = " << _fat << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Stability and dispersion
		std::cerr << "---------------------- Stability and dispersion -------------------" << std::endl;
		std::cerr << std::setprecision(2);
		// std::cerr << "Courant number = " << _Courant << " [-]" << std::endl;
		// std::cerr << "Dispersion ratio = " << _dispersionRatio << " [points/min wavelength]" << std::endl;
		std::cerr << "Minimum velocity value (of either vp or vs) = " << _minVp << " [km/s]" << std::endl;
		std::cerr << "Maximum velocity value (of either vp or vs) = " << _maxVp << " [km/s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Maximum frequency without dispersion = " << _minVp/(3.0*std::max(_dz, _dx)) << " [Hz]" << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << std::scientific; // Reset to scientific formatting notation
		std::cerr << std::setprecision(6); // Reset the default formatting precision
}


bool fdParamAcousticWaveEquation::checkModelSize(){
	if ( (_nz-2*_fat) % _blockSize != 0) {
		std::cerr << "**** ERROR: nz not a multiple of block size ****" << std::endl;
		return false;
	}
	if ((_nx-2*_fat) % _blockSize != 0) {
		std::cerr << "**** ERROR: nx not a multiple of block size ****" << std::endl;
		return false;
	}
	return true;
}

bool fdParamAcousticWaveEquation::checkParfileConsistencyTime(const std::shared_ptr<float2DReg> seismicTraces, int timeAxisIndex,  std::string fileToCheck) const {
	if (_nts != seismicTraces->getHyper()->getAxis(timeAxisIndex).n) {std::cerr << "**** ERROR: nts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dts - seismicTraces->getHyper()->getAxis(timeAxisIndex).d) > _errorTolerance ) {std::cerr << "**** ERROR: dts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ots - seismicTraces->getHyper()->getAxis(timeAxisIndex).o) > _errorTolerance ) {std::cerr << "**** ERROR: ots not consistent with parfile ****" << std::endl; return false;}
	return true;
}

bool fdParamAcousticWaveEquation::checkParfileConsistencySpace(const std::shared_ptr<float2DReg> model) const {

	// Vertical axis
	if (_nz != model->getHyper()->getAxis(1).n) {std::cerr << "**** ERROR: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - model->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cerr << "**** ERROR: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - model->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cerr << "**** ERROR: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal axis
	if (_nx != model->getHyper()->getAxis(2).n) {std::cerr << "**** ERROR nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - model->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cerr << "**** ERROR: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - model->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cerr << "**** ERROR: ox not consistent with parfile ****" << std::endl; return false;}

	return true;
}

// bool fdParamAcoustic::checkParfileConsistencySpace(const std::shared_ptr<float2DReg> modelExt) const {

// 	// Vertical axis
// 	if (_nz != modelExt->getHyper()->getAxis(1).n) {std::cerr << "**** ERROR: nz not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_dz - modelExt->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cerr << "**** ERROR: dz not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_oz - modelExt->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cerr << "**** ERROR: oz not consistent with parfile ****" << std::endl; return false;}

// 	// Vertical axis
// 	if (_nx != modelExt->getHyper()->getAxis(2).n) {std::cerr << "**** ERROR: nx not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_dx - modelExt->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cerr << "**** ERROR: dx not consistent with parfile ****" << std::endl; return false;}
// 	if ( std::abs(_ox - modelExt->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cerr << "**** ERROR: ox not consistent with parfile ****" << std::endl; return false;}

// 	// Extended axis
// 	if (_nExt != modelExt->getHyper()->getAxis(3).n) {std::cerr << "**** ERROR: nExt not consistent with parfile ****" << std::endl; return false;}
// 	if (_nExt>1) {
// 		if ( std::abs(_dExt - modelExt->getHyper()->getAxis(3).d) > _errorTolerance ) {std::cerr << "**** ERROR: dExt not consistent with parfile ****" << std::endl; return false;}
// 		if ( std::abs(_oExt - modelExt->getHyper()->getAxis(3).o) > _errorTolerance ) { std::cerr << "**** ERROR: oExt not consistent with parfile ****" << std::endl; return false;}
// 	}

// 	return true;
// }

fdParamAcousticWaveEquation::~fdParamAcousticWaveEquation(){
	_slsqDt2 = NULL;
}

