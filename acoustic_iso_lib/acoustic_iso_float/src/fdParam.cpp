#include <string>
#include <float2DReg.h>
#include "fdParam.h"
#include <math.h>
#include <iomanip>
#include <iostream>
using namespace SEP;

fdParam::fdParam(const std::shared_ptr<float2DReg> vel, const std::shared_ptr<paramObj> par) {

	_vel = vel;
	_par = par;

	/***** Coarse time-sampling *****/
	_nts = _par->getInt("nts");
	_dts = _par->getFloat("dts",0.0);
	_ots = _par->getFloat("ots", 0.0);
	_sub = _par->getInt("sub");
	_timeAxisCoarse = axis(_nts, _ots, _dts);

	/***** Fine time-sampling *****/
	_ntw = (_nts - 1) * _sub + 1;
	_dtw = _dts / float(_sub);
	_otw = _ots;
	_timeAxisFine = axis(_ntw, _otw, _dtw);

	/***** Vertical axis *****/
	_nz = _par->getInt("nz");
	_zPadPlus = _par->getInt("zPadPlus");
	_zPadMinus = _par->getInt("zPadMinus");
	_zPad = std::min(_zPadMinus, _zPadPlus);
	_dz = _par->getFloat("dz",-1.0);
	_oz = _vel->getHyper()->getAxis(1).o;
	_zAxis = axis(_nz, _oz, _dz);

	/***** Horizontal axis *****/
	_nx = _par->getInt("nx");
	_xPadPlus = _par->getInt("xPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPad = std::min(_xPadMinus, _xPadPlus);
	_dx = _par->getFloat("dx",-1.0);
	_ox = _vel->getHyper()->getAxis(2).o;
	_xAxis = axis(_nx, _ox, _dx);

	/***** Extended axis *****/
	_nExt = _par->getInt("nExt", 1);
	if (_nExt % 2 == 0) {std::cout << "**** ERROR: Length of extended axis must be an uneven number ****" << std::endl; throw std::runtime_error(""); }
	_hExt = (_nExt-1)/2;
	_extension = par->getString("extension", "none");
	if (_nExt>1 && _extension=="time"){
		_oExt = -_dts*_hExt;
		_dExt = _dts;
		_extAxis = axis(_nExt, _oExt, _dExt);
	}
	if (_nExt>1 && _extension=="offset"){
		_oExt = -_dx*_hExt;
		_dExt = _dx;
		_extAxis = axis(_nExt, _oExt, _dExt);
	}

	/***** Other parameters *****/
	_fMax = _par->getFloat("fMax",1000.0);
	_blockSize = _par->getInt("blockSize");
	_fat = _par->getInt("fat");
	_minPad = std::min(_zPad, _xPad);
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_alphaCos = par->getFloat("alphaCos", 0.99);
	_errorTolerance = par->getFloat("errorTolerance", 0.000001);

	/***** QC *****/
	if( not checkParfileConsistencySpace(_vel, "Velocity file")){
		throw std::runtime_error("");
	}; // Parfile - velocity file consistency
	axis nzSmallAxis(_nz-2*_fat, 0.0, _dz);
	axis nxSmallAxis(_nx-2*_fat, 0.0, _dx);
	std::shared_ptr<SEP::hypercube> smallVelHyper(new hypercube(nzSmallAxis, nxSmallAxis));
	_smallVel = std::make_shared<float2DReg>(smallVelHyper);
	for (int ix = 0; ix < _nx-2*_fat; ix++){
		for (int iz = 0; iz < _nz-2*_fat; iz++){
			(*_smallVel->_mat)[ix][iz] = (*_vel->_mat)[ix+_fat][iz+_fat];
		}
	}

	if( not checkFdStability()){
		throw std::runtime_error("");
	};
	if( not checkFdDispersion()){
		throw std::runtime_error("");
	};
	if( not checkModelSize()){
		throw std::runtime_error("");
	};

	/***** Scaling for propagation *****/
	// v^2 * dtw^2
	_vel2Dtw2 = new float[_nz * _nx * sizeof(float)];
	for (int ix = 0; ix < _nx; ix++){
		int i1 = ix * _nz;
		for (int iz = 0; iz < _nz; iz++, i1++) {
			_vel2Dtw2[i1] = (*_vel->_mat)[ix][iz] * (*_vel->_mat)[ix][iz] * _dtw * _dtw;
		}
	}

	// Compute reflectivity scaling
	_reflectivityScale = new float[_nz * _nx * sizeof(float)];
	for (int ix = 0; ix < _nx; ix++){
		int i1 = ix * _nz;
		for (int iz = 0; iz < _nz; iz++, i1++) {
			_reflectivityScale[i1] = 2.0 / ( (*_vel->_mat)[ix][iz] * (*_vel->_mat)[ix][iz] * (*_vel->_mat)[ix][iz] );
		}
	}
}

void fdParam::getInfo(){

		std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << "************************ FD PARAMETERS INFO ***********************" << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << " " << std::endl;

		// Coarse time sampling
		std::cout << "------------------------ Coarse time sampling ---------------------" << std::endl;
		std::cout << std::fixed;
		std::cout << std::setprecision(3);
		std::cout << "nts = " << _nts << " [samples], dts = " << _dts << " [s], ots = " << _ots << " [s]" << std::endl;
		std::cout << std::setprecision(1);
		std::cout << "Nyquist frequency = " << 1.0/(2.0*_dts) << " [Hz]" << std::endl;
		std::cout << "Maximum frequency from seismic source = " << _fMax << " [Hz]" << std::endl;
		std::cout << std::setprecision(3);
		std::cout << "Total recording time = " << (_nts-1) * _dts << " [s]" << std::endl;
		std::cout << "Subsampling = " << _sub << std::endl;
		std::cout << " " << std::endl;

		// Coarse time sampling
		std::cout << "------------------------ Fine time sampling -----------------------" << std::endl;
		std::cout << "ntw = " << _ntw << " [samples], dtw = " << _dtw << " [s], otw = " << _otw << " [s]" << std::endl;
		std::cout << " " << std::endl;

		// Vertical spatial sampling
		std::cout << "--------------------------- Vertical axis -------------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "nz = " << _nz-2*_fat-_zPadMinus-_zPadPlus << " [samples], dz = " << _dz << " [km], oz = " << _oz+(_fat+_zPadMinus)*_dz << " [km]" << std::endl;
		std::cout << "Model thickness (area of interest) = " << _oz+(_fat+_zPadMinus)*_dz+(_nz-2*_fat-_zPadMinus-_zPadPlus-1)*_dz << " [km]" << std::endl;
		std::cout << "Top padding = " << _zPadMinus << " [samples], bottom padding = " << _zPadPlus << " [samples]" << std::endl;
		std::cout << "nz (padded) = " << _nz << " [samples], oz (padded) = " << _oz << " [km]" << std::endl;
		std::cout << "Model thickness (padding+fat) = " << _oz+(_nz-1)*_dz << " [km]" << std::endl;
		std::cout << " " << std::endl;

		// Horizontal spatial sampling
		std::cout << "-------------------------- Horizontal x-axis ----------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "nx = " << _nx-2*_fat-_xPadMinus-_xPadPlus << " [samples], dx = " << _dx << " [km], ox = " << _ox+(_fat+_xPadMinus)*_dx << " [km]" << std::endl;
		std::cout << "Model width (area of interest) = " << _ox+(_fat+_xPadMinus)*_dx+(_nx-2*_fat-_xPadMinus-_xPadPlus-1)*_dx << " [km]" << std::endl;
		std::cout << "Left padding = " << _xPadMinus << " [samples], right padding = " << _xPadPlus << " [samples]" << std::endl;
		std::cout << "nx (padded) = " << _nx << " [samples], ox (padded) = " << _ox << " [km]" << std::endl;
		std::cout << "Model width (padding+fat) = " << _ox+(_nx-1)*_dx << " [km]" << std::endl;
		std::cout << " " << std::endl;

		// Extended axis
		if ( (_nExt>1) && (_extension=="time")){
			std::cout << std::setprecision(3);
			std::cout << "-------------------- Extended axis: time-lags ---------------------" << std::endl;
			std::cout << "nTau = " << _hExt << " [samples], dTau= " << _dExt << " [s], oTau = " << _oExt << " [s]" << std::endl;
			std::cout << "Total extension length nTau = " << _nExt << " [samples], which corresponds to " << _nExt*_dExt << " [s]" << std::endl;
			std::cout << " " << std::endl;
		}

		if ( (_nExt>1) && (_extension=="offset") ){
			std::cout << std::setprecision(2);
			std::cout << "---------- Extended axis: horizontal subsurface offsets -----------" << std::endl;
			std::cout << "nOffset = " << _hExt << " [samples], dOffset= " << _dExt << " [km], oOffset = " << _oExt << " [km]" << std::endl;
			std::cout << "Total extension length nOffset = " << _nExt << " [samples], which corresponds to " << _nExt*_dExt << " [km]" << std::endl;
			std::cout << " " << std::endl;
		}

		// GPU FD parameters
		std::cout << "---------------------- GPU kernels parameters ---------------------" << std::endl;
		std::cout << "Block size in z-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cout << "Block size in x-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cout << "Halo size for Laplacian 10th order [FAT] = " << _fat << " [samples]" << std::endl;
		std::cout << " " << std::endl;

		// Stability and dispersion
		std::cout << "---------------------- Stability and dispersion -------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "Courant number = " << _Courant << " [-]" << std::endl;
		std::cout << "Dispersion ratio = " << _dispersionRatio << " [points/min wavelength]" << std::endl;
		std::cout << "Minimum velocity value = " << _minVel << " [km/s]" << std::endl;
		std::cout << "Maximum velocity value = " << _maxVel << " [km/s]" << std::endl;
		std::cout << std::setprecision(1);
		std::cout << "Maximum frequency without dispersion = " << _minVel/(3.0*std::max(_dz, _dx)) << " [Hz]" << std::endl;
		std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << " " << std::endl;
		std::cout << std::scientific; // Reset to scientific formatting notation
		std::cout << std::setprecision(6); // Reset the default formatting precision
}

bool fdParam::checkFdStability(float CourantMax){
	_maxVel = _smallVel->max();
	_minDzDx = std::min(_dz, _dx);
	_Courant = _maxVel * _dtw / _minDzDx;
	if (_Courant > CourantMax){
		std::cout << "**** ERROR: Courant is too big: " << _Courant << " ****" << std::endl;
		std::cout << "Max velocity value: " << _maxVel << std::endl;
		std::cout << "Dtw: " << _dtw << " [s]" << std::endl;
		std::cout << "Min (dz, dx): " << _minDzDx << " [km]" << std::endl;
		return false;
	}
	return true;
}

bool fdParam::checkFdDispersion(float dispersionRatioMin){

	_minVel = _smallVel->min();
	_maxDzDx = std::max(_dz, _dx);
	_dispersionRatio = _minVel / (_fMax*_maxDzDx);

	if (_dispersionRatio < dispersionRatioMin){
		std::cout << "**** ERROR: Dispersion is too small: " << _dispersionRatio <<  " > " << dispersionRatioMin << " ****" << std::endl;
		std::cout << "Min velocity value = " << _minVel << " [km/s]" << std::endl;
		std::cout << "Max (dz, dx) = " << _maxDzDx << " [km]" << std::endl;
		std::cout << "Max frequency = " << _fMax << " [Hz]" << std::endl;
		return false;
	}
	return true;
}

bool fdParam::checkModelSize(){
	if ( (_nz-2*_fat) % _blockSize != 0) {
		std::cout << "**** ERROR: nz not a multiple of block size ****" << std::endl;
		return false;
	}
	if ((_nx-2*_fat) % _blockSize != 0) {
		std::cout << "**** ERROR: nx not a multiple of block size ****" << std::endl;
		return false;
	}
	return true;
}

bool fdParam::checkParfileConsistencyTime(const std::shared_ptr<float2DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const {
	if (_nts != seismicTraces->getHyper()->getAxis(timeAxisIndex).n) {std::cout << "**** [" << fileToCheck << "] ERROR: nts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dts - seismicTraces->getHyper()->getAxis(timeAxisIndex).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: dts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ots - seismicTraces->getHyper()->getAxis(timeAxisIndex).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: ots not consistent with parfile ****" << std::endl; return false;}
	return true;
}

bool fdParam::checkParfileConsistencySpace(const std::shared_ptr<float2DReg> model, std::string fileToCheck) const {

	// Vertical axis
	if (_nz != model->getHyper()->getAxis(1).n) {std::cout << "**** ["<< fileToCheck << "] ERROR: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - model->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - model->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: oz not consistent with parfile ****" << std::endl; return false;}

	// Vertical axis
	if (_nx != model->getHyper()->getAxis(2).n) {std::cout << "**** [" << fileToCheck << "] ERROR: nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - model->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - model->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: ox not consistent with parfile ****" << std::endl; return false;}

	return true;
}

bool fdParam::checkParfileConsistencySpace(const std::shared_ptr<float3DReg> modelExt, std::string fileToCheck) const {
	// Vertical axis
	if (_nz != modelExt->getHyper()->getAxis(1).n) {std::cout << "**** ["<< fileToCheck << "] ERROR: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - modelExt->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - modelExt->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal axis
	if (_nx != modelExt->getHyper()->getAxis(2).n) {std::cout << "**** [" << fileToCheck << "] ERROR: nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - modelExt->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - modelExt->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: ox not consistent with parfile ****" << std::endl; return false;}

	// Extended axis
	if (_nExt != modelExt->getHyper()->getAxis(3).n) {std::cout << "**** [" << fileToCheck << "] ERROR: nExt not consistent with parfile ****" << std::endl; return false;}
	if (_nExt>1) {
		if ( std::abs(_dExt - modelExt->getHyper()->getAxis(3).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR: dExt not consistent with parfile ****" << std::endl; return false;}
		if ( std::abs(_oExt - modelExt->getHyper()->getAxis(3).o) > _errorTolerance ) { std::cout << "**** [" << fileToCheck << "] ERROR: oExt not consistent with parfile ****" << std::endl; return false;}
	}

	return true;
}

fdParam::~fdParam(){
	// Deallocate _vel2Dtw2 on host
	delete [] _vel2Dtw2;
	_vel2Dtw2 = NULL;
}
