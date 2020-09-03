#ifndef FD_PARAM_H
#define FD_PARAM_H 1

#include <string>
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include <iostream>

using namespace SEP;

class fdParam{

 	public:

		// Constructor
  		fdParam(const std::shared_ptr<double2DReg> vel, const std::shared_ptr<paramObj> par);
		// Destructor
  		~fdParam();

		// QC stuff
		bool checkParfileConsistencyTime(const std::shared_ptr<double2DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const;
		bool checkParfileConsistencySpace(const std::shared_ptr<double2DReg> model, std::string fileToCheck) const;
		bool checkParfileConsistencySpace(const std::shared_ptr<double3DReg> modelExt, std::string fileToCheck) const;

		bool checkFdStability(double courantMax=0.45);
		bool checkFdDispersion(double dispersionRatioMin=3.0);
		bool checkModelSize(); // Make sure the domain size (without the FAT) is a multiple of the dimblock size
		void getInfo();

		// Variables
		std::shared_ptr<paramObj> _par;
		std::shared_ptr<double2DReg> _vel, _smallVel;
		axis _timeAxisCoarse, _timeAxisFine, _zAxis, _xAxis, _extAxis;

		double *_vel2Dtw2, *_reflectivityScale;
		double _errorTolerance;
		double _minVel, _maxVel, _minDzDx, _maxDzDx;
		int _nts, _sub, _ntw;
		double _ots, _dts, _otw, _dtw, _oExt, _dExt;
		double _Courant, _dispersionRatio;
		int _nz, _nx, _nExt, _hExt;
		int _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _zPad, _xPad, _minPad;
		double _dz, _dx, _oz, _ox, _fMax;
		int _saveWavefield, _blockSize, _fat, _freeSurface;
		double _alphaCos;
		std::string _extension;

};

#endif
