#ifndef FD_PARAM_H
#define FD_PARAM_H 1

#include <string>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include <iostream>

using namespace SEP;

class fdParam{

 	public:

		// Constructor
  		fdParam(const std::shared_ptr<float2DReg> vel, const std::shared_ptr<paramObj> par);
		// Destructor
  		~fdParam();

		// QC stuff
		bool checkParfileConsistencyTime(const std::shared_ptr<float2DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const;
		bool checkParfileConsistencySpace(const std::shared_ptr<float2DReg> model, std::string fileToCheck) const;
		bool checkParfileConsistencySpace(const std::shared_ptr<float3DReg> modelExt, std::string fileToCheck) const;

		bool checkFdStability(float courantMax=0.45);
		bool checkFdDispersion(float dispersionRatioMin=3.0);
		bool checkModelSize(); // Make sure the domain size (without the FAT) is a multiple of the dimblock size
		void getInfo();

		// Variables
		std::shared_ptr<paramObj> _par;
		std::shared_ptr<float2DReg> _vel, _smallVel;
		axis _timeAxisCoarse, _timeAxisFine, _zAxis, _xAxis, _extAxis;

		float *_vel2Dtw2, *_reflectivityScale;
		float _errorTolerance;
		float _minVel, _maxVel, _minDzDx, _maxDzDx;
		int _nts, _sub, _ntw;
		float _ots, _dts, _otw, _dtw, _oExt, _dExt;
		float _Courant, _dispersionRatio;
		int _nz, _nx, _nExt, _hExt;
		int _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _zPad, _xPad, _minPad;
		float _dz, _dx, _oz, _ox, _fMax;
		int _saveWavefield, _blockSize, _fat, _freeSurface;
		float _alphaCos;
		std::string _extension;

};

#endif
