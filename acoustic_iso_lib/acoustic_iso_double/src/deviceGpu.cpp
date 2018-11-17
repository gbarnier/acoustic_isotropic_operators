#include <double1DReg.h>
#include <double2DReg.h>
#include <iostream>
#include "deviceGpu.h"
#include <vector>

// Constructor #1
deviceGpu::deviceGpu(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double2DReg> vel, int &nt) {

	_vel = vel;
	_zCoord = zCoord;
	_xCoord = xCoord;
	checkOutOfBounds(_zCoord, _xCoord, _vel); // Make sure no device is on the edge of the domain
	_nDeviceIrreg = _zCoord->getHyper()->getAxis(1).n; // Nb of devices on irregular grid
	_nt = nt;
	int _nz = vel->getHyper()->getAxis(1).n;

	_gridPointIndex = new int[4*_nDeviceIrreg]; // Index of all the neighboring points of each device (non-unique) on the regular "1D" grid
	_weight = new double[4*_nDeviceIrreg]; // Weights for spatial interpolation

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		// Find the 4 neighboring points for all devices and compute the weights for the spatial interpolation
		int i1 = iDevice * 4;
		double wz = ( (*_zCoord->_mat)[iDevice] - vel->getHyper()->getAxis(1).o ) / vel->getHyper()->getAxis(1).d;
		double wx = ( (*_xCoord->_mat)[iDevice] - vel->getHyper()->getAxis(2).o ) / vel->getHyper()->getAxis(2).d;
		int zReg = wz; // z-coordinate on regular grid
		wz = wz - zReg;
		wz = 1.0 - wz;
		int xReg = wx; // x-coordinate on regular grid
		wx = wx - xReg;
		wx = 1.0 - wx;

		// Top left
		_gridPointIndex[i1] = xReg * _nz + zReg; // Index of this point for a 1D array representation
		_weight[i1] = wz * wx;

		// Bottom left
		_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
		_weight[i1+1] = (1.0 - wz) * wx;

		// Top right
		_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
		_weight[i1+2] = wz * (1.0 - wx);

		// Bottom right
		_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
		_weight[i1+3] = (1.0 - wz) * (1.0 - wx);
	}
	convertIrregToReg();
}

// Constructor #2
deviceGpu::deviceGpu(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<double2DReg> vel, int &nt) {

	_vel = vel;
	_nt = nt;
	_nDeviceIrreg = zGridVector.size(); // Nb of device
	int _nz = vel->getHyper()->getAxis(1).n;
	checkOutOfBounds(zGridVector, xGridVector, _vel); // Make sure no device is on the edge of the domain
	_gridPointIndex = new int[4*_nDeviceIrreg]; // All the neighboring points of each device (non-unique)
	_weight = new double[4*_nDeviceIrreg]; // Weights for spatial interpolation

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		int i1 = iDevice * 4;

		// Top left
		_gridPointIndex[i1] = xGridVector[iDevice] * _nz + zGridVector[iDevice];
		_weight[i1] = 1.0;

		// Bottom left
		_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
		_weight[i1+1] = 0.0;

		// Top right
		_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
		_weight[i1+2] = 0.0;

		// Bottom right
		_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
		_weight[i1+3] = 0.0;
	}
	convertIrregToReg();
}

// Constructor #3
deviceGpu::deviceGpu(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<double2DReg> vel, int &nt){

	_vel = vel;
	_nt = nt;
	_nDeviceIrreg = nzDevice * nxDevice; // Nb of devices on irregular grid
	int _nz = vel->getHyper()->getAxis(1).n;
	checkOutOfBounds(nzDevice, ozDevice, dzDevice , nxDevice, oxDevice, dxDevice, vel);
	_gridPointIndex = new int[4*_nDeviceIrreg]; // All the neighboring points of each device (non-unique)
	_weight = new double[4*_nDeviceIrreg]; // Weights for spatial interpolation

	int iDevice = -1;
	for (int ix = 0; ix < nxDevice; ix++) {
		int ixDevice = oxDevice + ix * dxDevice; // x-position of device on FD grid
		for (int iz = 0; iz < nzDevice; iz++) {
			int izDevice = ozDevice + iz * dzDevice; // z-position of device on FD grid
			iDevice++;
			int i1 = iDevice * 4;

			// Top left
			_gridPointIndex[i1] = ixDevice * _nz + izDevice;
			_weight[i1] = 1.0;

			// Bottom left
			_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
			_weight[i1+1] = 0.0;

			// Top right
			_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
			_weight[i1+2] = 0.0;

			// Bottom right
			_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
			_weight[i1+3] = 0.0;

		}
	}
	convertIrregToReg();
}

void deviceGpu::convertIrregToReg() {

	/* (1) Create map where:
		- Key = excited grid point index (points are unique)
		- Value = signal trace number
		(2) Create a vector containing the indices of the excited grid points
	*/

	_nDeviceReg = 0; // Initialize the number of regular devices to zero
	_gridPointIndexUnique.clear(); // Initialize to empty vector

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over gridPointIndex array
		for (int iCorner = 0; iCorner < 4; iCorner++){
			int i1 = iDevice * 4 + iCorner;

			// If the grid point is not already in the list
			if (_indexMap.count(_gridPointIndex[i1]) == 0) {
				_nDeviceReg++; // Increment the number of (unique) grid points excited by the signal
				_indexMap[_gridPointIndex[i1]] = _nDeviceReg - 1; // Add the pair to the map
				_gridPointIndexUnique.push_back(_gridPointIndex[i1]); // Append vector containing all unique grid point index
			}
		}
	}
}

void deviceGpu::forward(const bool add, const std::shared_ptr<double2DReg> signalReg, std::shared_ptr<double2DReg> signalIrreg) const {

	/* FORWARD: Go from REGULAR grid -> IRREGULAR grid */
	if (!add) signalIrreg->scale(0.0);

	std::shared_ptr<double2D> d = signalIrreg->_mat;
	std::shared_ptr<double2D> m = signalReg->_mat;
	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
		for (int iCorner = 0; iCorner < 4; iCorner++){ // Loop over neighboring points on regular grid
			int i1 = iDevice * 4 + iCorner;
			int i2 = _indexMap.find(_gridPointIndex[i1])->second;
			for (int it = 0; it < _nt; it++){
				(*d)[iDevice][it] += _weight[i1] * (*m)[i2][it];
			}
		}
	}
}

void deviceGpu::adjoint(const bool add, std::shared_ptr<double2DReg> signalReg, const std::shared_ptr<double2DReg> signalIrreg) const {

	/* ADJOINT: Go from IRREGULAR grid -> REGULAR grid */
	if (!add) signalReg->scale(0.0);
	std::shared_ptr<double2D> d = signalIrreg->_mat;
	std::shared_ptr<double2D> m = signalReg->_mat;

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
		for (int iCorner = 0; iCorner < 4; iCorner++){ // Loop over neighboring points on regular grid
			int i1 = iDevice * 4 + iCorner; // Grid point index
			int i2 = _indexMap.find(_gridPointIndex[i1])->second; // Get trace number for signalReg
			for (int it = 0; it < _nt; it++){
				(*m)[i2][it] += _weight[i1] * (*d)[iDevice][it];
			}
		}
	}
}

void deviceGpu::checkOutOfBounds(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double2DReg> vel){

	int nDevice = zCoord->getHyper()->getAxis(1).n;
	double zMax = vel->getHyper()->getAxis(1).o + (vel->getHyper()->getAxis(1).n - 1) * vel->getHyper()->getAxis(1).d;
	double xMax = vel->getHyper()->getAxis(2).o + (vel->getHyper()->getAxis(2).n - 1) * vel->getHyper()->getAxis(2).d;
	for (int iDevice = 0; iDevice < nDevice; iDevice++){
		if ( ((*zCoord->_mat)[iDevice] >= zMax) || ((*xCoord->_mat)[iDevice] >= xMax) ){
			std::cout << "**** ERROR: One of the device is out of bounds ****" << std::endl;
			assert (1==2);
		}
	}
}

void deviceGpu::checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<double2DReg> vel){

	double zIntMax = *max_element(zGridVector.begin(), zGridVector.end());
	double xIntMax = *max_element(xGridVector.begin(), xGridVector.end());
	if ( (zIntMax >= _vel->getHyper()->getAxis(1).n) || (xIntMax >= _vel->getHyper()->getAxis(2).n) ){
		std::cout << "**** ERROR: One of the device is out of bounds ****" << std::endl;
		assert (1==2);
	}
}

void deviceGpu::checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<double2DReg> vel){

	double zIntMax = ozDevice + (nzDevice - 1) * dzDevice;
	double xIntMax = oxDevice + (nxDevice - 1) * dxDevice;
	if ( (zIntMax >= _vel->getHyper()->getAxis(1).n) || (xIntMax >= _vel->getHyper()->getAxis(2).n) ){
		std::cout << "**** ERROR: One of the device is out of bounds ****" << std::endl;
		assert (1==2);
	}
}
