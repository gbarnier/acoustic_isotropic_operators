#ifndef DEVICE_GPU_H
#define DEVICE_GPU_H 1

#include "ioModes.h"
#include "float1DReg.h"
#include "float2DReg.h"
#include "operator.h"
#include <vector>

using namespace SEP;

class deviceGpu : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

		/* Spatial interpolation */
		std::shared_ptr<float2DReg> _vel;
		std::shared_ptr<float1DReg> _zCoord, _xCoord;
		std::vector<int> _gridPointIndexUnique; // Array containing all the positions of the excited grid points - each grid point is unique
		std::map<int, int> _indexMap;
		std::map<int, int>::iterator _iteratorIndexMap;
		float *_weight;
		int *_gridPointIndex;
		int _nDeviceIrreg, _nDeviceReg, _nt, _nz;

	public:

		/* Overloaded constructors */
		deviceGpu(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float2DReg> vel, int &nt);
		deviceGpu(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<float2DReg> vel, int &nt);
		deviceGpu(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<float2DReg> vel, int &nt);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<float2DReg> signalReg, std::shared_ptr<float2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> signalReg, const std::shared_ptr<float2DReg> signalIrreg) const;

		// Destructor
		~deviceGpu(){};

		// Other functions
  		void checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float2DReg> vel); // For constructor #1
  		void checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<float2DReg> vel); // For constructor #2
		void checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<float2DReg> vel); // For constructor #3
  		void convertIrregToReg();

  		int *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
  		int *getRegPos(){ return _gridPointIndex; }
  		int getNt(){ return _nt; }
  		int getNDeviceReg(){ return _nDeviceReg; }
  		int getNDeviceIrreg(){ return _nDeviceIrreg; }
  		float * getWeights() { return _weight; }
  		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
};

#endif
