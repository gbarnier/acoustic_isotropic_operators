#ifndef SPACE_INTERP_MULTI_H
#define SPACE_INTERP_MULTI_H 1

#include "float1DReg.h"
#include "float2DReg.h"
#include "complex2DReg.h"
#include "hypercube.h"
#include "operator.h"
#include <map>
#include <vector>
#include <string>

using namespace SEP;
//! This class transforms the data on an irregular space grid (positions of the receivers for example) into data on a regular grid for 5 elastic componenets
/*!

*/
class spaceInterp : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

		/* Spatial interpolation */
		//std::shared_ptr<float2DReg> _elasticPar; /** for dimension checking*/
		std::shared_ptr<SEP::hypercube> _vpParamHypercube;
		std::shared_ptr<float1DReg> _zCoord, _xCoord; /** Detailed description after the member */
		std::vector<int> _gridPointIndexUnique; /** Array containing all the positions of the excited grid points - each grid point is unique */
		std::map<int, int> _indexMap;
		std::map<int, int>::iterator _iteratorIndexMap;
		float *_weight;
		int *_gridPointIndex;
		int _nDeviceIrreg, _nDeviceReg, _nt, _nz, _nFilt,_nFilt2D,_nFiltTotal;
		float _ox,_oz,_dx,_dz;
		std::string _interpMethod;

	public:

		/* Overloaded constructors */
		spaceInterp(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt, std::string interpMethod, int nFilt);
		// spaceInterp(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt);
		// spaceInterp(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<float2DReg> signalReg, std::shared_ptr<float2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> signalReg, const std::shared_ptr<float2DReg> signalIrreg) const;

		// Destructor
		~spaceInterp(){};

		// Other functions
		void checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord); // For constructor #1
		void checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector); // For constructor #2
		void checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice); // For constructor #3
		void convertIrregToReg();

		std::vector<int> getRegPosUniqueVector(){ return _gridPointIndexUnique; }
		int *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
		int *getRegPos(){ return _gridPointIndex; }
		int getNt(){ return _nt; }
		int getNDeviceReg(){ return _nDeviceReg; }
		int getNDeviceIrreg(){ return _nDeviceIrreg; }
		float * getWeights() { return _weight; }
		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
		void getInfo();
		void calcLinearWeights();
		void calcSincWeights();
		void calcGaussWeights();
};

class spaceInterp_multi_exp : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

		/* Spatial interpolation */
		//std::shared_ptr<float2DReg> _elasticPar; /** for dimension checking*/
		std::shared_ptr<SEP::hypercube> _vpParamHypercube;
		std::shared_ptr<float1DReg> _zCoord, _xCoord, _expIndex; /** Detailed description after the member */
		std::vector<std::vector<int>> _gridPointIndexUnique; /** Vector of vectors. Each inner vector containing all the positions of the excited grid points - each grid point is unique - for one experiment */
		std::vector<std::map<int, int>> _indexMaps;  /** vector of index maps. one index map per experiment */
		float *_weight;
		int *_gridPointIndex;
		int _nDeviceIrreg, _nDeviceReg, _nt, _nz, _nFilt,_nFilt2D,_nFiltTotal,_nExp;
		float _ox,_oz,_dx,_dz;
		std::string _interpMethod;

	public:

		/* Overloaded constructors */
		spaceInterp_multi_exp(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float1DReg> _expIndex, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt, std::string interpMethod, int nFilt);
		// spaceInterp(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt);
		// spaceInterp(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<float2DReg> signalReg, std::shared_ptr<float2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> signalReg, const std::shared_ptr<float2DReg> signalIrreg) const;

		// Destructor
		~spaceInterp_multi_exp(){};

		// Other functions
		void checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord); // For constructor #1
		void checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector); // For constructor #2
		void checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice); // For constructor #3
		void convertIrregToReg();

		std::vector<std::vector<int>> getRegPosUniqueVector(){ return _gridPointIndexUnique; }
		std::vector<std::map<int,int>> getIndexMaps(){ return _indexMaps; }
		//int *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
		int *getRegPos(){ return _gridPointIndex; }
		int getNt(){ return _nt; }
		int getNDeviceReg(){ return _nDeviceReg; }
		int getNDeviceIrreg(){ return _nDeviceIrreg; }
		float * getWeights() { return _weight; }
		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
		void getInfo();
		void calcLinearWeights();
		void calcSincWeights();
		void calcGaussWeights();
};

class spaceInterp_multi_exp_complex : public Operator<SEP::complex2DReg, SEP::complex2DReg> {

	private:

		/* Spatial interpolation */
		//std::shared_ptr<complex2DReg> _elasticPar; /** for dimension checking*/
		std::shared_ptr<SEP::hypercube> _vpParamHypercube;
		std::shared_ptr<float1DReg> _zCoord, _xCoord, _expIndex; /** Detailed description after the member */
		std::vector<std::vector<int>> _gridPointIndexUnique; /** Vector of vectors. Each inner vector containing all the positions of the excited grid points - each grid point is unique - for one experiment */
		std::vector<std::map<int, int>> _indexMaps;  /** vector of index maps. one index map per experiment */
		float *_weight;
		int *_gridPointIndex;
		int _nDeviceIrreg, _nDeviceReg, _nt, _nz, _nFilt,_nFilt2D,_nFiltTotal,_nExp;
		float _ox,_oz,_dx,_dz;
		std::string _interpMethod;

	public:

		/* Overloaded constructors */
		spaceInterp_multi_exp_complex(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float1DReg> _expIndex, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt, std::string interpMethod, int nFilt);
		// spaceInterp(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt);
		// spaceInterp(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const std::shared_ptr<SEP::hypercube> vpParamHyper, int &nt);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<complex2DReg> signalReg, std::shared_ptr<complex2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<complex2DReg> signalReg, const std::shared_ptr<complex2DReg> signalIrreg) const;

		// Destructor
		~spaceInterp_multi_exp_complex(){};

		// Other functions
		void checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord); // For constructor #1
		void checkOutOfBounds(const std::vector<int> &zGridVector, const std::vector<int> &xGridVector); // For constructor #2
		void checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice); // For constructor #3
		void convertIrregToReg();

		std::vector<std::vector<int>> getRegPosUniqueVector(){ return _gridPointIndexUnique; }
		std::vector<std::map<int,int>> getIndexMaps(){ return _indexMaps; }
		//int *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
		int *getRegPos(){ return _gridPointIndex; }
		int getNt(){ return _nt; }
		int getNDeviceReg(){ return _nDeviceReg; }
		int getNDeviceIrreg(){ return _nDeviceIrreg; }
		float * getWeights() { return _weight; }
		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
		void getInfo();
		void calcLinearWeights();
		void calcSincWeights();
		void calcGaussWeights();

		bool dotTest(const bool verbose = false, const float maxError = .00001) const{
		 std::cerr << "cpp dot test not implemented.\n";
	 }
};
#endif
