#ifndef PADTRUNCATESOURCE_H
#define PADTRUNCATESOURCE_H 1


#include <float1DReg.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include <float4DReg.h>
#include <complex2DReg.h>
#include <complex4DReg.h>
#include <operator.h>
#include <map>
#include <vector>

using namespace SEP;
//! Pad or truncate to/from source space to wavefield on normal grid
/*!
 Used for creating a wavefield (z,x,t) from a source function (t,s) or functions
*/
class padTruncateSource : public Operator<float2DReg, float3DReg> {

	private:

    int _nx_model, _nx_data;
    int _nz_model, _nz_data;
    int _ox_model, _ox_data;
    int _oz_model, _oz_data;
    int _dx_model, _dx_data;
    int _dz_model, _dz_data;
    int _nt;
		std::vector<int> _gridPointIndexUnique; /** Array containing all the positions of the excited grid points - each grid point is unique */

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		padTruncateSource(const std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data, std::vector<int> gridPointIndexUnique);

    //! FWD
		/*!
    * this pads from source to wavefield
    */
    void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data) const;

    //! ADJ
    /*!
    * this truncates from wavefield to source
    */
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Desctructor
    */
		~padTruncateSource(){};

};

//! Pad or truncate to/from source space to wavefield on normal grid over mutliple seismic experiments.
/*!
 Used for creating a wavefield (z,x,t,s) from a source function (t,s) or functions
*/
class padTruncateSource_mutli_exp : public Operator<float2DReg, float4DReg> {

	private:

    int _nx_model, _nx_data;
    int _nz_model, _nz_data;
    int _ox_model, _ox_data;
    int _oz_model, _oz_data;
    int _dx_model, _dx_data;
    int _dz_model, _dz_data;
    int _nt;
		std::vector<std::vector<int>> _gridPointIndexUnique_byExperiment; /** Array containing all the positions of the excited grid points - each grid point is unique */
		std::vector<std::map<int, int>> _indexMaps;  /** vector of index maps. one index map per experiment*/

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		padTruncateSource_mutli_exp(const std::shared_ptr<float2DReg> model, const std::shared_ptr<float4DReg> data, std::vector<std::vector<int>> gridPointIndexUnique_byExperiment, std::vector<std::map<int,int>> indexMaps);  /** vector of index maps. one index map per experiment);

    //! FWD
		/*!
    * this pads from source to wavefield
    */
    void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float4DReg> data) const;

    //! ADJ
    /*!
    * this truncates from wavefield to source
    */
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float4DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Desctructor
    */
		~padTruncateSource_mutli_exp(){};

};

//! Pad or truncate to/from source space to wavefield on normal grid over mutliple seismic experiments where wavefield is complex number
/*!
 Used for creating a wavefield (z,x,t,s) from a source function (t,s) or functions
*/
class padTruncateSource_mutli_exp_complex : public Operator<complex2DReg, complex4DReg> {

	private:

    int _nx_model, _nx_data;
    int _nz_model, _nz_data;
    int _ox_model, _ox_data;
    int _oz_model, _oz_data;
    int _dx_model, _dx_data;
    int _dz_model, _dz_data;
    int _nt;
		std::vector<std::vector<int>> _gridPointIndexUnique_byExperiment; /** Array containing all the positions of the excited grid points - each grid point is unique */
		std::vector<std::map<int, int>> _indexMaps;  /** vector of index maps. one index map per experiment*/

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		padTruncateSource_mutli_exp_complex(const std::shared_ptr<complex2DReg> model, const std::shared_ptr<complex4DReg> data, std::vector<std::vector<int>> gridPointIndexUnique_byExperiment, std::vector<std::map<int,int>> indexMaps);  /** vector of index maps. one index map per experiment);

    //! FWD
		/*!
    * this pads from source to wavefield
    */
    void forward(const bool add, const std::shared_ptr<complex2DReg> model, std::shared_ptr<complex4DReg> data) const;

    //! ADJ
    /*!
    * this truncates from wavefield to source
    */
		void adjoint(const bool add, std::shared_ptr<complex2DReg> model, const std::shared_ptr<complex4DReg> data) const;


		bool dotTest(const bool verbose = false, const float maxError = .00001) const{
		 std::cerr << "cpp dot test not implemented.\n";
	 }
		//! Desctructor
    /*!
    * A more elaborate description of Desctructor
    */
		~padTruncateSource_mutli_exp_complex(){};

};

#endif
