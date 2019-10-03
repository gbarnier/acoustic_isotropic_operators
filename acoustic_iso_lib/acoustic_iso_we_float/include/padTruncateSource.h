#ifndef PADTRUNCATESOURCE_H
#define PADTRUNCATESOURCE_H 1


#include <float1DReg.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include <operator.h>

using namespace SEP;
//! Pad or truncate to/from source space to wavefield on normal grid
/*!
 Used for creating a wavefield (z,x,t) from a source function (t,x,z) or functions
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

#endif
