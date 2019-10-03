#ifndef WAVE_EQUATION_ELASTIC_H
#define WAVE_EQUATION_ELASTIC_H 1

#include <float3DReg.h>
#include <float2DReg.h>
#include <float2DReg.h>
#include <operator.h>
#include "waveEquationAcousticGpuFunctions.h"
#include "fdParamAcousticWaveEquation.h"

using namespace SEP;
//! Apply the elastic wave equation to a wavefield
/*!

*/
class waveEquationAcousticGpu : public Operator<SEP::float3DReg, SEP::float3DReg> {

  	private:
  		std::shared_ptr<fdParamAcousticWaveEquation> _fdParamAcoustic;
      int _info;
      int _nGpu,_iGpuAlloc;
      std::vector<int> _gpuList;
      std::vector<int> _firstTimeSamplePerGpu;
      std::vector<int> _lastTimeSamplePerGpu;
      std::shared_ptr<paramObj> _par;


  	public:
      //! Constructor.
  		/*!
      * Overloaded constructors from operator
      */
  		waveEquationAcousticGpu(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data, std::shared_ptr<SEP::float2DReg> slsq, std::shared_ptr<paramObj> par);

      //! FWD
  		/*!
      *
      */
      void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;

      //! ADJ
      /*!
      *
      */
  		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;


  		//! Desctructor
      /*!
      * A more elaborate description of Desctructor
      */

      bool checkGpuMemLimits(float byteLimits=15);

      void createGpuIdList();

      void createGpuSamplesList();

  		~waveEquationAcousticGpu(){};
};

#endif

