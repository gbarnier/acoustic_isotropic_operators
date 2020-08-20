#include "waveEquationAcousticGpu.h"
#include <cstring>
#include <cassert>
waveEquationAcousticGpu::waveEquationAcousticGpu(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data, std::shared_ptr<SEP::float2DReg> slsq, std::shared_ptr<paramObj> par){

  _par = par;

  _info = _par->getInt("info", 0);

  //check domain and range
  assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n);
  assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n);
  assert(data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(3).n);

  //fd check
  _fdParamAcoustic = std::make_shared<fdParamAcousticWaveEquation>(slsq, _par);
  if ( _info == 1 ){
    _fdParamAcoustic->getInfo();
  }

  //gpu check
  int _deviceNumberInfo = _par->getInt("deviceNumberInfo", 0);
  _nGpu =1;
  assert(getGpuInfo(_nGpu, _info, _deviceNumberInfo)); // Get info on GPU cluster and check that there are enough available GPUs

  //create gpu list
  createGpuIdList();
  float memAvailGB = getTotalGlobalMem(_nGpu, _info, _deviceNumberInfo);
  //std::cerr << "checkGpuMemLimits(memAvailGB):" << checkGpuMemLimits(memAvailGB) << std::endl;
  assert(checkGpuMemLimits(memAvailGB));
  //static_assert(1==2,"this is a test");

  assert(data->getHyper()->getAxis(1).n == _fdParamAcoustic->_nz);
  assert(data->getHyper()->getAxis(2).n == _fdParamAcoustic->_nx);
  assert(data->getHyper()->getAxis(1).d == _fdParamAcoustic->_dz);
  assert(data->getHyper()->getAxis(2).d == _fdParamAcoustic->_dx);
  assert(data->getHyper()->getAxis(1).o == _fdParamAcoustic->_oz);
  assert(data->getHyper()->getAxis(2).o == _fdParamAcoustic->_ox);

  //initialize on gpus
  for(int iGpu=0; iGpu<_nGpu; iGpu++){
    initWaveEquationAcousticGpu(_fdParamAcoustic->_dz, _fdParamAcoustic->_dx, _fdParamAcoustic->_nz, _fdParamAcoustic->_nx, _fdParamAcoustic->_nts, _fdParamAcoustic->_dts, _fdParamAcoustic->_minPad, _fdParamAcoustic->_blockSize, _nGpu, _gpuList[iGpu], _iGpuAlloc);

    //allocate on gpu

    allocateWaveEquationAcousticGpu(_fdParamAcoustic->_slsqDt2,_fdParamAcoustic->_cosDamp
                          ,iGpu, _gpuList[iGpu], _firstTimeSamplePerGpu[iGpu], _lastTimeSamplePerGpu[iGpu]);
  }

  setDomainRange(model,data);
}

void waveEquationAcousticGpu::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{
  assert(checkDomainRange(model,data));
  //if(!add) data->scale(0.);
  if(!add) std::memset(data->getVals(), 0, sizeof data->getVals());

  int nz = _fdParamAcoustic->_nz;
  int nx = _fdParamAcoustic->_nx;
  //set fat region to zero
  #pragma omp parallel for collapse(3)
  for(int it=0;it<_fdParamAcoustic->_nts;it++){
      for(int ix=0;ix<nx;ix++){
        for(int iz=0;iz<_fdParamAcoustic->_fat;iz++){
          (*model->_mat)[it][ix][iz]=0;
          (*model->_mat)[it][ix][nz-iz-1]=0;
        }
      }
  }
  #pragma omp parallel for collapse(3)
  for(int it=0;it<_fdParamAcoustic->_nts;it++){
      for(int ix=0;ix<_fdParamAcoustic->_fat;ix++){
        for(int iz=_fdParamAcoustic->_fat;iz<nz-_fdParamAcoustic->_fat;iz++){
          (*model->_mat)[it][ix][iz]=0;
          (*model->_mat)[it][nx-ix-1][iz]=0;
        }
      }
  }

  //call fwd gpu function
  #pragma omp parallel for num_threads(_nGpu)
  for (int iGpu=0; iGpu<_nGpu; iGpu++){
    waveEquationAcousticFwdGpu(model->getVals(), data->getVals(), iGpu, _gpuList[iGpu], _firstTimeSamplePerGpu[iGpu], _lastTimeSamplePerGpu[iGpu]);
  }

}

void waveEquationAcousticGpu::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const{
  assert(checkDomainRange(model,data));
  //if(!add) model->scale(0.);
  if(!add) std::memset(model->getVals(), 0, sizeof model->getVals());

  int nz = _fdParamAcoustic->_nz;
  int nx = _fdParamAcoustic->_nx;
  //set fat region to zero
  #pragma omp parallel for collapse(3)
  for(int it=0;it<_fdParamAcoustic->_nts;it++){
      for(int ix=0;ix<nx;ix++){
        for(int iz=0;iz<_fdParamAcoustic->_fat;iz++){
          (*data->_mat)[it][ix][iz]=0;
          (*data->_mat)[it][ix][nz-iz-1]=0;
        }
      }
  }
  #pragma omp parallel for collapse(3)
  for(int it=0;it<_fdParamAcoustic->_nts;it++){
      for(int ix=0;ix<_fdParamAcoustic->_fat;ix++){
        for(int iz=_fdParamAcoustic->_fat;iz<nz-_fdParamAcoustic->_fat;iz++){
          (*data->_mat)[it][ix][iz]=0;
          (*data->_mat)[it][nx-ix-1][iz]=0;
        }
      }
  }

  //call adj gpu function
  #pragma omp parallel for num_threads(_nGpu)
  for (int iGpu=0; iGpu<_nGpu; iGpu++){
    waveEquationAcousticAdjGpu(model->getVals(), data->getVals(), iGpu, _gpuList[iGpu], _firstTimeSamplePerGpu[iGpu], _lastTimeSamplePerGpu[iGpu]);
  }
}

bool waveEquationAcousticGpu::checkGpuMemLimits(float gbLimits){


  unsigned long long spaceUsedWflds = (unsigned long long)4*(unsigned long long)_fdParamAcoustic->_nz*(unsigned long long)_fdParamAcoustic->_nx*(unsigned long long)_fdParamAcoustic->_nts*(unsigned long long)2;
  unsigned long long spaceUsedAcousticParams = (unsigned long long)_fdParamAcoustic->_nz*(unsigned long long)_fdParamAcoustic->_nx;
  unsigned long long totalSpaceUsedGB = (spaceUsedWflds+spaceUsedAcousticParams)/((unsigned long long)(1024*1024*1024));

  if(_info==1){
    std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << "************************* GPU USAGE INFO **************************" << std::endl;
		std::cout << "*******************************************************************" << std::endl;

    std::cout << "GPUs selected to use: ";
    for(int i=0; i<_gpuList.size(); i++) std::cout << _gpuList[i] << " ";
    std::cout << "\nAvailable global memory per gpu: " << gbLimits << "[GB]\n";
    std::cout << "Global memory to be used per gpu: " << (float)totalSpaceUsedGB/(float)_nGpu << "[GB]\n";
    std::cerr << std::setprecision(2);
    std::cout << "Percentage of memory used per gpu: " << (float)totalSpaceUsedGB/(float)gbLimits*(float)100/(float)_nGpu << "%\n";
    createGpuSamplesList();
    std::cerr << "*******************************************************************" << std::endl;

  }
  if (totalSpaceUsedGB/gbLimits > 1) return false;
  else return true;
}

void waveEquationAcousticGpu::createGpuIdList(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);
	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR: Please provide a list of GPUs to be used ****" << std::endl; assert(1==2);}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR: Please make sure there are no duplicates in the GPU Id list ****" << std::endl; assert(1==2);
		}
	}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

void waveEquationAcousticGpu::createGpuSamplesList(){

  std::cerr << "Domain Decomposition Info:" << std::endl;
  //properly divide time samples between gpus
  int samplesPerGpu =  _fdParamAcoustic->_nts/_nGpu;
  for(int i=0; i<_nGpu;i++){
    if(i==0){
      _firstTimeSamplePerGpu.push_back(0);
      _lastTimeSamplePerGpu.push_back(samplesPerGpu-1);
    }
    else if(i==_nGpu-1){
      _firstTimeSamplePerGpu.push_back(i*samplesPerGpu-2);
      _lastTimeSamplePerGpu.push_back(_fdParamAcoustic->_nts-1);
    }
    else{
      _firstTimeSamplePerGpu.push_back(i*samplesPerGpu-2);
      _lastTimeSamplePerGpu.push_back((i+1)*samplesPerGpu-1);
    }
    std::cerr << "\t time samples on gpu " << i << ": it=" << _firstTimeSamplePerGpu[i] << "->" << _lastTimeSamplePerGpu[i] << std::endl;
  }




}

