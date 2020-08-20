#include <CausalMask.h>
#include <math.h>


CausalMask::CausalMask(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  float vmax,
  float tmin,
  int source_ix,
  int source_iz)
{
  // model and data have the same dimensions
  int n1, n2, n3;

  n1 = model->getHyper()->getAxis(1).n;
  n2 = model->getHyper()->getAxis(2).n;
  n3 = model->getHyper()->getAxis(3).n;
  assert(n1 == data->getHyper()->getAxis(1).n);
  assert(n2 == data->getHyper()->getAxis(2).n);
  assert(n3 == data->getHyper()->getAxis(3).n);

  _velMax= vmax;
  _tmin = tmin;
  int _tmin_index= (_tmin-data->getHyper()->getAxis(3).o)/data->getHyper()->getAxis(3).d;
  _fx_source = source_ix*data->getHyper()->getAxis(2).d+data->getHyper()->getAxis(2).o; 
  _fz_source = source_iz*data->getHyper()->getAxis(1).d+data->getHyper()->getAxis(1).o; 
  // set domain and range
  setDomainRange(model, data);

  _mask.reset(new SEP::float3DReg(data->getHyper()->getAxis(1).n,
                                    data->getHyper()->getAxis(2).n,
                                    data->getHyper()->getAxis(3).n));
_mask->set(1.0);

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
	int distToEdge = std::min(std::min(i2,i1),std::min(n1-i1-1,n2-i2-1));
	if(distToEdge < _spongeWidth){
	  float coeff = exp(-(_lambda*_lambda)*(_spongeWidth-distToEdge)*(_spongeWidth-distToEdge));
	  (*_mask->_mat)[i3][i2][i1] *= coeff; 
	}
	float fx = i2*data->getHyper()->getAxis(2).d+data->getHyper()->getAxis(2).o;
	float fz = i1*data->getHyper()->getAxis(1).d+data->getHyper()->getAxis(1).o;
	float distToSource = pow(pow(fx-_fx_source,2)+pow(fz-_fz_source,2),0.5); 
	float timeToSource = distToSource/_velMax;
	int itOffset = _tmin_index + timeToSource/data->getHyper()->getAxis(3).d;
	if(i3<itOffset){
	float coeff = exp(-(_lambda*_lambda)*(itOffset-i3)*(itOffset-i3));
	  (*_mask->_mat)[i3][i2][i1] *= coeff; 
	}
	if(i2==105 && i1==105){
		std::cerr<< "it : " << i3 << " distToSource: " << distToSource << " timeToSource: " << timeToSource << " itOffset: " << itOffset << " coeff: " << (*_mask->_mat)[i3][i2][i1] << std::endl;
	}
      }
    }
  }
}

// forward
void CausalMask::forward(const bool                         add,
                     const std::shared_ptr<SEP::float3DReg>model,
                     std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*d)[i3][i2][i1] += (*m)[i3][i2][i1] * (*_mask->_mat)[i3][i2][i1];
      }
    }
  }
}

// adjoint
void CausalMask::adjoint(const bool                         add,
                     std::shared_ptr<SEP::float3DReg>      model,
                     const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2][i1] += (*d)[i3][i2][i1] * (*_mask->_mat)[i3][i2][i1];
      }
    }
  }
}

