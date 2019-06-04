#include<PadSource.h>
using namespace waveform;
using namespace giee;

PadSource::PadSource(const std::shared_ptr<float1DReg> model,
  const std::shared_ptr<float3DReg> data, const int s2, const int s3){
  //ensure fast dimension of cube is the same size as the vector
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  //s2 and s3 are index locations of source on axis 1 and 2
  assert(s2 < data->getHyper()->getAxis(2).n); //s2 is in range
  assert(s3 < data->getHyper()->getAxis(3).n); //s3 is in range
  _s2=s2;
  _s3=s3;


  setDomainRange(model,data);
}

void PadSource::forward(const bool add,
  const std::shared_ptr<giee::Vector> model,
  std::shared_ptr<giee::Vector> data){


  assert(checkDomainRange(model,data,true));
  if(!add) data->scale(0.);
  const std::shared_ptr<float1D> m=((std::dynamic_pointer_cast<float1DReg>( model))->_mat);
  std::shared_ptr<float3D> d=((std::dynamic_pointer_cast<float3DReg>( data))->_mat);
  int n1=(std::dynamic_pointer_cast<float3DReg>( data))->getHyper()->getAxis(1).n;

  for(int i1=0; i1<n1; i1++){
    (*d)[_s3][_s2][i1]+=(*m)[i1];
  }

}

void PadSource::adjoint(const bool add,
  std::shared_ptr<giee::Vector> model,
  const std::shared_ptr<giee::Vector> data){

  assert(checkDomainRange(model,data,true));
  if(!add) model->scale(0.);
  std::shared_ptr<float1D> m=((std::dynamic_pointer_cast<float1DReg>( model))->_mat);
  const std::shared_ptr<float3D> d=((std::dynamic_pointer_cast<float3DReg>( data))->_mat);
  int n1=(std::dynamic_pointer_cast<float3DReg>( data))->getHyper()->getAxis(1).n;

  for(int i1=0; i1<n1; i1++){
    (*m)[i1]+=(*d)[_s3][_s2][i1];
  }

}
