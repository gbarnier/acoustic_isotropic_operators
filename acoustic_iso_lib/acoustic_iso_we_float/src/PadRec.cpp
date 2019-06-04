#include<PadRec.h>
using namespace waveform;
using namespace giee;

PadRec::PadRec(const std::shared_ptr<float2DReg> model,
  const std::shared_ptr<float3DReg> data, const int s2){

  //ensure 1st dimension (time) of cube are the same as slice
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);

  //ensure more x locations (2nd and 3rd dimensions) in data than in model
  assert(model->getHyper()->getAxis(2).n <= data->getHyper()->getAxis(3).n);

  //slice loaction s2 should be within bounds of cube's 1st dimension
  assert(s2 < data->getHyper()->getAxis(2).n); //s2 is in range
  _s2=s2;
  _nxm=model->getHyper()->getAxis(2).n;
  _oxm=model->getHyper()->getAxis(2).o;
  _dxm=model->getHyper()->getAxis(2).d;
  _nxd=data->getHyper()->getAxis(3).n;
  _oxd=data->getHyper()->getAxis(3).o;
  _dxd=data->getHyper()->getAxis(3).d;

  setDomainRange(model,data);
}

void PadRec::forward(const bool add,
  const std::shared_ptr<giee::Vector> model,
  std::shared_ptr<giee::Vector> data){


  assert(checkDomainRange(model,data,true));
  if(!add) data->scale(0.);
  const std::shared_ptr<float2D> m=((std::dynamic_pointer_cast<float2DReg>( model))->_mat);
  std::shared_ptr<float3D> d=((std::dynamic_pointer_cast<float3DReg>( data))->_mat);
  int n1=(std::dynamic_pointer_cast<float3DReg>( data))->getHyper()->getAxis(1).n;
  int n3=(std::dynamic_pointer_cast<float3DReg>( data))->getHyper()->getAxis(3).n;

  for(int i3=0; i3<n3; i3++){
    float ixf = ((i3*_dxd+_oxd)-_oxm);
    int ix = ixf/_dxm;
    if(fmod(ixf,_dxm)<.00001 && ix>=0 && ix < _nxm){
      for(int i1=0; i1<n1; i1++){
        (*d)[i3][_s2][i1]+=(*m)[ix][i1];
      }
    }
  }

}

void PadRec::adjoint(const bool add,
  std::shared_ptr<giee::Vector> model,
  const std::shared_ptr<giee::Vector> data){

  assert(checkDomainRange(model,data,true));
  if(!add) model->scale(0.);
  std::shared_ptr<float2D> m=((std::dynamic_pointer_cast<float2DReg>( model))->_mat);
  const std::shared_ptr<float3D> d=((std::dynamic_pointer_cast<float3DReg>( data))->_mat);
  int n1=(std::dynamic_pointer_cast<float3DReg>( data))->getHyper()->getAxis(1).n;
  int n3=(std::dynamic_pointer_cast<float3DReg>( data))->getHyper()->getAxis(3).n;

  for(int i3=0; i3<n3; i3++){
    float ixf = ((i3*_dxd+_oxd)-_oxm);
    int ix = ixf/_dxm;
    if(fmod(ixf,_dxm)<.00001 && ix>=0 && ix < _nxm){
      for(int i1=0; i1<n1; i1++){
        (*m)[ix][i1]+=(*d)[i3][_s2][i1];
      }
    }
  }

}
