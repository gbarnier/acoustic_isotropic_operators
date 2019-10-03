#include <SphericalSpreadingScale.h>
#include <algorithm>
using namespace SEP;

SphericalSpreadingScale::SphericalSpreadingScale(const std::shared_ptr<SEP::float2DReg>model,const std::shared_ptr<SEP::float2DReg>data,
                  const std::shared_ptr<float1DReg> zCoordSou, const std::shared_ptr<float1DReg> xCoordSou,
		  float t_pow,float vel){
  // data[2][1] - first dimension is z, second/
  // dimenstion is x. 
  //
  _n1 = data->getHyper()->getAxis(1).n;
  _n2 = data->getHyper()->getAxis(2).n;
  _o1 = data->getHyper()->getAxis(1).o;
  _o2 = data->getHyper()->getAxis(2).o;
  _d1 = data->getHyper()->getAxis(1).d;
  _d2 = data->getHyper()->getAxis(2).d;
  _vel=vel;
  _tpow=t_pow;

  // model and data should have same axis 
  assert(_n1 == model->getHyper()->getAxis(1).n);
  assert(_n2 == model->getHyper()->getAxis(2).n);
  assert(_d1 == model->getHyper()->getAxis(1).d);
  assert(_d2 == model->getHyper()->getAxis(2).d);
  assert(_o1 == model->getHyper()->getAxis(1).o);
  assert(_o2 == model->getHyper()->getAxis(2).o);

  _scale.reset(new float2DReg(_n1,_n2));
  _scale->set(1.0);
 
  int num_sources= xCoordSou->getHyper()->getAxis(1).n;

  //make scale expanding from sources
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
        //calculate min distance to any source
	float i_x = _o2 + (float)i2 * _d2;
	float i_z = _o1 + (float)i1 * _d1;
        float min_dist = -1;
        //loop over all sources 
	for(int i_s = 0; i_s< num_sources; i_s++){
		float is_x = (*xCoordSou->_mat)[i_s];	
		float is_z = (*zCoordSou->_mat)[i_s];	
		float cur_dist = pow((is_x - i_x)*(is_x - i_x) + (is_z - i_z)*(is_z - i_z),0.5);
		if(cur_dist < min_dist || min_dist < 0) min_dist = cur_dist;
	}
        //calculate scale 
        float tt = min_dist/_vel;
	float eps=0.0001;
        (*_scale->_mat)[i2][i1] = pow(tt+eps,_tpow);
	//std::cerr << "tt: " << tt << std::endl;
	//std::cerr << "tpow: " << _tpow << std::endl;
	//std::cerr << "scale: " << (*_scale->_mat)[i2][i1] << std::endl;
      }
    }
  

  // set domain and range
  setDomainRange(model, data);

}

void SphericalSpreadingScale::forward(const bool                         add,
                          const std::shared_ptr<SEP::float2DReg>model,
                          std::shared_ptr<SEP::float2DReg>      data) const{

  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  std::shared_ptr<float2D> d = data->_mat;
  const std::shared_ptr<float2D> m = model->_mat;

  #pragma omp parallel for collapse(2)
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	(*d)[i2][i1] += (*_scale->_mat)[i2][i1] * (*m)[i2][i1];
  }}
}

void SphericalSpreadingScale::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float2DReg>      model,
                          const std::shared_ptr<SEP::float2DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m = model->_mat;

  const std::shared_ptr<float2D> d = data->_mat;

  #pragma omp parallel for collapse(2)
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	(*m)[i2][i1] += (*_scale->_mat)[i2][i1] * (*d)[i2][i1];
  }}
}
SphericalSpreadingScale_Wfld::SphericalSpreadingScale_Wfld(const std::shared_ptr<SEP::float2DReg>model,const std::shared_ptr<SEP::float2DReg>data,
                  const std::shared_ptr<float3DReg> wfld){
  // data[2][1] - first dimension is z, second/
  // dimenstion is x. 
  //
  _n1 = data->getHyper()->getAxis(1).n;
  _n2 = data->getHyper()->getAxis(2).n;
  _n3 = wfld->getHyper()->getAxis(3).n;

  // model and data should have same axis 
  assert(_n1 == model->getHyper()->getAxis(1).n);
  assert(_n1 == wfld->getHyper()->getAxis(1).n);
  assert(_n2 == model->getHyper()->getAxis(2).n);
  assert(_n2 == wfld->getHyper()->getAxis(2).n);

  _scale.reset(new float2DReg(_n1,_n2));
  _scale->set(0.0);
 
  //stack over time
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	(*_scale->_mat)[i2][i1] += std::abs((*wfld->_mat)[i3][i2][i1]);
      }
    }
  } 
  for (int i2 = 0; i2 < _n2; i2++) {
    for (int i1 = 0; i1 < _n1; i1++) {
      if((*_scale->_mat)[i2][i1] >= 0.00000001){
        (*_scale->_mat)[i2][i1] = 1/(*_scale->_mat)[i2][i1];
      }
    }
  }

  // set domain and range
  setDomainRange(model, data);

}

void SphericalSpreadingScale_Wfld::forward(const bool                         add,
                          const std::shared_ptr<SEP::float2DReg>model,
                          std::shared_ptr<SEP::float2DReg>      data) const{

  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  std::shared_ptr<float2D> d = data->_mat;
  const std::shared_ptr<float2D> m = model->_mat;

  #pragma omp parallel for collapse(2)
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	(*d)[i2][i1] += (*_scale->_mat)[i2][i1] * (*m)[i2][i1];
  }}
}

void SphericalSpreadingScale_Wfld::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float2DReg>      model,
                          const std::shared_ptr<SEP::float2DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m = model->_mat;

  const std::shared_ptr<float2D> d = data->_mat;

  #pragma omp parallel for collapse(2)
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	(*m)[i2][i1] += (*_scale->_mat)[i2][i1] * (*d)[i2][i1];
  }}
}
