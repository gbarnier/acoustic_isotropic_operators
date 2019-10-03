#include <GF.h>
#include <algorithm>
using namespace SEP;

GF::GF(const std::shared_ptr<SEP::float3DReg>model,const std::shared_ptr<SEP::float3DReg>data,
                  const std::shared_ptr<float1DReg> zCoordSou, const std::shared_ptr<float1DReg> xCoordSou,
		  float t_start,float vel){
  // data[3][2][1] - first dimension is time, second dimension is z, thir/
  // dimenstion is x. The x and z locaitons to pull/push from/to model are
  // gathered from data hypercube. This allows the data to be made of traces
  // with regular sampling in the model space.

  _n1 = data->getHyper()->getAxis(1).n;
  _n2 = data->getHyper()->getAxis(2).n;
  _n3 = data->getHyper()->getAxis(3).n;
  _o1 = data->getHyper()->getAxis(1).o;
  _o2 = data->getHyper()->getAxis(2).o;
  _o3 = data->getHyper()->getAxis(3).o;
  _d1 = data->getHyper()->getAxis(1).d;
  _d2 = data->getHyper()->getAxis(2).d;
  _d3 = data->getHyper()->getAxis(3).d;
  _vel=vel;

  // model and data should have same axis 
  assert(_n1 == model->getHyper()->getAxis(1).n);
  assert(_n2 == model->getHyper()->getAxis(2).n);
  assert(_n3 == model->getHyper()->getAxis(3).n);
  assert(_d1 == model->getHyper()->getAxis(1).d);
  assert(_d2 == model->getHyper()->getAxis(2).d);
  assert(_d3 == model->getHyper()->getAxis(3).d);
  assert(_o1 == model->getHyper()->getAxis(1).o);
  assert(_o2 == model->getHyper()->getAxis(2).o);
  assert(_o3 == model->getHyper()->getAxis(3).o);

  _scale.reset(new float3DReg(_n1,_n2,_n3));
  _scale->set(0.0);
  _scale2D.reset(new float2DReg(_n1,_n2));
  _scale2D->set(0.0);
 
  int num_sources= xCoordSou->getHyper()->getAxis(1).n;

  //make scale expanding from sources
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
        //calculate min distance to any source
	float i_x = _o2 + (float)i2 * _d2;
	float i_z = _o1 + (float)i1 * _d1;
        float min_radius = -1;
        //loop over all sources 
	for(int i_s = 0; i_s< num_sources; i_s++){
		float is_x = (*xCoordSou->_mat)[i_s];	
		float is_z = (*zCoordSou->_mat)[i_s];	
		float cur_radius = pow((is_x - i_x)*(is_x - i_x) + (is_z - i_z)*(is_z - i_z),0.5);
		if(cur_radius < min_radius || min_radius < 0) min_radius = cur_radius;
	}
        //calculate min_tt
        for (int i3 = 0; i3 < _n3; i3++) {
          float t_actual = _o3 + i3*_d3 - t_start;
          if( t_actual - min_radius/_vel <= 0.000001){
            (*_scale->_mat)[i3][i2][i1] = 0;
          }
	  else{
	    if(t_actual*t_actual-min_radius*min_radius/_vel/_vel <=0){
              std::cerr << "sqrt < 0 at (" << i3 << "," << i2 << "," << i1 << ")= " << t_actual*t_actual-min_radius*min_radius/_vel/_vel << std::endl; 
	    }
            (*_scale->_mat)[i3][i2][i1] = 1/(2*M_PI)/std::sqrt(t_actual*t_actual-min_radius*min_radius/_vel/_vel);
          }
	}
      }
    }
  
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	(*_scale2D->_mat)[i2][i1] += (*_scale->_mat)[i3][i2][i1];
  }}}
 // for (int i2 = 0; i2 < _n2; i2++) {
 //   for (int i1 = 0; i1 < _n1; i1++) {
 //     (*_scale2D->_mat)[i2][i1] = 1/(*_scale2D->_mat)[i2][i1];
 // }}

  // set domain and range
  setDomainRange(model, data);

}

void GF::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const{

  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  std::shared_ptr<float3D> d = data->_mat;
  const std::shared_ptr<float3D> m = model->_mat;

  #pragma omp parallel for collapse(3)
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	//(*d)[i3][i2][i1] += (*_scale->_mat)[i3][i2][i1] * (*m)[i3][i2][i1];
	(*d)[i3][i2][i1] += (*_scale2D->_mat)[i2][i1] * (*m)[i3][i2][i1];
  }}}

}

void GF::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m = model->_mat;

  const std::shared_ptr<float3D> d = data->_mat;

  #pragma omp parallel for collapse(3)
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	//(*m)[i3][i2][i1] += (*_scale->_mat)[i3][i2][i1] * (*d)[i3][i2][i1];
	(*m)[i3][i2][i1] += (*_scale2D->_mat)[i2][i1] * (*d)[i3][i2][i1];
  }}}
}
 
