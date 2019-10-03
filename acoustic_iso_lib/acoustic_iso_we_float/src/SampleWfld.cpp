#include <SampleWfld.h>
#include <algorithm>
using namespace SEP;

SampleWfld::SampleWfld(const std::shared_ptr<SEP::float3DReg>model,const std::shared_ptr<SEP::float3DReg>data,
                  const std::shared_ptr<float1DReg> zCoordSou, const std::shared_ptr<float1DReg> xCoordSou,
		  const std::shared_ptr<float1DReg> zCoordRec, const std::shared_ptr<float1DReg> xCoordRec,float t_start,float max_vel){
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
  _max_vel=max_vel;

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

  _mask.reset(new float3DReg(_n1,_n2,_n3));
  _mask->set(0.0);
 
  int num_sources= xCoordSou->getHyper()->getAxis(1).n;
  int num_recs = xCoordRec->getHyper()->getAxis(1).n;

std::cerr << "here0\n";
  //set rec grid points to 1 for all time
  for(int i_r = 0; i_r< num_recs; i_r++){
    float ir_x = (*xCoordRec->_mat)[i_r];	
    float ir_z = (*zCoordRec->_mat)[i_r];	
    float ir_x_temp = (ir_x - _o2)/_d2; 
    float ir_z_temp = (ir_z - _o1)/_d1; 
    int ix = ir_x_temp; // x-coordinate on regular grid
    int iz = ir_z_temp; // z-coordinate on regular grid
    //std::cerr << "\nirec: " << i_r << "\nir_x: " << ir_x << " ix: " << ix << "\nir_z: " << ir_z << " iz: " << iz << std::endl;
    for (int i3 = 0; i3 < _n3; i3++) {
      //std::cerr << "i3: " << i3 << std::endl;
      (*_mask->_mat)[i3][ix][iz] = 1;   
    }
  }
std::cerr << "here1\n";

  //make mask expanding from sources
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
        //calculate min_tt
        float min_tt = min_dist/_max_vel;
        for (int i3 = 0; i3 < _n3; i3++) {
          float t_actual = _o3 + i3*_d3;
          if( t_actual-t_start < min_tt){
            (*_mask->_mat)[i3][i2][i1] = 1;
          }
	}
      }
    }
  

  // set domain and range
  setDomainRange(model, data);

}

void SampleWfld::forward(const bool                         add,
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
	(*d)[i3][i2][i1] += (*_mask->_mat)[i3][i2][i1] * (*m)[i3][i2][i1];
  }}}
//  #pragma omp parallel for collapse(3)
//  for (int i3 = 0; i3 < _n3m; i3++) {
//    for (int i2d = 0; i2d < _n2d; i2d++) {
//      for (int i1d = 0; i1d < _n1d; i1d++) {
//	int ix = i2d*_j2d+_o2d_int;
//	int iz = i1d*_j1d+_o1d_int;
//        (*d)[i3][ix][iz] += (*m)[i3][ix][iz];
//
//  }}}

}

void SampleWfld::adjoint(const bool                         add,
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
	(*m)[i3][i2][i1] += (*_mask->_mat)[i3][i2][i1] * (*d)[i3][i2][i1];
  }}}
//  #pragma omp parallel for collapse(3)
//  for (int i3 = 0; i3 < _n3m; i3++) {
//    for (int i2d = 0; i2d < _n2d; i2d++) {
//      for (int i1d = 0; i1d < _n1d; i1d++) {
//	int ix = i2d*_j2d+_o2d_int;
//	int iz = i1d*_j1d+_o1d_int;
//        (*m)[i3][ix][iz] += (*d)[i3][ix][iz];
//
//  }}}
 // for (int i3 = 0; i3 < _n3m; i3++) {
 //   for (int i2 = 0; i2 < _n2m; i2++) {
 //     for (int i1 = 0; i1 < _n1m; i1++) {
 //   	float fx = _o2m + (float)i2 * _d2m;
 //   	float ixd = ((float)fx-(float)_o2d)/(float)_d2d;

 //   	float fz = _o1m + (float)i1 * _d1m;
 //   	float izd = ((float)fz-(float)_o1d)/(float)_d1d;

 //       if(i3 < 10) (*m)[i3][i2][i1] += (*d)[i3][i2][i1];
 //       else if( ixd>=0 && std::abs(std::fmod(ixd,0.99999999)) <0.0000001 && izd>=0 && std::abs(std::fmod(izd,0.99999999)) <0.0000001){
 //       	 (*m)[i3][i2][i1] += (*d)[i3][i2][i1];
 //       }
 //       else  (*m)[i3][i2][i1] += 0.0;
 //     }
 //   }
 // }
 

}
SampleWfldTime::SampleWfldTime(const std::shared_ptr<SEP::float3DReg>model,
                                 const std::shared_ptr<SEP::float3DReg>data,int tmin)
{

  // model and data should have same time axis (3)size
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(1).o == data->getHyper()->getAxis(1).o);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(2).o == data->getHyper()->getAxis(2).o);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(3).o == data->getHyper()->getAxis(3).o);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  _tmin=tmin;
  setDomainRange(model, data);
  }

void SampleWfldTime::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const{

  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  std::shared_ptr<float3D> d = data->_mat;
  const std::shared_ptr<float3D> m = model->_mat;

  #pragma omp parallel for collapse(3)
  for (int i3 = 0; i3 < model->getHyper()->getAxis(3).n; i3++) {
    for (int i2 = 0; i2 < model->getHyper()->getAxis(2).n; i2++) {
      for (int i1 = 0; i1 < model->getHyper()->getAxis(1).n; i1++) {
        if(i3 < _tmin) (*d)[i3][i2][i1] += (*m)[i3][i2][i1];
        else  (*d)[i3][i2][i1] += 0.0;
  }}}

}

void SampleWfldTime::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m = model->_mat;

  const std::shared_ptr<float3D> d = data->_mat;

  #pragma omp parallel for collapse(3)
  for (int i3 = 0; i3 < model->getHyper()->getAxis(3).n; i3++) {
    for (int i2 = 0; i2 < model->getHyper()->getAxis(2).n; i2++) {
      for (int i1 = 0; i1 < model->getHyper()->getAxis(1).n; i1++) {
        if(i3 < _tmin) (*m)[i3][i2][i1] += (*d)[i3][i2][i1];
        else  (*m)[i3][i2][i1] += 0.0;
  }}}
 

}
