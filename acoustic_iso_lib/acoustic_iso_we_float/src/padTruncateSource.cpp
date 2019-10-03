#include <padTruncateSource.h>
#include <algorithm>
using namespace SEP;


padTruncateSource::padTruncateSource(const std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data,std::vector<int> gridPointIndexUnique){
  // assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(4).n); //z axis
  // assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(3).n); //x axis
  assert(data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(2).n); //time axis

  //ensure number of sources/rec to inject is equal to the number of gridpoints provided
  assert(model->getHyper()->getAxis(1).n == gridPointIndexUnique.size());

  _nz_data = data->getHyper()->getAxis(1).n;
  _nx_data = data->getHyper()->getAxis(2).n;
  _oz_data = data->getHyper()->getAxis(1).o;
  _ox_data = data->getHyper()->getAxis(2).o;
  _dz_data = data->getHyper()->getAxis(1).d;
  _dx_data= data->getHyper()->getAxis(2).d;
  _nt = data->getHyper()->getAxis(3).n;

  _gridPointIndexUnique = gridPointIndexUnique;
  //
  // _nz_model = model->getHyper()->getAxis(4).n;
  // _nx_model = model->getHyper()->getAxis(3).n;
  // _oz_model = model->getHyper()->getAxis(4).o;
  // _ox_model = model->getHyper()->getAxis(3).o;
  // _dz_model = model->getHyper()->getAxis(4).d;
  // _dx_model = model->getHyper()->getAxis(3).d;

  // float z_data_max = _oz_data + (_nz_data-1)*_dz_data;
  // float x_data_max = _ox_data + (_nx_data-1)*_dx_data;
  //
  // //ensure the x and z source locations align with the wfld x and z
  // for(int ix_model = 0; ix_model < _nx_model; ix_model++){
  //   for(int iz_model = 0; iz_model < _nz_model; iz_model++){
  //     //calc model x and z float
  //     float z_model = _oz_model + iz_model * _dz_model;
  //     float x_model = _ox_model + ix_model * _dx_model;
  //
  //     //assert z and x in data range
  //     assert(z_model<=z_data_max);
  //     assert(z_model>=_oz_data);
  //     assert(x_model<=x_data_max);
  //     assert(x_model>=_ox_data);
  //
  //     float z_rem = abs((z_model - _oz_data)/_dz_data - int((z_model - _oz_data)/_dz_data));
  //     float x_rem = abs((x_model - _ox_data)/_dx_data - int((x_model - _ox_data)/_dx_data));
  //
  //     //assert model x and z fall on data grid
  //     assert(z_rem < 0.0000001);
  //     assert(x_rem < 0.0000001);
  //
  //   }
  // }


  setDomainRange(model,data);
}

//! FWD
/*!
* this pads from source to wavefield
*/
void padTruncateSource::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) data->scale(0.);

  //for each device add to correct location in wavefield
  #pragma omp parallel for
  for(int id = 0; id < _gridPointIndexUnique.size(); id++){
      int gridPoint = _gridPointIndexUnique[id];
      int ix = (int)gridPoint/_nz_data;
      int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
      //#pragma omp parallel for
      for(int it = 0; it < _nt; it++){
        // (*data->_mat)[0][it][x_index_data][z_index_data] += (*model->_mat)[iz_model][ix_model][0][it];
        (*data->_mat)[it][ix][iz] += (*model->_mat)[it][id];
      }
  }

}

//! ADJ
/*!
* this truncates from wavefield to source
*/
void padTruncateSource::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) model->scale(0.);

  //for each device add to correct location in wavefield
  #pragma omp parallel for
  for(int id = 0; id < _gridPointIndexUnique.size(); id++){
      int gridPoint = _gridPointIndexUnique[id];
      int ix = (int)gridPoint/_nz_data;
      int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
      for(int it = 0; it < _nt; it++){
        // (*data->_mat)[0][it][x_index_data][z_index_data] += (*model->_mat)[iz_model][ix_model][0][it];
        (*model->_mat)[it][id] += (*data->_mat)[it][ix][iz];
      }
  }

}
