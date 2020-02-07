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

//multiple shots on last axis
padTruncateSource_mutli_exp::padTruncateSource_mutli_exp(const std::shared_ptr<float2DReg> model, const std::shared_ptr<float4DReg> data, std::vector<std::vector<int>> gridPointIndexUnique_byExperiment, std::vector<std::map<int, int>> indexMaps ){
  // assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(4).n); //z axis
  // assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(3).n); //x axis
  assert(data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(2).n); //time axis

  //ensure number of sources/rec to inject is equal to the number of gridpoints provided
  //assert(model->getHyper()->getAxis(1).n == gridPointIndexUnique_byExperiment.size());
  assert(data->getHyper()->getAxis(4).n == gridPointIndexUnique_byExperiment.size());

  _nz_data = data->getHyper()->getAxis(1).n;
  _nx_data = data->getHyper()->getAxis(2).n;
  _oz_data = data->getHyper()->getAxis(1).o;
  _ox_data = data->getHyper()->getAxis(2).o;
  _dz_data = data->getHyper()->getAxis(1).d;
  _dx_data= data->getHyper()->getAxis(2).d;
  _nt = data->getHyper()->getAxis(3).n;

  _gridPointIndexUnique_byExperiment = gridPointIndexUnique_byExperiment;
  _indexMaps=indexMaps;

  setDomainRange(model,data);
}

//! FWD
/*!
* this pads from source to wavefield
*/
void padTruncateSource_mutli_exp::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float4DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) data->scale(0.);

  //for each experiment
  //#pragma omp parallel for collapse(2)
  for(int iExp = 0; iExp < _gridPointIndexUnique_byExperiment.size(); iExp++){
    // for each device add to correct location in wavefield
    for(int id = 0; id < _gridPointIndexUnique_byExperiment[iExp].size(); id++){
        int gridPoint = _gridPointIndexUnique_byExperiment[iExp][id];
        int ix = (int)gridPoint/_nz_data;
        int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
        int itrace = _indexMaps[iExp].find(gridPoint)->second;
        #pragma omp parallel for
        for(int it = 0; it < _nt; it++){
          // (*data->_mat)[0][it][x_index_data][z_index_data] += (*model->_mat)[iz_model][ix_model][0][it];
          (*data->_mat)[iExp][it][ix][iz] += (*model->_mat)[it][itrace];
        }
    }
  }
}

//! ADJ
/*!
* this truncates from wavefield to source
*/
void padTruncateSource_mutli_exp::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float4DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) model->scale(0.);

  //for each experiment
  //#pragma omp parallel for collapse(2)
  for(int iExp = 0; iExp < _gridPointIndexUnique_byExperiment.size(); iExp++){
    //for each device add to correct location in wavefield
    for(int id = 0; id < _gridPointIndexUnique_byExperiment[iExp].size(); id++){
        int gridPoint = _gridPointIndexUnique_byExperiment[iExp][id];
        int ix = (int)gridPoint/_nz_data;
        int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
        int itrace = _indexMaps[iExp].find(gridPoint)->second;
        #pragma omp parallel for
        for(int it = 0; it < _nt; it++){
          // (*data->_mat)[0][it][x_index_data][z_index_data] += (*model->_mat)[iz_model][ix_model][0][it];
          (*model->_mat)[it][itrace] += (*data->_mat)[iExp][it][ix][iz];
        }
    }
  }

}

//multiple shots on last axis with complex wavefield
padTruncateSource_mutli_exp_complex::padTruncateSource_mutli_exp_complex(const std::shared_ptr<complex2DReg> model, const std::shared_ptr<complex4DReg> data, std::vector<std::vector<int>> gridPointIndexUnique_byExperiment, std::vector<std::map<int, int>> indexMaps ){
  // assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(4).n); //z axis
  // assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(3).n); //x axis
  assert(data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(2).n); //time axis

  //ensure number of sources/rec to inject is equal to the number of gridpoints provided
  //assert(model->getHyper()->getAxis(1).n == gridPointIndexUnique_byExperiment.size());
  assert(data->getHyper()->getAxis(4).n == gridPointIndexUnique_byExperiment.size());

  _nz_data = data->getHyper()->getAxis(1).n;
  _nx_data = data->getHyper()->getAxis(2).n;
  _oz_data = data->getHyper()->getAxis(1).o;
  _ox_data = data->getHyper()->getAxis(2).o;
  _dz_data = data->getHyper()->getAxis(1).d;
  _dx_data= data->getHyper()->getAxis(2).d;
  _nt = data->getHyper()->getAxis(3).n;

  _gridPointIndexUnique_byExperiment = gridPointIndexUnique_byExperiment;
  _indexMaps=indexMaps;

  setDomainRange(model,data);
}

//! FWD
/*!
* this pads from source to wavefield
*/
void padTruncateSource_mutli_exp_complex::forward(const bool add, const std::shared_ptr<complex2DReg> model, std::shared_ptr<complex4DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) data->scale(0.);

  //for each experiment
  //#pragma omp parallel for collapse(2)
  for(int iExp = 0; iExp < _gridPointIndexUnique_byExperiment.size(); iExp++){
    // for each device add to correct location in wavefield
    for(int id = 0; id < _gridPointIndexUnique_byExperiment[iExp].size(); id++){
        int gridPoint = _gridPointIndexUnique_byExperiment[iExp][id];
        int ix = (int)gridPoint/_nz_data;
        int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
        int itrace = _indexMaps[iExp].find(gridPoint)->second;
        #pragma omp parallel for
        for(int it = 0; it < _nt; it++){
          // (*data->_mat)[0][it][x_index_data][z_index_data] += (*model->_mat)[iz_model][ix_model][0][it];
          (*data->_mat)[iExp][it][ix][iz] += (*model->_mat)[it][itrace];
        }
    }
  }
}

//! ADJ
/*!
* this truncates from wavefield to source
*/
void padTruncateSource_mutli_exp_complex::adjoint(const bool add, std::shared_ptr<complex2DReg> model, const std::shared_ptr<complex4DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) model->scale(0.);

  //for each experiment
  //#pragma omp parallel for collapse(2)
  for(int iExp = 0; iExp < _gridPointIndexUnique_byExperiment.size(); iExp++){
    //for each device add to correct location in wavefield
    for(int id = 0; id < _gridPointIndexUnique_byExperiment[iExp].size(); id++){
        int gridPoint = _gridPointIndexUnique_byExperiment[iExp][id];
        int ix = (int)gridPoint/_nz_data;
        int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
        int itrace = _indexMaps[iExp].find(gridPoint)->second;
        #pragma omp parallel for
        for(int it = 0; it < _nt; it++){
          // (*data->_mat)[0][it][x_index_data][z_index_data] += (*model->_mat)[iz_model][ix_model][0][it];
          (*model->_mat)[it][itrace] += (*data->_mat)[iExp][it][ix][iz];
        }
    }
  }

}
