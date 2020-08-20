#include <SecondDeriv_multi_exp_V2.h>

SecondDeriv_multi_exp_V2::SecondDeriv_multi_exp_V2(const std::shared_ptr<SEP::float4DReg>model,
                         const std::shared_ptr<SEP::float4DReg>data
                         ) {
  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(4).n == data->getHyper()->getAxis(4).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //t
   n4 = model->getHyper()->getAxis(4).n; //experiment

  // set domain and range
  setDomainRange(model, data);

  _dt = model->getHyper()->getAxis(3).d;

  // calculate lapl coefficients

  C0t_10 = -2.927222222 / (_dt * _dt);
  C1t_10 =  1.666666667 / (_dt * _dt);
  C2t_10 = -0.23809524 / (_dt * _dt);
  C3t_10 =  0.03968254 / (_dt * _dt);
  C4t_10 = -0.00496032 / (_dt * _dt);
  C5t_10 =  0.00031746 / (_dt * _dt);

  C0t_8 = -2.84722222222 / (_dt * _dt);
  C1t_8 =  1.600000000 / (_dt * _dt);
  C2t_8 = -0.200000000/ (_dt * _dt);
  C3t_8 =  0.02539682539 / (_dt * _dt);
  C4t_8 = -0.00178571428 / (_dt * _dt);

  C0t_6 = -2.72222222222 / (_dt * _dt);
  C1t_6 =  1.500000000 / (_dt * _dt);
  C2t_6 = -0.150000000/ (_dt * _dt);
  C3t_6 =  0.01111111111 / (_dt * _dt);

  C0t_4 = -2.50000000000 / (_dt * _dt);
  C1t_4 =  1.33333333333 / (_dt * _dt);
  C2t_4 = -0.08333333333 / (_dt * _dt);

  C0t_2 = -2.00000000000 / (_dt * _dt);
  C1t_2 =  1.000000000 / (_dt * _dt);

  setDomainRange(model, data);

}

void SecondDeriv_multi_exp_V2::forward(const bool                         add,
                          const std::shared_ptr<SEP::float4DReg>model,
                          std::shared_ptr<SEP::float4DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);


    const std::shared_ptr<float4D> m =
      ((std::dynamic_pointer_cast<float4DReg>(model))->_mat);
     std::shared_ptr<float4D> d =
      ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);

  //boundary condition
  #pragma omp parallel for collapse(3)
  for (int iexp = 0; iexp < n4; iexp++) { //experiment
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*d)[iexp][0][ix][iz] += //second time deriv
  		//C0t_2*(  (*m)[iexp][1][ix ][iz ])  -
         		(C0t_2*  (*m)[iexp][1][ix ][iz ] + \
                  C1t_2*(  (*m)[iexp][1-1][ix ][iz ]+(*m)[iexp][1+1][ix ][iz])) ;
        (*d)[iexp][1][ix][iz] += //second time deriv
         		//C0t_4*(  (*m)[iexp][2][ix ][iz ])   -
         		(C0t_4*  (*m)[iexp][2][ix ][iz ] + \
                  C1t_4*(  (*m)[iexp][2-1][ix ][iz ]+(*m)[iexp][2 + 1][ix ][iz]) + \
                  C2t_4*(  (*m)[iexp][2-2][ix ][iz ]+(*m)[iexp][2 + 2][ix ][iz])) ;
        (*d)[iexp][2][ix][iz] += //second time deriv
  		//C0t_6*(  (*m)[iexp][3][ix ][iz ])   -
         		(C0t_6*  (*m)[iexp][3][ix ][iz ] + \
                  C1t_6*(  (*m)[iexp][3-1][ix ][iz ]+(*m)[iexp][3 + 1][ix ][iz]) + \
                  C2t_6*(  (*m)[iexp][3-2][ix ][iz ]+(*m)[iexp][3 + 2][ix ][iz]) + \
                  C3t_6*(  (*m)[iexp][3-3][ix ][iz ]+(*m)[iexp][3 + 3][ix ][iz])) ;
        (*d)[iexp][3][ix][iz] += //second time deriv
  		//C0t_8*(  (*m)[iexp][4][ix ][iz ])   -
         		(C0t_8*  (*m)[iexp][4][ix ][iz ] + \
                  C1t_8*(  (*m)[iexp][4-1][ix ][iz ]+(*m)[iexp][4 + 1][ix ][iz]) + \
                  C2t_8*(  (*m)[iexp][4-2][ix ][iz ]+(*m)[iexp][4 + 2][ix ][iz]) + \
                  C3t_8*(  (*m)[iexp][4-3][ix ][iz ]+(*m)[iexp][4 + 3][ix ][iz])  + \
                  C4t_8*(  (*m)[iexp][4-4][ix ][iz ]+(*m)[iexp][4 + 4][ix ][iz])) ;
        (*d)[iexp][n3-6][ix][iz] += //second time deriv
  		//C0t_8*(  (*m)[iexp][n3-5][ix ][iz ])   -
         		(C0t_8*  (*m)[iexp][n3-5][ix ][iz ]+ \
                  C1t_8*(  (*m)[iexp][n3-5-1][ix ][iz ]+(*m)[iexp][n3-5 + 1][ix ][iz]) + \
                  C2t_8*(  (*m)[iexp][n3-5-2][ix ][iz ]+(*m)[iexp][n3-5 + 2][ix ][iz]) + \
                  C3t_8*(  (*m)[iexp][n3-5-3][ix ][iz ]+(*m)[iexp][n3-5 + 3][ix ][iz])  + \
                  C4t_8*(  (*m)[iexp][n3-5-4][ix ][iz ]+(*m)[iexp][n3-5 + 4][ix ][iz])) ;
        (*d)[iexp][n3-5][ix][iz] += //second time deriv
  		//C0t_6*(  (*m)[iexp][n3-4][ix][iz])   -
         		(C0t_6*  (*m)[iexp][n3-4][ix ][iz ] + \
                  C1t_6*(  (*m)[iexp][n3-4-1][ix ][iz ]+(*m)[iexp][n3-4 + 1][ix ][iz]) + \
                  C2t_6*(  (*m)[iexp][n3-4-2][ix ][iz ]+(*m)[iexp][n3-4 + 2][ix ][iz]) + \
                  C3t_6*(  (*m)[iexp][n3-4-3][ix ][iz ]+(*m)[iexp][n3-4 + 3][ix ][iz])) ;
        (*d)[iexp][n3-4][ix][iz] += //second time deriv
  		//C0t_4*(  (*m)[iexp][n3-3][ix ][iz ])   -
         		(C0t_4*  (*m)[iexp][n3-3][ix ][iz ] + \
                  C1t_4*(  (*m)[iexp][n3-3-1][ix ][iz ]+(*m)[iexp][n3-3 + 1][ix ][iz]) + \
                  C2t_4*(  (*m)[iexp][n3-3-2][ix ][iz ]+(*m)[iexp][n3-3 + 2][ix ][iz])) ;
        (*d)[iexp][n3-3][ix][iz] += //second time deriv
  		//C0t_2*(  (*m)[iexp][n3-2][ix ][iz ])   -
         		(C0t_2*  (*m)[iexp][n3-2][ix ][iz ] + \
                  C1t_2*(  (*m)[iexp][n3-2-1][ix ][iz ]+(*m)[iexp][n3-2 + 1][ix ][iz])) ;
        (*d)[iexp][n3-2][ix][iz] += (*m)[iexp][0 ][ix ][iz]-(*m)[iexp][1 ][ix ][iz];
        (*d)[iexp][n3-1][ix][iz] += (*m)[iexp][1 ][ix ][iz];
      }
    }
  }
  #pragma omp parallel for collapse(4)
  for (int iexp = 0; iexp < n4; iexp++) { //experiment
    for (int it = 5; it < n3-5; it++) { //time
      for (int ix = 0; ix < n2; ix++) { //x
        for (int iz = 0; iz < n1; iz++) { //z
          (*d)[iexp][it-1][ix][iz] +=//second time deriv
         		(C0t_10*  (*m)[iexp][it][ix ][iz ] + \
                  C1t_10*(  (*m)[iexp][it-1][ix ][iz ]+(*m)[iexp][it + 1][ix ][iz]) + \
                  C2t_10*(  (*m)[iexp][it-2][ix ][iz ]+(*m)[iexp][it + 2][ix ][iz]) + \
                  C3t_10*(  (*m)[iexp][it-3][ix ][iz ]+(*m)[iexp][it + 3][ix ][iz]) + \
                  C4t_10*(  (*m)[iexp][it-4][ix ][iz ]+(*m)[iexp][it + 4][ix ][iz]) + \
                  C5t_10*(  (*m)[iexp][it-5][ix ][iz ]+(*m)[iexp][it + 5][ix ][iz]));
        }
      }
    }
  }

}

void SecondDeriv_multi_exp_V2::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float4DReg>      model,
                          const std::shared_ptr<SEP::float4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

    std::shared_ptr<float4D> m =
      ((std::dynamic_pointer_cast<float4DReg>(model))->_mat);
    const std::shared_ptr<float4D> d =
      ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);


  #pragma omp parallel for collapse(3)
  for (int iexp = 0; iexp < n4; iexp++) { //experiment
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*m)[iexp][0][ix][iz] +=   //second time deriv
                           (C1t_2*( (*d)[iexp][0][ix ][iz]) + \
                           C2t_4*( (*d)[iexp][1][ix ][iz]) + \
                           C3t_6*( (*d)[iexp][2][ix ][iz]) + \
                           C4t_8*( (*d)[iexp][3][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][4][ix ][iz]))  + \
                           //initial condition
         			 (*d)[iexp][n3-2][ix ][iz ];
        (*m)[iexp][1][ix][iz] +=   //second time deriv
  			//C0t_2*( (*d)[iexp][0][ix ][iz])   -
                           (C0t_2*( (*d)[iexp][0][ix ][iz]) + \
                           C1t_4*( (*d)[iexp][1][ix ][iz]) + \
                           C2t_6*( (*d)[iexp][2][ix ][iz]) + \
                           C3t_8*( (*d)[iexp][3][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][4][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][5][ix ][iz]))  +
                           //initial condition
         			 (*d)[iexp][n3-1][ix ][iz ] - (*d)[iexp][n3-2][ix ][iz ];
        (*m)[iexp][2][ix][iz] +=   //second time deriv
  			//C0t_4*( (*d)[iexp][1][ix ][iz])   -
                           (C0t_4*( (*d)[iexp][1][ix ][iz]) + \
                           C1t_2*(*d)[iexp][1-1][ix ][iz] + C1t_6*( (*d)[iexp][1+1][ix ][iz]) + \
                           C2t_8*( (*d)[iexp][1+2][ix ][iz]) + \
                           C3t_10*( (*d)[iexp][1+3][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][1+4][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][1+5][ix ][iz])) ;
        (*m)[iexp][3][ix][iz] +=   //second time deriv
  			//C0t_6*( (*d)[iexp][2][ix ][iz])   -
                           (C0t_6*( (*d)[iexp][2][ix ][iz]) + \
                           C1t_4*(*d)[iexp][1][ix ][iz] + C1t_8*( (*d)[iexp][3][ix ][iz]) + \
                           C2t_10*( (*d)[iexp][4][ix ][iz]) + \
                           C3t_10*( (*d)[iexp][5][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][6][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][7][ix ][iz])) ;
        (*m)[iexp][4][ix][iz] +=   //second time deriv
                           //C0t_8*( (*d)[iexp][3][ix ][iz])   -
                           (C0t_8*( (*d)[iexp][3][ix ][iz]) + \
                           C1t_6*(*d)[iexp][2][ix ][iz] + C1t_10*( (*d)[iexp][4][ix ][iz]) + \
                           C2t_4*(*d)[iexp][1][ix ][iz] + C2t_10*( (*d)[iexp][5][ix ][iz]) + \
                           C3t_10*( (*d)[iexp][6][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][7][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][8][ix ][iz])) ;
        (*m)[iexp][5][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][4][ix ][iz])   -
                           (C0t_10*( (*d)[iexp][4][ix ][iz]) + \
                           C1t_8*(*d)[iexp][3][ix ][iz] + C1t_10*( (*d)[iexp][5][ix ][iz]) + \
                           C2t_6*(*d)[iexp][2][ix ][iz] + C2t_10*( (*d)[iexp][6][ix ][iz]) + \
                           C3t_10*( (*d)[iexp][7][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][8][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][9][ix ][iz])) ;
        (*m)[iexp][6][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][5][ix ][iz])   -
                          (C0t_10*( (*d)[iexp][5][ix ][iz]) + \
                           C1t_10*(*d)[iexp][4][ix ][iz] + C1t_10*( (*d)[iexp][6][ix ][iz]) + \
                           C2t_8*(*d)[iexp][3][ix ][iz] + C2t_10*( (*d)[iexp][7][ix ][iz]) + \
                           C3t_6*(*d)[iexp][2][ix ][iz] + C3t_10*( (*d)[iexp][8][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][9][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][10][ix ][iz])) ;
        (*m)[iexp][7][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][6][ix ][iz])   -
                           (C0t_10*( (*d)[iexp][6][ix ][iz])+ \
                           C1t_10*(*d)[iexp][5][ix ][iz] + C1t_10*( (*d)[iexp][7][ix ][iz]) + \
                           C2t_10*(*d)[iexp][4][ix ][iz] + C2t_10*( (*d)[iexp][8][ix ][iz]) + \
                           C3t_8*(*d)[iexp][3][ix ][iz] + C3t_10*( (*d)[iexp][9][ix ][iz]) + \
                           C4t_10*( (*d)[iexp][10][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][11][ix ][iz])) ;
        (*m)[iexp][8][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][7][ix ][iz])  -
                          (C0t_10*( (*d)[iexp][7][ix ][iz]) + \
                           C1t_10*(*d)[iexp][6][ix ][iz] + C1t_10*( (*d)[iexp][8][ix ][iz]) + \
                           C2t_10*(*d)[iexp][5][ix ][iz] + C2t_10*( (*d)[iexp][9][ix ][iz]) + \
                           C3t_10*(*d)[iexp][4][ix ][iz] + C3t_10*( (*d)[iexp][10][ix ][iz]) + \
                           C4t_8*(*d)[iexp][3][ix ][iz] + C4t_10*( (*d)[iexp][11][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][12][ix ][iz])) ;
        (*m)[iexp][9][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][8][ix ][iz])  -
                          (C0t_10*( (*d)[iexp][8][ix ][iz]) + \
                           C1t_10*(*d)[iexp][7][ix ][iz] + C1t_10*( (*d)[iexp][9][ix ][iz]) + \
                           C2t_10*(*d)[iexp][6][ix ][iz] + C2t_10*( (*d)[iexp][10][ix ][iz]) + \
                           C3t_10*(*d)[iexp][5][ix ][iz] + C3t_10*( (*d)[iexp][11][ix ][iz]) + \
                           C4t_10*(*d)[iexp][4][ix ][iz] + C4t_10*( (*d)[iexp][12][ix ][iz]) + \
                           C5t_10*( (*d)[iexp][13][ix ][iz])) ;
        (*m)[iexp][n3-10][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][n3-11][ix ][iz])   -
                           (C0t_10*( (*d)[iexp][n3-11][ix ][iz]) + \
                           C1t_10*(*d)[iexp][n3-11-1][ix ][iz] + C1t_10*( (*d)[iexp][n3-11+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-11-2][ix ][iz] + C2t_10*( (*d)[iexp][n3-11+2][ix ][iz]) + \
                           C3t_10*(*d)[iexp][n3-11-3][ix ][iz] + C3t_10*( (*d)[iexp][n3-11+3][ix ][iz]) + \
                           C4t_10*(*d)[iexp][n3-11-4][ix ][iz] + C4t_10*( (*d)[iexp][n3-11+4][ix ][iz]) + \
                           C5t_10*((*d)[iexp][n3-11-5][ix ][iz])) ;
        (*m)[iexp][n3-9][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][n3-10][ix ][iz])   -
                           (C0t_10*( (*d)[iexp][n3-10][ix ][iz]) + \
                           C1t_10*(*d)[iexp][n3-10-1][ix ][iz] + C1t_10*( (*d)[iexp][n3-10+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-10-2][ix ][iz] + C2t_10*( (*d)[iexp][n3-10+2][ix ][iz]) + \
                           C3t_10*(*d)[iexp][n3-10-3][ix ][iz] + C3t_10*( (*d)[iexp][n3-10+3][ix ][iz]) + \
                           C4t_10*(*d)[iexp][n3-10-4][ix ][iz] + C4t_8*( (*d)[iexp][n3-10+4][ix ][iz]) + \
                           C5t_10*((*d)[iexp][n3-10-5][ix ][iz])) ;
        (*m)[iexp][n3-8][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][n3-9][ix ][iz])   -
                           (C0t_10*( (*d)[iexp][n3-9][ix ][iz]) + \
                           C1t_10*(*d)[iexp][n3-9-1][ix ][iz] + C1t_10*( (*d)[iexp][n3-9+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-9-2][ix ][iz] + C2t_10*( (*d)[iexp][n3-9+2][ix ][iz]) + \
                           C3t_10*(*d)[iexp][n3-9-3][ix ][iz] + C3t_8*( (*d)[iexp][n3-9+3][ix ][iz]) + \
                           C4t_10*(*d)[iexp][n3-9-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-9-5][ix ][iz])) ;
        (*m)[iexp][n3-7][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][n3-8][ix ][iz])   -
                           (C0t_10*( (*d)[iexp][n3-8][ix ][iz]) + \
                           C1t_10*(*d)[iexp][n3-8-1][ix ][iz] + C1t_10*( (*d)[iexp][n3-8+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-8-2][ix ][iz] + C2t_8*( (*d)[iexp][n3-8+2][ix ][iz]) + \
                           C3t_10*(*d)[iexp][n3-8-3][ix ][iz] + C3t_6*( (*d)[iexp][n3-8+3][ix ][iz]) + \
                           C4t_10*(*d)[iexp][n3-8-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-8-5][ix ][iz])) ;
        (*m)[iexp][n3-6][ix][iz] +=   //second time deriv
  			//C0t_10*( (*d)[iexp][n3-7][ix ][iz])  -
                           (C0t_10*( (*d)[iexp][n3-7][ix ][iz]) + \
                           C1t_10*(*d)[iexp][n3-7-1][ix ][iz] + C1t_8*( (*d)[iexp][n3-7+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-7-2][ix ][iz] + C2t_6*( (*d)[iexp][n3-7+2][ix ][iz]) + \
                           C3t_10*(*d)[iexp][n3-7-3][ix ][iz] + \
                           C4t_10*(*d)[iexp][n3-7-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-7-5][ix ][iz])) ;
        (*m)[iexp][n3-5][ix][iz] +=   //second time deriv
  			//C0t_8*( (*d)[iexp][n3-6][ix ][iz])   -
                           (C0t_8*( (*d)[iexp][n3-6][ix ][iz]) + \
                           C1t_10*(*d)[iexp][n3-6-1][ix ][iz] + C1t_6*( (*d)[iexp][n3-6+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-6-2][ix ][iz] + C2t_4*( (*d)[iexp][n3-6+2][ix ][iz]) + \
                           C3t_10*(*d)[iexp][n3-6-3][ix ][iz] + \
                           C4t_10*(*d)[iexp][n3-6-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-6-5][ix ][iz])) ;
        (*m)[iexp][n3-4][ix][iz] +=   //second time deriv
  			//C0t_6*( (*d)[iexp][n3-5][ix ][iz])   -
                           (C0t_6*( (*d)[iexp][n3-5][ix ][iz]) + \
                           C1t_8*(*d)[iexp][n3-5-1][ix ][iz] + C1t_4*( (*d)[iexp][n3-5+1][ix ][iz]) + \
                           C2t_10*(*d)[iexp][n3-5-2][ix ][iz] + \
                           C3t_10*(*d)[iexp][n3-5-3][ix ][iz] + \
                           C4t_10*(*d)[iexp][n3-5-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-5-5][ix ][iz])) ;
        (*m)[iexp][n3-3][ix][iz] +=   //second time deriv
  			//C0t_4*( (*d)[iexp][n3-4][ix ][iz])   -
                           (C0t_4*( (*d)[iexp][n3-4][ix ][iz]) + \
                           C1t_6*(*d)[iexp][n3-4-1][ix ][iz] + C1t_2*( (*d)[iexp][n3-4+1][ix ][iz]) + \
                           C2t_8*(*d)[iexp][n3-4-2][ix ][iz] + \
                           C3t_10*(*d)[iexp][n3-4-3][ix ][iz] + \
                           C4t_10*(*d)[iexp][n3-4-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-4-5][ix ][iz])) ;
        (*m)[iexp][n3-2][ix][iz] +=   //second time deriv
  			//C0t_2*( (*d)[iexp][n3-3][ix ][iz])   -
                           (C0t_2*( (*d)[iexp][n3-3][ix ][iz]) + \
                           C1t_4*(*d)[iexp][n3-3-1][ix ][iz] + \
                           C2t_6*(*d)[iexp][n3-3-2][ix ][iz] + \
                           C3t_8*(*d)[iexp][n3-3-3][ix ][iz] + \
                           C4t_10*(*d)[iexp][n3-3-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-3-5][ix ][iz]))  ;
        (*m)[iexp][n3-1][ix][iz] +=   //second time deriv
                           (C1t_2*(*d)[iexp][n3-2-1][ix ][iz] + \
                           C2t_4*(*d)[iexp][n3-2-2][ix ][iz] + \
                           C3t_6*(*d)[iexp][n3-2-3][ix ][iz] + \
                           C4t_8*(*d)[iexp][n3-2-4][ix ][iz] + \
                           C5t_10*((*d)[iexp][n3-2-5][ix ][iz])) ;
      }
    }
  }

  #pragma omp parallel for collapse(4)
  for (int iexp = 0; iexp < n4; iexp++) { //experiment
    for (int it = 9; it < n3-11; it++) {
      for (int ix = 0; ix < n2; ix++) {
        for (int iz = 0; iz < n1; iz++) {
          (*m)[iexp][it+1][ix][iz] += //second time deriv
                           //C0t_10*( (*d)[iexp][it][ix ][iz])*(*s)[ix][iz] -
         			 (C0t_10*  (*d)[iexp][it][ix ][iz ]+ \
                           C1t_10*((*d)[iexp][it-1][ix ][iz ] +  (*d)[iexp][it + 1][ix ][iz]) + \
                           C2t_10*((*d)[iexp][it-2][ix ][iz ] +  (*d)[iexp][it + 2][ix ][iz]) + \
                           C3t_10*((*d)[iexp][it-3][ix ][iz ] +  (*d)[iexp][it + 3][ix ][iz]) + \
                           C4t_10*((*d)[iexp][it-4][ix ][iz ] +  (*d)[iexp][it + 4][ix ][iz]) + \
                           C5t_10*((*d)[iexp][it-5][ix ][iz ] +  (*d)[iexp][it + 5][ix ][iz]));
        }
      }
    }
  }
}
