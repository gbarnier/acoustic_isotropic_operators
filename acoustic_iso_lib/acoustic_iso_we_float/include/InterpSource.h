/********************************************//**
 Author: Stuart Farris
 Date: Summer 2017
 Description:  Meant to interpolate source signature between recording sampling(coarse) and propogating sampling(fine).
 ***********************************************/
#pragma once
#include <operator.h>
#include <float1DReg.h>

class InterpSource: public Operator<SEP::float1DReg, SEP::float1DReg>{
  public:
  InterpSource(const std::shared_ptr<SEP::float1DReg> model,
    	const std::shared_ptr<SEP::float1DReg> data,
      float oversamp);

  InterpSource(const std::shared_ptr<SEP::float1DReg> model,
  	const std::shared_ptr<SEP::float1DReg> data,
  	const std::shared_ptr<SEP::float1DReg> dataCoordinates,
    float oversamp);

  void forward(const bool add, const std::shared_ptr<SEP::float1DReg> model,
    std::shared_ptr<SEP::float1DReg> data);

  void adjoint(const bool add,  std::shared_ptr<SEP::float1DReg> model,
    const std::shared_ptr<SEP::float1DReg> data);

private:
	float _o1; /**< origin of time samples in model */
	float _d1; /**< delta between course time samples */
  float _scale; /**< 1/(oversampling from course to fine) */
	std::shared_ptr<SEP::float1DReg> _dataCoordinates; /**< holds time values at fine sampling rate*/
};
