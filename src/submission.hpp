#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols):
    rows_{rows}, cols_{cols}, data_(rows * cols, 0.0) {}

  double& operator()(std::size_t i, std::size_t j){
    return data_[i * cols_ + j];
  }

  double  operator()(std::size_t i, std::size_t j) const{
    return data_[i * cols_ + j];
  }

  std::size_t rows() const{ return rows_; }
  std::size_t cols() const{ return cols_; }

  double* data(){ return data_.data(); }
  const double* data() const{ return data_.data(); }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid){
  const std::size_t numRows = old_grid.rows();
  const std::size_t numCols = old_grid.cols();

  // copy boundary points from old_grid unchanged
  for(std::size_t i = 0; i < numRows; ++i){
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, numCols - 1) = old_grid(i, numCols - 1);
  }

  for(std::size_t j = 0; j < numCols; ++j){
    new_grid(0, j) = old_grid(0, j);
    new_grid(numRows - 1, j) = old_grid(numRows - 1, j);
  }

  // calculate new_grid interior points using weighted average
  const double* oldData = old_grid.data();
  double* newData = new_grid.data();

  for(std::size_t i = 1; i < numRows - 1; ++i){
    const double* prev = oldData + (i - 1) * numCols;
    const double* curr = oldData + i * numCols;
    const double* next = oldData + (i + 1) * numCols;
    double* out = newData + i * numCols;
    for(std::size_t j = 1; j < numCols - 1; ++j){
      out[j] = 0.5 * curr[j]
              + 0.125 * (prev[j] + curr[j - 1] + curr[j + 1] + next[j]);
    }
  }
}
