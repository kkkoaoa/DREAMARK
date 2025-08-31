#include "vol/Contouring.hpp"

#include <stddef.h>
#include <array>
#include <vector>
#include <iostream>

namespace dc{

using namespace math;
using namespace util;


template<typename vector3, typename formula1, typename formula2>
void dual_contouring(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, formula1 f, formula2 df,
    std::vector<double>& vertices, std::vector<typename vector3::size_type>& polygons){
    using coord_type = typename vector3::value_type;
    using size_type = typename vector3::size_type;

    // Some initial checks
    if(numx < 2 || numy < 2 || numz < 2)
        return;

    if(!std::equal(std::begin(lower), std::end(lower), std::begin(upper),
                   [](double a, double b)->bool {return a <= b;}))
        return;

    // numx, numy and numz are the numbers of evaluations in each direction
    --numx; --numy; --numz;

    coord_type dx = (upper[0] - lower[0]) / static_cast<coord_type>(numx);
    coord_type dy = (upper[1] - lower[1]) / static_cast<coord_type>(numy);
    coord_type dz = (upper[2] - lower[2]) / static_cast<coord_type>(numz);

    const int num_shared_indices = 2 * (numy + 1) * (numz + 1);
    std::vector<size_type> shared_indices_x(num_shared_indices);
    std::vector<size_type> shared_indices_y(num_shared_indices);
    std::vector<size_type> shared_indices_z(num_shared_indices);
    auto _offset = [&](size_t i, size_t j, size_t k){return i*(numy+1)*(numz+1) + j*(numz+1) + k;};

    const auto Size = math::Vec3u(numx);
    Field field(Size, Plane{+INF, Zero});

    foreach3D(Size, [&](const math::Vec3u& p){
        // std::cout << p << f(p.x, p.y, p.z) << std::endl;
        auto& cell = field[p];
        cell.dist = f(p.x, p.y, p.z);
        cell.normal = Vec3(df(p.x, p.y, p.z, 0), df(p.x, p.y, p.z, 1), df(p.x, p.y, p.z, 2));
        // std::cout << p << cell.normal << std::endl;
    });

    const auto mesh = dualContouring(field);
    
    for (auto v: mesh.vecs) {
        vertices.push_back(v.x);
        vertices.push_back(v.y);
        vertices.push_back(v.z);
    }
    for (auto f: mesh.triangles) {
        polygons.push_back(f[0]);
        polygons.push_back(f[1]);
        polygons.push_back(f[2]);
    }
}


}