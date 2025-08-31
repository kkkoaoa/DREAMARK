
#include "wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_query", &knn_query);
    m.def("ball_query", &ball_query);
    m.def("graph_neighbor_query", &graph_neighbor_query);

    m.def("gather_points", &gather_points);
    m.def("gather_points_grad", &gather_points_grad);

    m.def("group_points", &group_points);
    m.def("group_points_grad", &group_points_grad);

    m.def("furthest_point_sampling", &furthest_point_sampling);

    m.def("sdf_by_normal", &sdf_by_normal);

}
