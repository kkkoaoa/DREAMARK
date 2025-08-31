#include "remesh.h"

#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

#include <boost/iterator/function_output_iterator.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel     K;
typedef CGAL::Surface_mesh<K::Point_3>                          Surface_mesh;
typedef Surface_mesh::Vertex_index                              vertex_descriptor;
typedef Surface_mesh::Face_index                                face_descriptor;

typedef boost::graph_traits<Surface_mesh>::halfedge_descriptor  halfedge_descriptor;
typedef boost::graph_traits<Surface_mesh>::edge_descriptor      edge_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

struct halfedge2edge {
    halfedge2edge(const Surface_mesh& m, std::vector<edge_descriptor>& edges)
            : m_mesh(m), m_edges(edges)
    {}
    void operator()(const halfedge_descriptor& h) const
    {
        m_edges.push_back(edge(h, m_mesh));
    }
    const Surface_mesh& m_mesh;
    std::vector<edge_descriptor>& m_edges;
};


std::vector<float> edge_analysis(Surface_mesh mesh) {
    float mean = 0, min, max;
    int count = 0; bool init = true;
    for (edge_descriptor ed: mesh.edges()) {
        float length = PMP::edge_length(ed, mesh);
        mean += length;
        if (init) {
            init = false;
            min = max = length;
        } else {
            min = std::min(min, length);
            max = std::max(max, length);
        }
        ++count;
    }
    return std::vector<float>{mean/count, min, max};
}

std::vector<at::Tensor> remesh_impl(at::Tensor xyz, at::Tensor triangles, double target_edge_length, unsigned int nb_iter) {
    Surface_mesh mesh;
    std::vector<vertex_descriptor> vertex_descriptors(xyz.size(0));
    for (int i=0; i<xyz.size(0); ++i) {
        vertex_descriptor id = mesh.add_vertex(K::Point_3(
                xyz[i][0].item<float>(), xyz[i][1].item<float>(), xyz[i][2].item<float>()
        ));
        vertex_descriptors[i] = id;
    }
    for (int i=0; i<triangles.size(0); ++i) {
        int v0 = triangles[i][0].item<int>();
        int v1 = triangles[i][1].item<int>();
        int v2 = triangles[i][2].item<int>();
        face_descriptor id = mesh.add_face(
                vertex_descriptors[v0],
                vertex_descriptors[v1],
                vertex_descriptors[v2]
        );
    }
    std::vector<float> ans = edge_analysis(mesh);
    std::cout << ans[0] << " " << ans[1] << " " << ans[2] << std::endl;

//    std::cout << "Split border...";
    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
    PMP::split_long_edges(border, target_edge_length, mesh);
//    std::cout << "done." << std::endl;

    std::cout << "Start remeshing of " << mesh.num_vertices()
              << " (" << mesh.num_faces() << " faces)..." << std::endl;

    PMP::isotropic_remeshing(faces(mesh), target_edge_length, mesh,
                             CGAL::parameters::number_of_iterations(nb_iter)
                                     .protect_constraints(true)); //i.e. protect border, here
    std::cout << "end remeshing as " << mesh.num_vertices()
              << " (" << mesh.num_faces() << " faces)..." << std::endl;

    CGAL::IO::write_polygon_mesh("out.off", mesh, CGAL::parameters::stream_precision(17));
    at::Tensor new_xyz = torch::empty({mesh.num_vertices(), 3}, at::device(xyz.device()).dtype(at::ScalarType::Float));
    at::Tensor new_triangles = torch::empty({mesh.num_faces(), 3}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    return std::vector<at::Tensor>{new_xyz, new_triangles};
}