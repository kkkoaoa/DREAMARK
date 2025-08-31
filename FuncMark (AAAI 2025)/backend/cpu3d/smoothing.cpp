
#include "smoothing.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SparseLU>

#include "cuda_utils.h"

typedef Eigen::Triplet<double, int> Triplet;
/// declares a column-major sparse matrix type of double
typedef Eigen::SparseMatrix<double> Sparse_mat;

static std::pair<int, int> pair_from_tri(const int* tri, int current_vert) {
    int ids[2] = {-1, -1};
    for(int i = 0; i < 3; i++)
    {
        if(tri[i] == current_vert)
        {
            ids[0] = tri[ (i+1) % 3 ];
            ids[1] = tri[ (i+2) % 3 ];
            break;
        }
    }
    return std::pair<int, int>(ids[0], ids[1]);
}

static bool add_to_ring(std::deque<int>& ring, std::pair<int, int> p) {
    if(ring[ring.size()-1] == p.first)
    {
        ring.push_back(p.second);
        return true;
    }
    else if(ring[ring.size()-1] == p.second)
    {

        ring.push_back(p.first);
        return true;
    }
    else if(ring[0] == p.second)
    {
        ring.push_front(p.first);
        return true;
    }
    else if(ring[0] == p.first)
    {
        ring.push_front(p.second);
        return true;
    }
    return false;
}

/// Add an element to the ring only if it does not already exists
/// @return true if already exists
static bool add_to_ring(std::deque<int>& ring, int neigh)
{
    std::deque<int>::iterator it;
    for(it = ring.begin(); it != ring.end(); ++it)
        if(*it == neigh) return true;

    ring.push_back( neigh );
    return false;
}

static std::vector<std::vector<Triplet>> get_normalized_laplacian(at::Tensor vertices, const std::vector< std::vector<int> >& edges ) {
    unsigned nv = vertices.size(0);
    std::vector<std::vector<Triplet>> mat_elemts(nv);
    for(int i = 0; i < nv; ++i)
        mat_elemts[i].reserve(10);

    for(int i = 0; i < nv; ++i) {
        const at::Tensor c_pos = vertices[i];

        //get laplacian
        double sum = 0.;
        int nb_edges = edges[i].size();
        for(int e = 0; e < nb_edges; ++e)
        {
            int next_edge = (e + 1           ) % nb_edges;
            int prev_edge = (e + nb_edges - 1) % nb_edges;

            at::Tensor v1 = c_pos                 - vertices[edges[i][prev_edge]];
            at::Tensor v2 = vertices[edges[i][e]] - vertices[edges[i][prev_edge]];
            at::Tensor v3 = c_pos                 - vertices[edges[i][next_edge]];
            at::Tensor v4 = vertices[edges[i][e]] - vertices[edges[i][next_edge]];

            double cotan1 = ((v1.dot(v2)) / (1e-6 + (v1.cross(v2)).norm().item<float>() )).item<float>();
            double cotan2 = ((v3.dot(v4)) / (1e-6 + (v3.cross(v4)).norm().item<float>() )).item<float>();

            double w = (cotan1 + cotan2)*0.5;
            sum += w;
            mat_elemts[i].push_back( Triplet(i, edges[i][e], w) );
        }

        for( Triplet& t : mat_elemts[i] )
            t = Triplet( t.row(), t.col(), t.value() / sum);

        mat_elemts[i].push_back( Triplet(i, i, -1.0) );
    }
    return mat_elemts;
}

void implicit_laplacian_smoothing_impl(at::Tensor vertices, at::Tensor faces, float alpha, at::Tensor new_vertices) {
    const int N = vertices.size(0);
    const int M = faces.size(0);

    // vertex_to_face
    std::vector<std::vector<int> > _1st_ring_tris(N);
    std::vector<bool> _is_vertex_connected(N, false);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < 3; ++j) {
            int v = faces[i][j].item<int>();
            assert(v >= 0);
            _1st_ring_tris[v].push_back(i);
            _is_vertex_connected[v] = true;
        }
    }
    //vertex_to_1st_ring_vertices_compute
    bool _is_mesh_closed = true;
    bool _is_mesh_manifold = true;
    std::vector<int> _not_manifold_verts;
    std::vector<int> _on_side_verts;

    std::vector<bool> _is_vert_on_side(N);
    std::vector< std::vector<int> > _rings_per_vertex(N);

    std::vector<std::pair<int, int> > list_pairs;
    list_pairs.reserve(16);

    for (int i = 0; i < N; ++i) {
        if(_1st_ring_tris[i].size() > 0)
            _rings_per_vertex[i].reserve(_1st_ring_tris[i].size());

        if( !_is_vertex_connected[i] )
            continue;

        list_pairs.clear();
        // fill pairs with the first ring of neighborhood of triangles
        for(unsigned j = 0; j < _1st_ring_tris[i].size(); j++)
            list_pairs.push_back(pair_from_tri(faces.data_ptr<int>() + _1st_ring_tris[i][j] * 3, i));

        // Try to build the ordered list of the first ring of neighborhood of i
        std::deque<int> ring;
        ring.push_back(list_pairs[0].first );
        ring.push_back(list_pairs[0].second);
        std::vector<std::pair<int, int> >::iterator it = list_pairs.begin();
        list_pairs.erase(it);
        size_t  pairs_left = list_pairs.size();
        bool manifold = true;
        while( (pairs_left = list_pairs.size()) != 0)
        {
            for(it = list_pairs.begin(); it < list_pairs.end(); ++it)
            {
                if(add_to_ring(ring, *it)) {
                    list_pairs.erase(it);
                    break;
                }
            }

            if(pairs_left == list_pairs.size()) {
                // Not manifold we push neighborhoods of vert 'i'
                // in a random order
                add_to_ring(ring, list_pairs[0].first );
                add_to_ring(ring, list_pairs[0].second);
                list_pairs.erase(list_pairs.begin());
                manifold = false;
            }
        }

        if(!manifold) {
            _is_mesh_manifold = false;
//                DEBUG(i);
            _not_manifold_verts.push_back(i);
        }

        if(ring[0] != ring[ring.size()-1]){
            _is_vert_on_side[i] = true;
            _on_side_verts.push_back(i);
            _is_mesh_closed = false;
        } else {
            _is_vert_on_side[i] = false;
            ring.pop_back();
        }

        for(unsigned int j = 0; j < ring.size(); j++)
            _rings_per_vertex[i].push_back( ring[j] );
    }
//        DEBUG(_not_manifold_verts.size());
//        DEBUG(_on_side_verts.size());
    //smoothing
    std::vector<Eigen::VectorXd> xyz;
    std::vector<Eigen::VectorXd> rhs;

    xyz.resize(3, Eigen::VectorXd::Zero(N));
    rhs.resize(3, Eigen::VectorXd::Zero(N));

    for(int i = 0; i < N; ++i) {
        rhs[0][i] = vertices[i][0].item<float>();
        rhs[1][i] = vertices[i][1].item<float>();
        rhs[2][i] = vertices[i][2].item<float>();
    }

    // Build laplacian
    std::vector<std::vector<Triplet>> mat_elemts = get_normalized_laplacian(vertices, _rings_per_vertex);

    Eigen::SparseMatrix<double> L(N, N);
    std::vector<Triplet> triplets;
    triplets.reserve(N * 10);
    for( const std::vector<Triplet>& row : mat_elemts)
        for( const Triplet& elt : row )
            triplets.push_back( elt );

    L.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseMatrix<double> I = Eigen::MatrixXd::Identity(N, N).sparseView();
    L = I - L*alpha;

    L = L*L*L;

    // Solve for x, y, z
    Eigen::SparseLU<Sparse_mat> solver;
    solver.compute( L );

    for(int k = 0; k < 3; k++){
        xyz[k] = solver.solve(rhs[k]);
    }

    for(int i = 0; i < N; ++i){
        new_vertices[i][0] = xyz[0][i];
        new_vertices[i][1] = xyz[1][i];
        new_vertices[i][2] = xyz[2][i];
    }
}