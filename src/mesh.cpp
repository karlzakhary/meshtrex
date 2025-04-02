#include "common.h"

#include "mesh.h"

#include <fast_obj.h>

#include "meshoptimizer.h"

bool loadObj(std::vector<Vertex>& vertices, const char* path)
{
    fastObjMesh* obj = fast_obj_read(path);
    if (!obj)
        return false;

    size_t index_count = 0;

    for (unsigned int i = 0; i < obj->face_count; ++i)
        index_count += 3 * (obj->face_vertices[i] - 2);

    vertices.resize(index_count);

    size_t vertex_offset = 0;
    size_t index_offset = 0;

    for (unsigned int i = 0; i < obj->face_count; ++i)
    {
        for (unsigned int j = 0; j < obj->face_vertices[i]; ++j)
        {
            fastObjIndex gi = obj->indices[index_offset + j];

            if (j >= 3)
            {
                vertices[vertex_offset + 0] = vertices[vertex_offset - 3];
                vertices[vertex_offset + 1] = vertices[vertex_offset - 1];
                vertex_offset += 2;
            }

            Vertex& v = vertices[vertex_offset++];

            v.vx = obj->positions[gi.p * 3 + 0];
            v.vy = obj->positions[gi.p * 3 + 1];
            v.vz = obj->positions[gi.p * 3 + 2];
            v.nx = static_cast<uint8_t>(obj->normals[gi.n * 3 + 0] * 127.f + 127.5f);
            v.ny = static_cast<uint8_t>(obj->normals[gi.n * 3 + 1] * 127.f + 127.5f);
            v.nz = static_cast<uint8_t>(obj->normals[gi.n * 3 + 2] * 127.f + 127.5f);

            v.tu = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 0]);
            v.tv = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 1]);
        }

        index_offset += obj->face_vertices[i];
    }

    assert(vertex_offset == index_count);

    fast_obj_destroy(obj);

    return true;
}

bool loadMesh(Mesh& result, const char* path)
{
    std::vector<Vertex> vertices;
    if (!loadObj(vertices, path))
        return false;

    std::vector<uint32_t> indices(vertices.size());
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = static_cast<uint32_t>(i);

    result.indices = indices;
    result.vertices = vertices;

    return true;
}