#version 460
#extension GL_NV_mesh_shader : require
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_shuffle_relative : enable

layout(max_vertices=4*gl_WorkGroupSize.x, max_primitives=2*gl_WorkGroupSize.x) out;
layout(triangles) out;

void main () {

}