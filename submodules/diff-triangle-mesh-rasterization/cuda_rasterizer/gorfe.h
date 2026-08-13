/*
 * Exact sparse-design replay used by the GoRFE-V1 experiment.
 *
 * This is deliberately separate from the renderer.  It consumes the immutable
 * state produced by one completed forward pass and cannot alter that pass.
 */

#ifndef CUDA_RASTERIZER_GORFE_H_INCLUDED
#define CUDA_RASTERIZER_GORFE_H_INCLUDED

#include <cstdint>
#include <cuda_runtime.h>

namespace GORFE
{
	constexpr int DIAGNOSTIC_COUNT = 12;

	void countRows(
		int W, int H,
		int output_W, int output_H,
		const uint2* ranges,
		const uint32_t* point_list,
		const float2* normals,
		const float* offsets,
		const float4* conic_opacity,
		const float2* phi_center,
		const float2* p_image,
		const int* triangles_indices,
		const int* face_edge_ids,
		float sigma,
		const float* final_T,
		const uint32_t* n_contrib,
		int32_t* row_counts,
		int64_t* diagnostics,
		cudaStream_t stream);

	void writeRows(
		int W, int H,
		int output_W, int output_H,
		const uint2* ranges,
		const uint32_t* point_list,
		const float2* normals,
		const float* offsets,
		const float4* conic_opacity,
		const float2* phi_center,
		const float2* p_image,
		const int* triangles_indices,
		const int* face_edge_ids,
		const float* vertex_sh1,
		float sigma,
		const int32_t* row_counts,
		const int32_t* inclusive_offsets,
		int64_t row_capacity,
		int32_t* low_pixel_ids,
		int32_t* group_ids,
		float* features,
		int64_t* diagnostics,
		cudaStream_t stream);
}

#endif
