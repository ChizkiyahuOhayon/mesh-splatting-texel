/*
 * Exact forward-order sparse-design replay for GoRFE-V1.
 *
 * Each CUDA thread replays one high-resolution pixel's sorted triangle list.
 * The count and write passes instantiate the same templated device routine, so
 * their row predicates and arithmetic cannot drift apart.  Rows are emitted in
 * (high-resolution pixel, forward contributor, local edge) order.  Reduction
 * over equal (low pixel, group) keys intentionally remains outside this module.
 */

#include "gorfe.h"

#include "auxiliary.h"
#include "config.h"

#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

namespace
{
	enum Diagnostic : int
	{
		RAW_ROWS = 0,
		COUNT_ALPHA_ACCEPTED = 1,
		COUNT_BLENDED = 2,
		WRITE_ALPHA_ACCEPTED = 3,
		WRITE_BLENDED = 4,
		FINAL_T_MISMATCH_PIXELS = 5,
		LAST_CONTRIBUTOR_MISMATCH_PIXELS = 6,
		COUNT_WRITE_MISMATCH_PIXELS = 7,
		WRITE_OVERFLOW_ROWS = 8,
		HIGH_PIXELS = 9,
		OUTPUT_PIXELS = 10,
		REPLAY_PASSES = 11,
	};

	__device__ inline void addDiagnostic(int64_t* diagnostics, int index, uint64_t value = 1)
	{
		atomicAdd(
			reinterpret_cast<unsigned long long*>(diagnostics + index),
			static_cast<unsigned long long>(value));
	}

	template <bool WRITE>
	__global__ void replayCUDA(
		int W, int H,
		int output_W, int output_H,
		const uint2* __restrict__ ranges,
		const uint32_t* __restrict__ point_list,
		const float2* __restrict__ normals,
		const float* __restrict__ offsets,
		const float4* __restrict__ conic_opacity,
		const float2* __restrict__ phi_center,
		const float2* __restrict__ p_image,
		const int* __restrict__ triangles_indices,
		const int* __restrict__ face_edge_ids,
		const float* __restrict__ vertex_sh1,
		float sigma,
		const float* __restrict__ saved_final_T,
		const uint32_t* __restrict__ saved_n_contrib,
		int32_t* __restrict__ row_counts,
		const int32_t* __restrict__ inclusive_offsets,
		int64_t row_capacity,
		int32_t* __restrict__ low_pixel_ids,
		int32_t* __restrict__ group_ids,
		float* __restrict__ features,
		int64_t* __restrict__ diagnostics)
	{
		const uint32_t pix_id = cg::this_grid().thread_rank();
		const uint32_t pixel_count = static_cast<uint32_t>(W) * static_cast<uint32_t>(H);
		if (pix_id >= pixel_count)
			return;

		const uint32_t pix_x = pix_id % W;
		const uint32_t pix_y = pix_id / W;
		const float2 pixf = {static_cast<float>(pix_x), static_cast<float>(pix_y)};
		const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		const uint32_t tile_x = pix_x / BLOCK_X;
		const uint32_t tile_y = pix_y / BLOCK_Y;
		const uint2 range = ranges[tile_y * horizontal_blocks + tile_x];

		float T = 1.0f;
		uint32_t contributor = 0;
		uint32_t last_contributor = 0;
		int64_t local_rows = 0;
		uint64_t alpha_accepted = 0;
		uint64_t blended = 0;
		bool done = false;

		for (uint32_t cursor = range.x; cursor < range.y && !done; ++cursor)
		{
			++contributor;
			const int face_id = static_cast<int>(point_list[cursor]);
			const float4 con_o = conic_opacity[face_id];
			float max_val = -INFINITY;
			bool outside = false;
			for (int k = 0; k < 3; ++k)
			{
				const float2 normal = normals[3 * face_id + k];
				const float dist = normal.x * pixf.x
					+ normal.y * pixf.y
					+ offsets[3 * face_id + k];
				if (dist > 0)
				{
					outside = true;
					break;
				}
				max_val = fmaxf(max_val, dist);
			}
			if (outside)
				continue;

			const float phi_final = max_val * phi_center[face_id].x;
			const float Cx = fmaxf(0.0f, __powf(phi_final, sigma));
			const float alpha = min(0.999f, con_o.w * Cx);
			if (alpha < 1.0f / 255.0f)
				continue;

			++alpha_accepted;
			const float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			++blended;
			const float blending_weight = alpha * T;
			const int edge_id_0 = face_edge_ids[3 * face_id + 0];
			const int edge_id_1 = face_edge_ids[3 * face_id + 1];
			const int edge_id_2 = face_edge_ids[3 * face_id + 2];
			if (edge_id_0 >= 0 || edge_id_1 >= 0 || edge_id_2 >= 0)
			{
				const float2 uv0 = p_image[3 * face_id + 0];
				const float2 uv1 = p_image[3 * face_id + 1];
				const float2 uv2 = p_image[3 * face_id + 2];
				const float2 v0 = {uv1.x - uv0.x, uv1.y - uv0.y};
				const float2 v1 = {uv2.x - uv0.x, uv2.y - uv0.y};
				const float2 v2 = {pixf.x - uv0.x, pixf.y - uv0.y};
				const float denom = v0.x * v1.y - v1.x * v0.y;
				const float invDen = 1.0f / denom;
				const float b0 = (v2.x * v1.y - v1.x * v2.y) * invDen;
				const float b1 = (-v2.x * v0.y + v0.x * v2.y) * invDen;
				const float b2 = 1.0f - b0 - b1;
				const float wA = b2;
				const float wB = b0;
				const float wC = b1;
				const int edge_ids[3] = {edge_id_0, edge_id_1, edge_id_2};
				const float edge_bases[3] = {
					4.0f * wA * wB,
					4.0f * wB * wC,
					4.0f * wC * wA,
				};

				float sh1_interp[3] = {0.0f, 0.0f, 0.0f};
				if (WRITE)
				{
					const int vertex_idx0 = triangles_indices[3 * face_id + 0];
					const int vertex_idx1 = triangles_indices[3 * face_id + 1];
					const int vertex_idx2 = triangles_indices[3 * face_id + 2];
					for (int k = 0; k < 3; ++k)
					{
						sh1_interp[k] = wA * vertex_sh1[3 * vertex_idx0 + k]
							+ wB * vertex_sh1[3 * vertex_idx1 + k]
							+ wC * vertex_sh1[3 * vertex_idx2 + k];
					}
				}

				for (int local_edge = 0; local_edge < 3; ++local_edge)
				{
					const int group_id = edge_ids[local_edge];
					const float edge_basis = edge_bases[local_edge];
					if (group_id < 0 || edge_basis == 0.0f)
						continue;

					if (WRITE)
					{
						const int64_t row_start = static_cast<int64_t>(inclusive_offsets[pix_id])
							- static_cast<int64_t>(row_counts[pix_id]);
						const int64_t row = row_start + local_rows;
						if (row < 0 || row >= row_capacity)
						{
							addDiagnostic(diagnostics, WRITE_OVERFLOW_ROWS);
						}
						else
						{
							const int32_t low_pixel = static_cast<int32_t>(
								(pix_y / 4) * output_W + (pix_x / 4));
							const float dc = blending_weight * edge_basis * (1.0f / 16.0f);
							low_pixel_ids[row] = low_pixel;
							group_ids[row] = group_id;
							features[4 * row + 0] = dc;
							features[4 * row + 1] = dc * sh1_interp[0];
							features[4 * row + 2] = dc * sh1_interp[1];
							features[4 * row + 3] = dc * sh1_interp[2];
						}
					}
					++local_rows;
				}
			}

			T = test_T;
			last_contributor = contributor;
		}

		if (WRITE)
		{
			if (local_rows != static_cast<int64_t>(row_counts[pix_id]))
				addDiagnostic(diagnostics, COUNT_WRITE_MISMATCH_PIXELS);
			if (alpha_accepted > 0)
				addDiagnostic(diagnostics, WRITE_ALPHA_ACCEPTED, alpha_accepted);
			if (blended > 0)
				addDiagnostic(diagnostics, WRITE_BLENDED, blended);
		}
		else
		{
			row_counts[pix_id] = local_rows <= 0x7fffffffLL
				? static_cast<int32_t>(local_rows)
				: -1;
			if (alpha_accepted > 0)
				addDiagnostic(diagnostics, COUNT_ALPHA_ACCEPTED, alpha_accepted);
			if (blended > 0)
				addDiagnostic(diagnostics, COUNT_BLENDED, blended);
			if (__float_as_uint(T) != __float_as_uint(saved_final_T[pix_id]))
				addDiagnostic(diagnostics, FINAL_T_MISMATCH_PIXELS);
			if (last_contributor != saved_n_contrib[pix_id])
				addDiagnostic(diagnostics, LAST_CONTRIBUTOR_MISMATCH_PIXELS);
		}
	}
}

void GORFE::countRows(
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
	cudaStream_t stream)
{
	const int64_t pixels = static_cast<int64_t>(W) * H;
	replayCUDA<false><<<(pixels + 255) / 256, 256, 0, stream>>>(
		W, H, output_W, output_H,
		ranges, point_list, normals, offsets, conic_opacity, phi_center,
		p_image, triangles_indices, face_edge_ids, nullptr, sigma,
		final_T, n_contrib, row_counts, nullptr, 0, nullptr, nullptr, nullptr,
		diagnostics);
}

void GORFE::writeRows(
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
	cudaStream_t stream)
{
	const int64_t pixels = static_cast<int64_t>(W) * H;
	replayCUDA<true><<<(pixels + 255) / 256, 256, 0, stream>>>(
		W, H, output_W, output_H,
		ranges, point_list, normals, offsets, conic_opacity, phi_center,
		p_image, triangles_indices, face_edge_ids, vertex_sh1, sigma,
		nullptr, nullptr, const_cast<int32_t*>(row_counts), inclusive_offsets,
		row_capacity, low_pixel_ids, group_ids, features, diagnostics);
}
