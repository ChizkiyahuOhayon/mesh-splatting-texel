/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/gorfe.h"
#include <fstream>
#include <string>
#include <functional>
#include <array>
#include <cmath>
#include <limits>
#include "cuda_rasterizer/utils.h"
#include "cuda_rasterizer/adam.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizetrianglesCUDA(
	const torch::Tensor& background,
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const torch::Tensor& vertex_weights,
	const float sigma,
    const torch::Tensor& colors,
	const torch::Tensor& texels,
	const int texel_order,
	const torch::Tensor& edge_details,
	const torch::Tensor& face_edge_ids,
	const torch::Tensor& window_source,
	const torch::Tensor& donor_indices,
	const int donor_mode,
	torch::Tensor& scaling,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{

  const int P = triangles_indices.size(0);
  if (donor_mode != 0) {
    TORCH_CHECK(donor_mode >= 1 && donor_mode <= 7, "donor_mode must be a bitmask in [1, 7], got ", donor_mode);
    TORCH_CHECK(window_source.dim() == 1 && window_source.size(0) == P,
                "window_source must be [F] with F = ", P, ", got ", window_source.sizes());
    TORCH_CHECK(donor_indices.dim() == 2 && donor_indices.size(1) == 3,
                "donor_indices must be [D, 3], got ", donor_indices.sizes());
    TORCH_CHECK(window_source.scalar_type() == torch::kInt32 && donor_indices.scalar_type() == torch::kInt32,
                "window_source and donor_indices must be int32");
    TORCH_CHECK(window_source.is_cuda() && window_source.is_contiguous(), "window_source must be contiguous CUDA");
    TORCH_CHECK(donor_indices.is_cuda() && donor_indices.is_contiguous(), "donor_indices must be contiguous CUDA");
  }
  if (texel_order > 0) {
    TORCH_CHECK(texels.dim() == 3, "texels must be [F, order*order, 3], got dim ", texels.dim());
    TORCH_CHECK(texels.size(0) == P, "texel face count ", texels.size(0), " != triangle count ", P);
    TORCH_CHECK(texels.size(1) == texel_order * texel_order,
                "texels slot dim ", texels.size(1), " != texel_order^2 ", texel_order * texel_order);
    TORCH_CHECK(texels.size(2) == 3, "texels must have 3 channels, got ", texels.size(2));
    TORCH_CHECK(texels.is_cuda() && texels.is_contiguous(), "texels must be contiguous CUDA");
    TORCH_CHECK(texels.scalar_type() == torch::kFloat32, "texels must be float32");
  }
  const bool edge_details_enabled = edge_details.numel() > 0 || face_edge_ids.numel() > 0;
  int edge_detail_dim = 0;
  if (edge_details_enabled) {
    TORCH_CHECK(P > 0, "edge_details require at least one triangle");
    const bool dc_shape = edge_details.dim() == 2 && edge_details.size(1) == NUM_CHANNELS;
    const bool dc_sh1_shape = edge_details.dim() == 3
        && edge_details.size(1) == 4 && edge_details.size(2) == NUM_CHANNELS;
    TORCH_CHECK(dc_shape || dc_sh1_shape,
                "edge_details must be [E, 3] or [E, 4, 3], got ", edge_details.sizes());
    edge_detail_dim = dc_sh1_shape ? 4 : 1;
    TORCH_CHECK(edge_details.size(0) > 0,
                "edge_details must contain at least one row when enabled");
    TORCH_CHECK(face_edge_ids.dim() == 2 && face_edge_ids.size(0) == P && face_edge_ids.size(1) == 3,
                "face_edge_ids must be [F, 3] with F = ", P, ", got ", face_edge_ids.sizes());
    TORCH_CHECK(edge_details.is_cuda() && edge_details.is_contiguous(),
                "edge_details must be contiguous CUDA");
    TORCH_CHECK(face_edge_ids.is_cuda() && face_edge_ids.is_contiguous(),
                "face_edge_ids must be contiguous CUDA");
    TORCH_CHECK(edge_details.scalar_type() == torch::kFloat32,
                "edge_details must be float32");
    TORCH_CHECK(face_edge_ids.scalar_type() == torch::kInt32,
                "face_edge_ids must be int32");
    TORCH_CHECK(donor_mode == 0,
                "edge_details cannot be combined with RITS window donors");
    const int min_edge_id = face_edge_ids.min().item<int>();
    const int max_edge_id = face_edge_ids.max().item<int>();
    TORCH_CHECK(min_edge_id >= -1 && max_edge_id < edge_details.size(0),
                "face_edge_ids values must lie in [-1, E), got [", min_edge_id,
                ", ", max_edge_id, "] for E = ", edge_details.size(0));
  }
  const int V = vertices.size(0); 
  const int H = image_height;
  const int W = image_width;

  auto int_opts = vertices.options().dtype(torch::kInt32);
  auto float_opts = vertices.options().dtype(torch::kFloat32);
  torch::Tensor edge_sh1 = edge_detail_dim == 4
      ? torch::empty({V, 3}, float_opts)
      : torch::empty({0}, float_opts);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, vertices.options().dtype(torch::kInt32));

  torch::Tensor proba_existence = torch::full({P}, 0.0, float_opts);
  torch::Tensor was_rendered = torch::full({P}, 0, vertices.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor max_blending = torch::full({P}, 0.0, float_opts);

  const int total_nb_points = P * 3; // FOR EACH TRIANGLE, WE CAN HAVE 3 NORMALS, OFFSETS,...
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, V, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		vertices.contiguous().data_ptr<float>(),
		triangles_indices.contiguous().data_ptr<int>(),
		vertex_weights.contiguous().data_ptr<float>(),
		sigma,
		total_nb_points,
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		texel_order > 0 ? texels.contiguous().data_ptr<float>() : nullptr,
		texel_order,
		edge_details_enabled ? edge_details.contiguous().data_ptr<float>() : nullptr,
		edge_detail_dim,
		edge_detail_dim == 4 ? edge_sh1.contiguous().data_ptr<float>() : nullptr,
		edge_details_enabled ? face_edge_ids.contiguous().data_ptr<int>() : nullptr,
		donor_mode != 0 ? window_source.contiguous().data_ptr<int>() : nullptr,
		donor_mode != 0 ? donor_indices.contiguous().data_ptr<int>() : nullptr,
		donor_mode,
		scaling.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		out_others.contiguous().data_ptr<float>(),
		max_blending.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		was_rendered.contiguous().data_ptr<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ExportGoRFERowsCUDA(
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const float sigma,
	const torch::Tensor& face_edge_ids,
	const int64_t edge_count,
	const int image_height,
	const int image_width,
	const int output_height,
	const int output_width,
	const int output_scaling,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int64_t num_rendered,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug)
{
	TORCH_CHECK(std::isfinite(sigma), "GoRFE sigma must be finite, got ", sigma);
	TORCH_CHECK(vertices.is_cuda() && vertices.is_contiguous(),
		"vertices must be contiguous CUDA");
	TORCH_CHECK(vertices.scalar_type() == torch::kFloat32
		&& vertices.dim() == 2 && vertices.size(1) == 3,
		"vertices must be float32 [V, 3], got ", vertices.sizes());
	TORCH_CHECK(triangles_indices.is_cuda() && triangles_indices.is_contiguous(),
		"triangles_indices must be contiguous CUDA");
	TORCH_CHECK(triangles_indices.scalar_type() == torch::kInt32
		&& triangles_indices.dim() == 2 && triangles_indices.size(1) == 3,
		"triangles_indices must be int32 [F, 3], got ", triangles_indices.sizes());
	const int64_t P64 = triangles_indices.size(0);
	const int64_t V64 = vertices.size(0);
	TORCH_CHECK(P64 > 0 && P64 <= std::numeric_limits<int>::max() / 3,
		"GoRFE export requires 1..INT_MAX faces, got ", P64);
	TORCH_CHECK(V64 > 0 && V64 <= std::numeric_limits<int>::max(),
		"GoRFE export requires 1..INT_MAX vertices, got ", V64);
	const int P = static_cast<int>(P64);
	const int V = static_cast<int>(V64);

	TORCH_CHECK(face_edge_ids.is_cuda() && face_edge_ids.is_contiguous(),
		"GoRFE face_edge_ids must be contiguous CUDA");
	TORCH_CHECK(face_edge_ids.scalar_type() == torch::kInt32
		&& face_edge_ids.dim() == 2
		&& face_edge_ids.size(0) == P && face_edge_ids.size(1) == 3,
		"GoRFE face_edge_ids must be int32 [F, 3] with F = ", P,
		", got ", face_edge_ids.sizes());
	TORCH_CHECK(edge_count >= 0 && edge_count <= std::numeric_limits<int>::max(),
		"GoRFE edge_count must be in [0, INT_MAX], got ", edge_count);
	const int min_edge_id = face_edge_ids.min().item<int>();
	const int max_edge_id = face_edge_ids.max().item<int>();
	TORCH_CHECK(min_edge_id >= -1 && max_edge_id < edge_count,
		"GoRFE face_edge_ids values must lie in [-1, edge_count), got [",
		min_edge_id, ", ", max_edge_id, "] for edge_count = ", edge_count);

	TORCH_CHECK(image_height > 0 && image_width > 0
		&& output_height > 0 && output_width > 0,
		"GoRFE image dimensions must be positive");
	TORCH_CHECK(output_scaling == 4,
		"GoRFE-V1 output_scaling must equal 4, got ", output_scaling);
	TORCH_CHECK(static_cast<int64_t>(image_height)
			== static_cast<int64_t>(output_height) * output_scaling
		&& static_cast<int64_t>(image_width)
			== static_cast<int64_t>(output_width) * output_scaling,
		"GoRFE high-resolution dimensions must be exactly 4x output: got ",
		image_height, "x", image_width, " and ", output_height, "x", output_width);
	const int64_t high_pixels = static_cast<int64_t>(image_height) * image_width;
	const int64_t output_pixels = static_cast<int64_t>(output_height) * output_width;
	TORCH_CHECK(high_pixels > 0 && high_pixels <= std::numeric_limits<uint32_t>::max(),
		"GoRFE high-resolution pixel count exceeds uint32 range: ", high_pixels);
	TORCH_CHECK(output_pixels > 0 && output_pixels < (int64_t{1} << 31),
		"GoRFE output pixel count must be below 2^31, got ", output_pixels);

	TORCH_CHECK(campos.is_cuda() && campos.is_contiguous()
		&& campos.scalar_type() == torch::kFloat32 && campos.numel() == 3,
		"campos must be contiguous CUDA float32 with three elements");
	TORCH_CHECK(vertices.get_device() == triangles_indices.get_device()
		&& vertices.get_device() == face_edge_ids.get_device()
		&& vertices.get_device() == campos.get_device(),
		"all GoRFE tensor inputs must be on the same CUDA device");

	auto check_buffer = [&](const torch::Tensor& buffer, const char* name) {
		TORCH_CHECK(buffer.is_cuda() && buffer.is_contiguous()
			&& buffer.scalar_type() == torch::kUInt8,
			name, " must be a contiguous CUDA uint8 forward buffer");
		TORCH_CHECK(buffer.get_device() == vertices.get_device(),
			name, " must be on the same CUDA device as vertices");
	};
	check_buffer(geomBuffer, "geomBuffer");
	check_buffer(binningBuffer, "binningBuffer");
	check_buffer(imageBuffer, "imageBuffer");
	TORCH_CHECK(num_rendered >= 0 && num_rendered <= std::numeric_limits<int>::max(),
		"num_rendered must be in [0, INT_MAX], got ", num_rendered);
	c10::cuda::CUDAGuard device_guard(vertices.device());
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device());

	const int total_nb_points = P * 3;
	const size_t expected_geom = CudaRasterizer::required<CudaRasterizer::GeometryState>(
		P, total_nb_points, V, false);
	const size_t expected_binning = CudaRasterizer::required<CudaRasterizer::BinningState>(
		num_rendered, 0, 0, false);
	const size_t expected_image = CudaRasterizer::required<CudaRasterizer::ImageState>(
		high_pixels, 0, 0, false);
	TORCH_CHECK(static_cast<size_t>(geomBuffer.numel()) == expected_geom,
		"geomBuffer size does not match the declared forward state");
	TORCH_CHECK(static_cast<size_t>(binningBuffer.numel()) == expected_binning,
		"binningBuffer size does not match the declared forward state");
	TORCH_CHECK(static_cast<size_t>(imageBuffer.numel()) == expected_image,
		"imageBuffer size does not match the declared forward state");

	char* geom_chunk = reinterpret_cast<char*>(geomBuffer.data_ptr<uint8_t>());
	char* binning_chunk = reinterpret_cast<char*>(binningBuffer.data_ptr<uint8_t>());
	char* image_chunk = reinterpret_cast<char*>(imageBuffer.data_ptr<uint8_t>());
	auto geom = CudaRasterizer::GeometryState::fromChunk(
		geom_chunk, P, total_nb_points, V, false);
	auto binning = CudaRasterizer::BinningState::fromChunk(
		binning_chunk, static_cast<size_t>(num_rendered));
	auto image = CudaRasterizer::ImageState::fromChunk(
		image_chunk, static_cast<size_t>(high_pixels));

	auto int_options = vertices.options().dtype(torch::kInt32);
	auto long_options = vertices.options().dtype(torch::kInt64);
	auto float_options = vertices.options().dtype(torch::kFloat32);
	torch::Tensor row_counts = torch::zeros({high_pixels}, int_options);
	torch::Tensor diagnostics = torch::zeros({GORFE::DIAGNOSTIC_COUNT}, long_options);

	// The legacy renderer launches its forward kernels on the CUDA default
	// stream.  The public API calls this routine immediately after that forward;
	// synchronize once so the opaque state is immutable before replay starts.
	TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
		"GoRFE could not synchronize the completed forward state");
	GORFE::countRows(
		image_width, image_height, output_width, output_height,
		image.ranges, binning.point_list, geom.normals, geom.offsets,
		geom.conic_opacity, geom.phi_center, geom.p_image,
		triangles_indices.data_ptr<int>(), face_edge_ids.data_ptr<int>(), sigma,
		image.accum_alpha, image.n_contrib, row_counts.data_ptr<int32_t>(),
		diagnostics.data_ptr<int64_t>(), stream);
	cudaError_t launch_error = cudaGetLastError();
	TORCH_CHECK(launch_error == cudaSuccess,
		"GoRFE count kernel launch failed: ", cudaGetErrorString(launch_error));

	const int32_t minimum_pixel_rows = row_counts.min().item<int32_t>();
	TORCH_CHECK(minimum_pixel_rows >= 0,
		"GoRFE per-pixel row count overflowed int32");
	const int64_t raw_rows = row_counts.sum(torch::kInt64).item<int64_t>();
	TORCH_CHECK(raw_rows >= 0 && raw_rows < (int64_t{1} << 31),
		"GoRFE raw row count must be below 2^31, got ", raw_rows);
	torch::Tensor inclusive_offsets = torch::cumsum(row_counts, 0, torch::kInt32);
	if (high_pixels > 0)
	{
		const int32_t scanned_rows = inclusive_offsets[high_pixels - 1].item<int32_t>();
		TORCH_CHECK(static_cast<int64_t>(scanned_rows) == raw_rows,
			"GoRFE int32 scan total disagrees with int64 row total: ",
			scanned_rows, " versus ", raw_rows);
	}

	torch::Tensor low_pixel_ids = torch::empty({raw_rows}, int_options);
	torch::Tensor group_ids = torch::empty({raw_rows}, int_options);
	torch::Tensor features = torch::empty({raw_rows, 4}, float_options);
	torch::Tensor vertex_sh1 = torch::empty({V, 3}, float_options);
	FORWARD::computeVertexSH1Factors(
		V, vertices.data_ptr<float>(),
		reinterpret_cast<const glm::vec3*>(campos.data_ptr<float>()),
		vertex_sh1.data_ptr<float>());
	launch_error = cudaGetLastError();
	TORCH_CHECK(launch_error == cudaSuccess,
		"GoRFE SH1-factor kernel launch failed: ", cudaGetErrorString(launch_error));
	// computeVertexSH1Factors is part of the stock renderer and launches on its
	// legacy default stream.  Complete it before the current-stream replay uses
	// the factors; this path is evaluation-only and prioritises exact ordering.
	TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
		"GoRFE SH1-factor computation failed during synchronization");

	GORFE::writeRows(
		image_width, image_height, output_width, output_height,
		image.ranges, binning.point_list, geom.normals, geom.offsets,
		geom.conic_opacity, geom.phi_center, geom.p_image,
		triangles_indices.data_ptr<int>(), face_edge_ids.data_ptr<int>(),
		vertex_sh1.data_ptr<float>(), sigma, row_counts.data_ptr<int32_t>(),
		inclusive_offsets.data_ptr<int32_t>(), raw_rows,
		low_pixel_ids.data_ptr<int32_t>(), group_ids.data_ptr<int32_t>(),
		features.data_ptr<float>(), diagnostics.data_ptr<int64_t>(), stream);
	launch_error = cudaGetLastError();
	TORCH_CHECK(launch_error == cudaSuccess,
		"GoRFE write kernel launch failed: ", cudaGetErrorString(launch_error));
	// The replay uses PyTorch's current (typically non-blocking) stream.  A
	// synchronous cudaMemcpy on the legacy default stream is not a cross-stream
	// dependency, so complete the write explicitly before reading diagnostics.
	TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
		"GoRFE write kernel failed during synchronization");

	std::array<int64_t, GORFE::DIAGNOSTIC_COUNT> host_diagnostics{};
	TORCH_CHECK(cudaMemcpy(
		host_diagnostics.data(), diagnostics.data_ptr<int64_t>(),
		sizeof(int64_t) * GORFE::DIAGNOSTIC_COUNT,
		cudaMemcpyDeviceToHost) == cudaSuccess,
		"failed to retrieve GoRFE replay diagnostics");
	host_diagnostics[0] = raw_rows;
	host_diagnostics[9] = high_pixels;
	host_diagnostics[10] = output_pixels;
	host_diagnostics[11] = 2;
	TORCH_CHECK(host_diagnostics[1] == host_diagnostics[3],
		"GoRFE count/write alpha-accepted counts disagree: ",
		host_diagnostics[1], " versus ", host_diagnostics[3]);
	TORCH_CHECK(host_diagnostics[2] == host_diagnostics[4],
		"GoRFE count/write blended-fragment counts disagree: ",
		host_diagnostics[2], " versus ", host_diagnostics[4]);
	TORCH_CHECK(host_diagnostics[5] == 0,
		"GoRFE replay transmittance disagrees with forward for ",
		host_diagnostics[5], " pixels");
	TORCH_CHECK(host_diagnostics[6] == 0,
		"GoRFE replay contributor count disagrees with forward for ",
		host_diagnostics[6], " pixels");
	TORCH_CHECK(host_diagnostics[7] == 0,
		"GoRFE count/write row counts disagree for ", host_diagnostics[7], " pixels");
	TORCH_CHECK(host_diagnostics[8] == 0,
		"GoRFE write attempted ", host_diagnostics[8], " rows outside its allocation");
	TORCH_CHECK(cudaMemcpy(
		diagnostics.data_ptr<int64_t>(), host_diagnostics.data(),
		sizeof(int64_t) * GORFE::DIAGNOSTIC_COUNT,
		cudaMemcpyHostToDevice) == cudaSuccess,
		"failed to finalize GoRFE replay diagnostics");

	if (debug)
		TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
			"GoRFE replay failed during debug synchronization");
	return std::make_tuple(low_pixel_ids, group_ids, features, diagnostics);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizetrianglesBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const torch::Tensor& vertex_weights,
    const float sigma,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& texels,
	const int texel_order,
	const torch::Tensor& edge_details,
	const torch::Tensor& face_edge_ids,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = triangles_indices.size(0); // number of triangles
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  const int V = vertices.size(0); // total number of vertices
  const int total_nb_points = P * 3;

  torch::Tensor dL_dvertices3D = torch::zeros({V, 3}, vertices.options());
  torch::Tensor dL_dvertice_weight = torch::zeros({V}, vertices.options());
  torch::Tensor dL_dpoints2D = torch::zeros({V, 2}, vertices.options());

  torch::Tensor dL_dnormals = torch::zeros({total_nb_points, 3}, vertices.options());
  torch::Tensor dL_doffsets = torch::zeros({total_nb_points, 3}, vertices.options());

  torch::Tensor dL_dcolors = torch::zeros({V, NUM_CHANNELS}, vertices.options());
  // Always allocate so the returned tuple has a stable shape; empty when disabled.
  torch::Tensor dL_dtexels = texel_order > 0
      ? torch::zeros_like(texels)
      : torch::zeros({0}, vertices.options());
  const bool edge_details_enabled = edge_details.numel() > 0 || face_edge_ids.numel() > 0;
  int edge_detail_dim = 0;
  if (edge_details_enabled) {
    TORCH_CHECK(P > 0, "edge_details require at least one triangle");
    const bool dc_shape = edge_details.dim() == 2 && edge_details.size(1) == NUM_CHANNELS;
    const bool dc_sh1_shape = edge_details.dim() == 3
        && edge_details.size(1) == 4 && edge_details.size(2) == NUM_CHANNELS;
    TORCH_CHECK(dc_shape || dc_sh1_shape,
                "edge_details must be [E, 3] or [E, 4, 3], got ", edge_details.sizes());
    edge_detail_dim = dc_sh1_shape ? 4 : 1;
    TORCH_CHECK(edge_details.size(0) > 0,
                "edge_details must contain at least one row when enabled");
    TORCH_CHECK(face_edge_ids.dim() == 2 && face_edge_ids.size(0) == P && face_edge_ids.size(1) == 3,
                "face_edge_ids must be [F, 3] with F = ", P, ", got ", face_edge_ids.sizes());
    TORCH_CHECK(edge_details.is_cuda() && edge_details.is_contiguous(),
                "edge_details must be contiguous CUDA");
    TORCH_CHECK(face_edge_ids.is_cuda() && face_edge_ids.is_contiguous(),
                "face_edge_ids must be contiguous CUDA");
    TORCH_CHECK(edge_details.scalar_type() == torch::kFloat32,
                "edge_details must be float32");
    TORCH_CHECK(face_edge_ids.scalar_type() == torch::kInt32,
                "face_edge_ids must be int32");
    const int min_edge_id = face_edge_ids.min().item<int>();
    const int max_edge_id = face_edge_ids.max().item<int>();
    TORCH_CHECK(min_edge_id >= -1 && max_edge_id < edge_details.size(0),
                "face_edge_ids values must lie in [-1, E), got [", min_edge_id,
                ", ", max_edge_id, "] for E = ", edge_details.size(0));
  }
  torch::Tensor dL_dedge_details = edge_details_enabled
      ? torch::zeros_like(edge_details)
      : torch::zeros({0}, vertices.options());
  torch::Tensor edge_sh1 = edge_detail_dim == 4
      ? torch::empty({V, 3}, vertices.options())
      : torch::empty({0}, vertices.options());
  torch::Tensor dL_dedge_sh1 = edge_detail_dim == 4
      ? torch::zeros({V, 3}, vertices.options())
      : torch::empty({0}, vertices.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, vertices.options());
  torch::Tensor dL_dsh = torch::zeros({V, M, 3}, vertices.options());

  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, vertices.options());
  torch::Tensor dL_dnormal3D = torch::zeros({P, 3}, vertices.options());

  torch::Tensor dL_dvertice_depth = torch::zeros({V}, vertices.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, V, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  vertices.contiguous().data_ptr<float>(),
	  triangles_indices.contiguous().data_ptr<int>(),
	  vertex_weights.contiguous().data_ptr<float>(),
	  sigma,
	  total_nb_points,
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  texel_order > 0 ? texels.contiguous().data_ptr<float>() : nullptr,
	  texel_order,
	  edge_details_enabled ? edge_details.contiguous().data_ptr<float>() : nullptr,
	  edge_detail_dim,
	  edge_detail_dim == 4 ? edge_sh1.contiguous().data_ptr<float>() : nullptr,
	  edge_details_enabled ? face_edge_ids.contiguous().data_ptr<int>() : nullptr,
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_others.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dnormal3D.contiguous().data_ptr<float>(),
	  dL_dvertices3D.contiguous().data_ptr<float>(),
	  dL_dvertice_weight.contiguous().data_ptr<float>(),
	  dL_dnormals.contiguous().data_ptr<float>(),
	  dL_doffsets.contiguous().data_ptr<float>(),
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  texel_order > 0 ? dL_dtexels.contiguous().data_ptr<float>() : nullptr,
	  edge_details_enabled ? dL_dedge_details.contiguous().data_ptr<float>() : nullptr,
	  edge_detail_dim == 4 ? dL_dedge_sh1.contiguous().data_ptr<float>() : nullptr,
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dpoints2D.contiguous().data_ptr<float>(),
	  dL_dvertice_depth.contiguous().data_ptr<float>(),
	  debug);
  }

  return std::make_tuple(dL_dvertices3D, dL_dvertice_weight, dL_dcolors, dL_dsh, dL_dtexels, dL_dedge_details);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}


std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}


void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}
