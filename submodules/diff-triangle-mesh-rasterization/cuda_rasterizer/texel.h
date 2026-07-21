/*
 * Ptex-style per-face texel indexing.
 *
 * Appearance in MeshSplatting is carried by vertices (SH interpolated barycentrically),
 * which welds the spatial frequency of appearance to the tessellation density. A per-face
 * texel grid decouples the two: appearance detail is refined by adding texels, not
 * vertices. Unlike a global UV atlas it needs no unwrapping, and on export the per-face
 * texels pack into a standard texture atlas, so the result stays a plain textured mesh
 * with no custom fragment shader.
 *
 * Each triangle is subdivided barycentrically into order^2 texels. For barycentric
 * (a, b, c) with a + b + c = 1:
 *
 *     i = floor(a*L), j = floor(b*L), k = floor(c*L),  s = i + j + k
 *     s == L-1 -> upright sub-triangle,  s == L-2 -> inverted
 *     row r = L-1-i   (row r holds 2r+1 texels, offset r^2)
 *     slot  = r^2 + 2j + inverted
 *
 * Row sizes sum to L^2 and the map is area-uniform: every texel covers 1/L^2 of the
 * triangle, so "texels per face" is a fair measure of appearance resolution. Lookup is
 * nearest (piecewise constant), matching a GL_NEAREST texture fetch.
 *
 * NOTE on gradients: because the lookup is piecewise constant, the texel term has zero
 * derivative with respect to the barycentric coordinates almost everywhere. The
 * backward pass therefore leaves the existing barycentric/geometry gradient paths
 * completely untouched -- only a single extra atomicAdd on the texel itself is needed.
 *
 * This mirrors maclab/ptex.py, whose indexing is unit-tested in
 * maclab/tests/test_ptex.py (bijectivity and area-uniformity to <6%).
 */

#ifndef CUDA_RASTERIZER_TEXEL_H_INCLUDED
#define CUDA_RASTERIZER_TEXEL_H_INCLUDED

// Number of texels stored per face at the given order. order <= 0 disables the carrier.
__host__ __device__ __forceinline__ int texelSlots(int order)
{
	return order <= 0 ? 0 : order * order;
}

// Face-local texel index for barycentric weights (wA, wB, wC).
// Returns a value in [0, order^2).
__device__ __forceinline__ int texelSlot(float wA, float wB, float wC, int order)
{
	if (order <= 1)
		return 0;

	const float L = (float)order;
	// clamp guards against fp noise and the slight non-normalisation of screen-space
	// barycentrics at triangle edges
	int i = (int)floorf(fmaxf(wA, 0.0f) * L);
	int j = (int)floorf(fmaxf(wB, 0.0f) * L);
	int k = (int)floorf(fmaxf(wC, 0.0f) * L);
	i = min(max(i, 0), order - 1);
	j = min(max(j, 0), order - 1);
	k = min(max(k, 0), order - 1);

	const int s = i + j + k;
	const int inverted = (s <= order - 2) ? 1 : 0;
	int r = order - 1 - i;
	r = min(max(r, 0), order - 1);
	// upright needs j <= r, inverted needs j <= r-1
	const int jmax = max(r - inverted, 0);
	j = min(j, jmax);

	const int slot = r * r + 2 * j + inverted;
	return min(max(slot, 0), order * order - 1);
}

#endif
