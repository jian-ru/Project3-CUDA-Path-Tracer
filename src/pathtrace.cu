#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#define IMPLEMENT_DEVICE_FUNTIONS
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec4* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec4 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / pix.w * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / pix.w * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / pix.w * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec4 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

enum PathType
{
	Initial,
	Lambert,
	Mirror,
	Glass,
	Terminated,
	PathTypeCount
};
unsigned h_pathCountsPerType[PathTypeCount] = { 0 };
unsigned *dev_pathTypes = nullptr;
unsigned *dev_pathIndices = nullptr; // dev_pathIndices[i] contains the path index of the path that thread i operates on

const Geom **dev_lights = nullptr;
__constant__ int dev_numLights;

static Triangle *dev_triangles = nullptr;
static BVH::GpuBVH dev_bvh;


namespace MyUtilities
{
	template <typename T>
	__device__ __forceinline__ void swap(T &a, T &b)
	{
		T tmp(a);
		a = b;
		b = tmp;
	}
}


__global__ void kernInitLightPtrs(
	const Geom **dev_lights,
	const Geom *dev_geoms,
	const Material *dev_materials)
{
	int offset = 0;
	for (int i = 0; i < dev_numLights; ++i)
	{
		int matId = dev_geoms[i].materialid;
		if (dev_materials[matId].emittance > 0.f)
		{
			dev_lights[offset++] = &dev_geoms[i];
		}
	}
}

void initDeviceLightInfo(Scene *scene)
{
	const auto &h_mats = scene->materials;
	int numLights = 0;
	for (const auto &g : scene->geoms)
	{
		int matId = g.materialid;
		if (h_mats[matId].emittance > 0.f)
		{
			++numLights;
		}
	}

	cudaMemcpyToSymbol(dev_numLights, &numLights, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&dev_lights, numLights * sizeof(Geom *));
	kernInitLightPtrs<<<1, 1>>>(dev_lights, dev_geoms, dev_materials);
	checkCUDAError("pathtraceInit");
}

__global__ void kernInitPathIndices(size_t numPaths, unsigned *dev_pathIndices)
{
	unsigned pathIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (pathIdx < numPaths)
	{
		dev_pathIndices[pathIdx] = pathIdx;
	}
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec4));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec4));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	initDeviceLightInfo(scene);

	cudaMalloc(&dev_pathTypes, pixelcount * sizeof(unsigned));
	cudaMemset(dev_pathTypes, 0, pixelcount * sizeof(unsigned));

	cudaMalloc(&dev_pathIndices, pixelcount * sizeof(unsigned));

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	dev_bvh = scene->bvh->getGpuBuffers();
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_lights);
	cudaFree(dev_pathTypes);
	cudaFree(dev_pathIndices);
	cudaFree(dev_triangles);
	cudaFree(dev_bvh.d_nodes);
	cudaFree(dev_bvh.d_primimitives);

    checkCUDAError("pathtraceFree");
}

__device__ void printVec3(const glm::vec3 &v, const char *name)
{
	printf("%s = (%f, %f, %f)\n", name, v.x, v.y, v.z);
}

__device__ void initPathSegment(
	PathSegment &segment,
	int x,
	int y,
	int index,
	int traceDepth,
	const Camera &cam,
	thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	x += u01(rng);
	y += u01(rng);

	//segment.color = glm::vec3(1.0f, 1.0f, 1.0f); // IS initial
	segment.color = glm::vec3(0.0f, 0.0f, 0.0f); // MIS initial
	segment.misWeight = glm::vec3(1.f, 1.f, 1.f);

	glm::vec3 tmpd = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
		- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
	float ft = cam.focalDistance / glm::dot(cam.view, tmpd);
	glm::vec3 pFocus = cam.position + ft * tmpd;
	float lensU, lensV;
	ShapeSampling::concentricSamplingDisk(u01(rng), u01(rng), lensU, lensV);
	glm::vec3 tmpo = cam.position + lensU * cam.lensRadius * cam.right + lensV * cam.lensRadius * cam.up;
	segment.ray.origin = tmpo;
	segment.ray.direction = glm::normalize(pFocus - tmpo);

	segment.pixelIndex = index;
	segment.remainingBounces = traceDepth;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int curDepth, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y)
	{
		int index = x + (y * cam.resolution.x);
		PathSegment &segment = pathSegments[index];

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, curDepth);
		initPathSegment(segment, x, y, index, traceDepth, cam, rng);
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth, 
	int num_paths, 
	PathSegment * pathSegments, 
	unsigned *pathTypes,
	const unsigned *pathIndices,
	Geom * geoms, 
	int geoms_size, 
	ShadeableIntersection *intersections,
	const Material *materials,
	const Triangle *triangles,
	int numTriangles,
	BVH::GpuBVH bvh)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_paths)
	{
		if (pathTypes[tid] == Terminated)
		{
			return;
		}
		int path_index = pathIndices[tid];

		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 normal;
		int hit_geom_index = -1;

		if (geoms_size + numTriangles < MIN_OBJECTS_REQUIRED_FOR_BVH) // Brute force
		{
			SceneIntersection::getRaySceneIntersection(t, hit_geom_index, normal, pathSegment.ray, geoms, geoms_size, triangles, numTriangles);
		}
		else // BVH
		{
			BVH::getRaySceneIntersection(t, hit_geom_index, normal, pathSegment.ray, bvh, geoms, triangles);
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			pathSegments[path_index].remainingBounces = 0;
			//pathSegments[path_index].color *= 0.f; // IS
			pathTypes[tid] = Terminated;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;

			const Material &mat = materials[geoms[hit_geom_index].materialid];
			
			switch (mat.type)
			{
			case M_Lambert:
				pathTypes[tid] = Lambert;
				break;
			case M_Mirror:
				pathTypes[tid] = Mirror;
				break;
			case M_Glass:
				pathTypes[tid] = Glass;
				break;
			}
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterials(
	int depth,
	int iter, 
	int num_paths, 
	ShadeableIntersection * shadeableIntersections, 
	PathSegment * pathSegments, 
	unsigned *pathTypes,
	const unsigned *pathIndices,
	Material * materials,
	const Geom **dev_lights,
	const Geom *geoms,
	int numGeoms,
	const Triangle *triangles,
	int numTriangles,
	BVH::GpuBVH bvh
	)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_paths)
	{
		PathType pathType = static_cast<PathType>(pathTypes[tid]);

		//printf("(%d, %d, %d)\n", tid, pathType, pathIndices[tid]);
		//if (blockIdx.x == 0 && threadIdx.x == 0) printf("device: %d, %d\n", sizeof(PathSegment), (char*)(pathSegments+1) - (char*)pathSegments);

		if (pathType != Terminated)
		{
			int idx = pathIndices[tid];
			ShadeableIntersection intersection = shadeableIntersections[idx];
			const PathSegment &pathSegment = pathSegments[idx];

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			if (material.emittance > 0.0f)
			{
				//pathSegments[idx].color *= materialColor * material.emittance; // IS
				pathSegments[idx].color += pathSegment.misWeight * materialColor * material.emittance; // MIS
				pathSegments[idx].remainingBounces = 0;
				pathTypes[tid] = Terminated;
			}
			else if (pathType == Mirror)
			{
				// Perfect specular
				glm::vec3 isecPt = getPointOnRay(pathSegment.ray, intersection.t) + 1e-4f * intersection.surfaceNormal;
				glm::vec3 reflectDir = glm::normalize(glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal));
				pathSegments[idx].ray = { isecPt, reflectDir };
				pathSegments[idx].misWeight *= materialColor;
			}
			else if (pathType == Glass)
			{
				// Assume participating media is air
				float fresnel;
				float cosi, cost;
				float etai, etat;
				glm::vec3 reflectDir, refractDir;
				glm::vec3 isecPt;

				isecPt = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
				cosi = glm::dot(pathSegment.ray.direction, intersection.surfaceNormal);
				etai = 1.f;
				etat = material.indexOfRefraction;
				
				if (cosi > 0.f)
				{
					MyUtilities::swap(etai, etat);
					intersection.surfaceNormal = -intersection.surfaceNormal;
				}

				reflectDir = glm::normalize(glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal));
				refractDir = glm::normalize(glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, etai / etat));
				cost = glm::dot(refractDir, intersection.surfaceNormal);
				fresnel = Fresnel::frDiel(fabs(cosi), etai, fabs(cost), etat);

				float rn = u01(rng);
				if (rn < fresnel || glm::length2(refractDir) < FLT_EPSILON) // deal with total internal reflection
				{
					pathSegments[idx].ray = { isecPt + 1e-4f * intersection.surfaceNormal, reflectDir };
					pathSegments[idx].misWeight *= materialColor;
				}
				else
				{
					pathSegments[idx].ray = { isecPt - 1e-4f * intersection.surfaceNormal, refractDir };
					pathSegments[idx].misWeight *= materialColor;
				}
			}
			else
			{
				// Debug shading
				//float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				//pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not
				//return;

				// Lambertian surface (perfect diffuse)
				// 1. Get two rays by sampling BRDF and light source
				// 2. Compute two pdfs for the two rays
				// 3. Compute MIS weights using power heuristic
				// 4. Accumulate direct lighting contribution to pathSegments[idx].color if shadow feeler is not blocked
				// 5. Update throughput, a.k.a pathSegments[idx].misWeight
				const glm::vec3 lambertBrdf = materialColor / PI;

				int lightIdx = static_cast<int>(u01(rng) * dev_numLights);
				lightIdx = (lightIdx <= dev_numLights - 1) ? lightIdx : (dev_numLights - 1);
				const Geom &lightSrc = *dev_lights[lightIdx];
				glm::vec3 isecPt = getPointOnRay(pathSegment.ray, intersection.t) + 1e-4f * intersection.surfaceNormal;

				bool isBlocked;
				float llPdf, blPdf;
				float lbPdf, bbPdf;
				glm::vec3 lightDir;

				LightSourceSampling::sampleLight_Sphere(lightDir, isBlocked, llPdf, lightSrc, isecPt, geoms, numGeoms, triangles, numTriangles, bvh, rng, u01);
				glm::vec3 brdfDir = calculateRandomDirectionInHemisphere(bbPdf, intersection.surfaceNormal, rng, u01);
				blPdf = LightSourceSampling::sphereLightPdf(lightSrc, isecPt, brdfDir);
				lbPdf = cosWeightedHemispherePdf(intersection.surfaceNormal, lightDir);

				// IS
				//pathSegments[idx].color *= lambertBrdf * fmax(0.f, glm::dot(intersection.surfaceNormal, brdfDir)) / (bbPdf + FLT_EPSILON);

				// MIS
				float wLight = powerHeuristic(1, llPdf, 1, lbPdf);
				float wBrdf = powerHeuristic(1, bbPdf, 1, blPdf);
				const Material &lightMat = materials[lightSrc.materialid];
				glm::vec3 lightColor = lightMat.color * lightMat.emittance;

				if (!isBlocked)
				{
					pathSegments[idx].color +=
						pathSegment.misWeight *
						lambertBrdf * lightColor * fmax(0.f, glm::dot(intersection.surfaceNormal, lightDir)) *
						wLight / (llPdf + FLT_EPSILON);
				}
				pathSegments[idx].misWeight *=
					lambertBrdf * fmax(0.f, glm::dot(intersection.surfaceNormal, brdfDir)) *
					wBrdf / (bbPdf + FLT_EPSILON);

				pathSegments[idx].ray = { isecPt, brdfDir };
				if (--pathSegments[idx].remainingBounces <= 0)
				{
					pathTypes[tid] = Terminated;
				}
			}
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec4 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += glm::vec4(iterationPath.color, 1.f);
	}
}


template <unsigned pathType>
struct EqualToPathType
{
	__device__ bool operator()(unsigned x) const
	{
		return x == pathType;
	}
};


void groupPathsByType(unsigned numPaths, bool doCounting = false)
{
	thrust::device_ptr<unsigned> thrust_pathTypes(dev_pathTypes);
	thrust::device_ptr<unsigned> thrust_pathIndices(dev_pathIndices);
	//thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
	//thrust::device_ptr<ShadeableIntersection> thrust_isecs(dev_intersections);

	thrust::stable_sort_by_key(thrust_pathTypes, thrust_pathTypes + numPaths, thrust_pathIndices);
	//thrust::device_vector<int> indices(numPaths);
	//thrust::device_vector<PathSegment> pvec(thrust_paths, thrust_paths + numPaths);
	//thrust::device_vector<ShadeableIntersection> ivec(thrust_isecs, thrust_isecs + numPaths);
	//thrust::sequence(indices.begin(), indices.end());
	//thrust::stable_sort_by_key(thrust_pathTypes, thrust_pathTypes + numPaths, indices.begin());
	//thrust::gather(indices.begin(), indices.end(), pvec.begin(), thrust_paths);
	//thrust::gather(indices.begin(), indices.end(), ivec.begin(), thrust_isecs);

	if (doCounting)
	{
		h_pathCountsPerType[Terminated] =
			thrust::count_if(thrust_pathTypes, thrust_pathTypes + numPaths, EqualToPathType<Terminated>());
		//h_pathCountsPerType[Initial] =
		// thrust::count_if(thrust_pathTypes, thrust_pathTypes + numPaths, EqualToPathType<Initial>());
		//h_pathCountsPerType[Lambert] =
		// thrust::count_if(thrust_pathTypes, thrust_pathTypes + numPaths, EqualToPathType<Lambert>());
		//h_pathCountsPerType[Mirror] =
		// thrust::count_if(thrust_pathTypes, thrust_pathTypes + numPaths, EqualToPathType<Mirror>());
	}
}


__global__ void kernRegenPaths(
	glm::vec4 *image,
	PathSegment *paths,
	unsigned *pathTypes,
	const unsigned *pathIndices,
	int numPaths,
	Camera cam,
	int iter,
	int curDepth,
	int maxDepth)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numPaths)
	{
		PathType pathType = static_cast<PathType>(pathTypes[tid]);

		if (pathType == Terminated)
		{
			int index = pathIndices[tid];
			PathSegment &segment = paths[index];

			// Store color from last iteration
			image[segment.pixelIndex] += glm::vec4(segment.color, 1.f);

			// Generate new path (discard old data)
			int y = segment.pixelIndex / cam.resolution.x;
			int x = segment.pixelIndex - cam.resolution.x * y;

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, curDepth);
			initPathSegment(segment, x, y, index, maxDepth, cam, rng);
			pathTypes[tid] = Initial;
		}
	}
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	//if (iter == 0)
	//{
		generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, 0, traceDepth, dev_paths);
		cudaMemset(dev_pathTypes, 0, pixelcount * sizeof(unsigned));
		kernInitPathIndices << <NUM_BLOCKS(pixelcount, 256), 256 >> >(pixelcount, dev_pathIndices);
		checkCUDAError("generate camera ray");
	//}

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete)
	//while (depth < traceDepth)
	{
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		//kernRegenPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
		//	dev_image,
		//	dev_paths,
		//	dev_pathTypes,
		//	dev_pathIndices,
		//	num_paths,
		//	cam,
		//	iter,
		//	depth,
		//	traceDepth);

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		
		// tracing
		computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth, 
			num_paths, 
			dev_paths, 
			dev_pathTypes,
			dev_pathIndices,
			dev_geoms, 
			hst_scene->geoms.size(), 
			dev_intersections,
			dev_materials,
			dev_triangles,
			hst_scene->triangles.size(),
			dev_bvh
			);
		checkCUDAError("trace one bounce");
		//cudaDeviceSynchronize();

		groupPathsByType(num_paths, true);
		//groupPathsByType(num_paths);

		//printf("host: %d, %d", sizeof(PathSegment), __alignof(PathSegment));
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		shadeMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth,
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_pathTypes,
			dev_pathIndices,
			dev_materials,
			dev_lights,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_triangles,
			hst_scene->triangles.size(),
			dev_bvh
			);
		checkCUDAError("shadeMaterials");

		//groupPathsByType(num_paths, true);
		//groupPathsByType(num_paths);
		++depth;
		if (h_pathCountsPerType[Terminated] == num_paths)
		{
			iterationComplete = true;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
