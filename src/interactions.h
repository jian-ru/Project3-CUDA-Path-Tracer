#pragma once

#include "intersections.h"

__host__ __device__ __forceinline__ float cosWeightedHemispherePdf(
	const glm::vec3 &normal, const glm::vec3 &dir)
{
	return fmax(0.f, glm::dot(normal, dir)) / PI;
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
	float &pdf,
	glm::vec3 normal,
	thrust::default_random_engine &rng,
	thrust::uniform_real_distribution<float> &u01)
{
	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	glm::vec3 result = glm::normalize(up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2);

	pdf = cosWeightedHemispherePdf(normal, result);
	return result;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
}

__device__ __forceinline__ float powerHeuristic(int nf, float pdff, int ng, float pdfg)
{
	float f = nf * pdff;
	float g = ng * pdfg;
	return f * f / (f * f + g * g);
}

namespace ShapeSampling
{
	__device__ __forceinline__ float uniformConePdf(float costheta_max)
	{
		return 1.0f / (TWO_PI * (1.0f - costheta_max));
	}

	__device__ glm::vec3 uniformSampleCone(
		float cos_theta_max,
		thrust::default_random_engine &rng,
		thrust::uniform_real_distribution<float> &u01)
	{
		float u = u01(rng);
		float v = u01(rng);

		float costheta = (1.0f - u) + u * cos_theta_max;
		float sintheta = sqrtf(1.0f - costheta * costheta);
		float phi = v * TWO_PI;

		return glm::vec3(cosf(phi) * sintheta, sinf(phi) * sintheta, costheta);
	}

	__device__ void getCone2LocalT(const glm::vec3 &cz, glm::mat3 &cone2local)
	{
		glm::vec3 cx, cy;

		cx = glm::cross(glm::vec3(0, 1, 0), cz);

		if (glm::length2(cx) < FLT_EPSILON)
		{
			cx = glm::vec3(1, 0, 0);
			cy = glm::vec3(0, 0, -1);
		}
		else
		{
			cx = glm::normalize(cx);
			cy = glm::normalize(glm::cross(cz, cx));
		}

		cone2local[0] = cx;
		cone2local[1] = cy;
		cone2local[2] = cz;
	}

	__device__ void concentricSamplingDisk(float u, float v, float &x_ret, float &y_ret)
	{
		float r, phi;
		float x = 2 * u - 1;    // [0, 1] -> [-1, 1]
		float y = 2 * v - 1;

		// handle degeneracy at the origin
		if (fabsf(x) < FLT_EPSILON && fabsf(y) > FLT_EPSILON)
		{
			x_ret = 0.f;
			y_ret = 0.f;
		}

		if (x > -y)
		{
			if (x > y)
			{
				// first region
				r = x;
				phi = y / x;
			}
			else
			{
				r = y;
				phi = 2 - x / y;
			}
		}
		else
		{
			if (x < y)
			{
				r = -x;
				phi = 4 + y / x;
			}
			else
			{
				r = -y;
				phi = 6 - x / y;
			}
		}

		phi *= 0.25f * PI;
		x_ret = r * cosf(phi);
		y_ret = r * sinf(phi);
	}
}

namespace LightSourceSampling
{
	using namespace ShapeSampling;

	__device__ float sphereLightPdf(const Geom &sphere, const glm::vec3 &isecPt, const glm::vec3 &dir)
	{
		glm::vec3 p = glm::vec3(sphere.inverseTransform * glm::vec4(isecPt, 1.0f));
		glm::vec3 d = glm::normalize(glm::vec3(sphere.inverseTransform * glm::vec4(dir, 0.f)));
		float sin_theta_max2 = 0.25f / glm::distance2(glm::vec3(0.0f), p);
		float cos_theta_max = sqrtf(fmax(0.0f, 1.0f - sin_theta_max2));
		float cos_theta = glm::dot(glm::normalize(-p), d);
		return cos_theta >= cos_theta_max ? uniformConePdf(cos_theta_max) : 0.f;
	}

	__device__ void sampleLight_Sphere(
		glm::vec3 &sample_dir,
		bool &isBlocked,
		float &pdf,
		const Geom &sphere,
		const glm::vec3 &isecPt,
		const Geom *geoms,
		int numGeoms,
		const Triangle *triangles,
		int numTriangles,
		BVH::GpuBVH bvh,
		thrust::default_random_engine &rng,
		thrust::uniform_real_distribution<float> &u01)
	{
		glm::vec3 p = glm::vec3(sphere.inverseTransform * glm::vec4(isecPt, 1.0f));   // local p
		float sin_theta_max2 = 0.25f / glm::distance2(glm::vec3(0.0f), p);
		float cos_theta_max = sqrtf(fmax(0.0f, 1.0f - sin_theta_max2));

		// get the transform from cone local space to sphere local space to
		// make it easier to sample the cone
		glm::vec3 cz = glm::normalize(-p);
		glm::mat3 cone2local;
		glm::vec3 cone_sample_dir = uniformSampleCone(cos_theta_max, rng, u01);

		getCone2LocalT(cz, cone2local);
		sample_dir = glm::normalize(glm::vec3(sphere.transform * glm::vec4(cone2local * cone_sample_dir, 0.0f))); // to world space

		Ray r = { isecPt, sample_dir };
		float t;
		int hitGeomIdx;
		glm::vec3 normal;

		if (numGeoms + numTriangles < MIN_OBJECTS_REQUIRED_FOR_BVH)
		{
			SceneIntersection::getRaySceneIntersection(t, hitGeomIdx, normal, r, geoms, numGeoms, triangles, numTriangles);
		}
		else
		{
			BVH::getRaySceneIntersection(t, hitGeomIdx, normal, r, bvh, geoms, triangles);
		}

		pdf = uniformConePdf(cos_theta_max);
		isBlocked = true;
		if (t > 0.f && hitGeomIdx == sphere.idx)
		{
			isBlocked = false;
		}
	}
}

namespace Fresnel
{
	// Assume IOR is invariant across light spectrum (which is not true)
	__device__ __forceinline__ float frDiel(float cosi, float etai, float cost, float etat)
	{
		// PBRT
		//float rParl = (etat * cosi - etai * cost) / (etat * cosi + etai * cost + FLT_EPSILON);
		//float rPerp = (etai * cosi - etat * cost) / (etai * cosi + etat * cost + FLT_EPSILON);
		//return (rParl * rParl + rPerp * rPerp) * 0.5f;

		// Schlick's
		float R0 = (etai - etat) / (etai + etat);
		R0 = R0 * R0;
		return R0 + (1.f - R0) * powf(1.f - cosi, 5.f);
	}
}