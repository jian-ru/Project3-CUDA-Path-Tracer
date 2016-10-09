#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(const Geom &mesh, const Triangle &tri,
	const Ray &r, glm::vec3 &intersectionPoint, glm::vec3 &normal)
{
	// in BVH, triangle is responsible of transforming ray into local space
	glm::vec3 ro = glm::vec3(mesh.inverseTransform * glm::vec4(r.origin, 1.f));
	glm::vec3 rd = glm::normalize(glm::vec3(mesh.inverseTransform * glm::vec4(r.direction, 0.f)));

//#define NAIVE_TRI_TEST
#ifdef NAIVE_TRI_TEST
	//1. Ray-plane intersection
	glm::vec3 plane_normal = glm::normalize(glm::cross(tri.positions[1] - tri.positions[0], tri.positions[2] - tri.positions[0]));
	float t = glm::dot(plane_normal, (tri.positions[0] - ro)) / glm::dot(plane_normal, rd);

	if (t <= 0.f)
	{
		// intersection is behind ray origin
		return t;
	}

	glm::vec3 P = t * rd + ro;   // local p
	//2. Barycentric test
	float S = 0.5f * glm::length(glm::cross(tri.positions[0] - tri.positions[1], tri.positions[0] - tri.positions[2]));
	float s1 = 0.5f * glm::length(glm::cross(P - tri.positions[1], P - tri.positions[2])) / S;
	float s2 = 0.5f * glm::length(glm::cross(P - tri.positions[2], P - tri.positions[0])) / S;
	float s3 = 0.5f * glm::length(glm::cross(P - tri.positions[0], P - tri.positions[1])) / S;

	t = FLT_MAX;
	if (fabs(s1 + s2 + s3 - 1.f) < 1e-4f)
	{
		intersectionPoint = glm::vec3(mesh.transform * glm::vec4(P, 1.f));
		normal = glm::normalize(glm::vec3(mesh.invTranspose *
			glm::vec4(s1 * tri.normals[0] + s2 * tri.normals[1] + s3 * tri.normals[2], 0.f)));
		t = glm::distance(intersectionPoint, r.origin);
	}

	return t;
#else // Moller-Trumbore ray-triangle intersection
	glm::vec3 e1, e2;
	glm::vec3 P, Q, T;
	float det, inv_det, u, v;
	float t;

	e1 = tri.positions[1] - tri.positions[0];
	e2 = tri.positions[2] - tri.positions[0];

	P = glm::cross(rd, e2);
	det = glm::dot(e1, P);
	if (det > -1e-5f && det < 1e-5f) return -1.f;
	inv_det = 1.f / det;

	T = ro - tri.positions[0];
	u = glm::dot(T, P) * inv_det;
	if (u < 0.f || u > 1.f) return -1.f;

	Q = glm::cross(T, e1);
	v = glm::dot(rd, Q) * inv_det;
	if (v < 0.f || u + v > 1.f) return -1.f;

	t = glm::dot(e2, Q) * inv_det;

	if (t > 1e-5f)
	{
		intersectionPoint = glm::vec3(mesh.transform * glm::vec4(ro + t * rd, 1.f));
		normal = glm::normalize((1.f - u - v) * tri.normals[0] +
			u * tri.normals[1] + v * tri.normals[2]);
		return glm::distance(intersectionPoint, r.origin); // world t
	}

	return -1.f;
#endif
}

namespace SceneIntersection
{
	__device__ void getRaySceneIntersection(
		float &t_ret,
		int &hit_geom_index_ret,
		glm::vec3 &normal_ret,
		const Ray &ray,
		const Geom * geoms,
		int geoms_size,
		const Triangle *triangles,
		int numTriangles)
	{
		float t;
		float t_min = FLT_MAX;
		bool outside = true;
		int hit_geom_index = -1;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec3 normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++)
		{
			const Geom &geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				normal = tmp_normal;
			}
		}

		for (int i = 0; i < numTriangles; ++i)
		{
			const Triangle &tri = triangles[i];
			const Geom &geom = geoms[tri.meshIdx];

			t = triangleIntersectionTest(geom, tri, ray, tmp_intersect, tmp_normal);

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = tri.meshIdx;
				normal = tmp_normal;
			}
		}

		t_ret = t_min;
		hit_geom_index_ret = hit_geom_index;
		normal_ret = normal;
	}
}