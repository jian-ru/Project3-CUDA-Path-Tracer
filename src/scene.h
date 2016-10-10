#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

namespace BVH
{
	class BVH;
}

class Scene {
private:
	ifstream fp_in;
	int loadMaterial(string materialid);
	int loadGeom(string objectid);
	int loadCamera();
	void loadMesh(Geom *mesh, const std::string &filename);
public:
	Scene(string filename);
	~Scene();

	std::shared_ptr<BVH::BVH> bvh;
	std::vector<Triangle> triangles;
	std::vector<Geom> geoms;
	std::vector<Material> materials;
	RenderState state;
};


extern __host__ __device__ float boxIntersectionTest(Geom box, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside);

extern __host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside);

extern __host__ __device__ float triangleIntersectionTest(const Geom &mesh, const Triangle &tri,
	const Ray &r, glm::vec3 &intersectionPoint, glm::vec3 &normal);


namespace BVH
{
	class BBox
	{
	public:
		glm::vec3 p_min;
		glm::vec3 p_max;
		glm::vec3 centroid;

		BBox();

		BBox(const glm::vec3 &min_corner, const glm::vec3 &max_corner);

		// return the dimension with maximum extent
		// 0 = x, 1 = y, 2 = z
		int dimMaxExtent() const;

		float surfaceArea() const;

		bool intersect(Ray r) const;

		friend BBox bbox_point_union(const BBox &b, const glm::vec3 &p);
		friend BBox bbox_union(const BBox &b0, const BBox &b1);
	};

	BBox bbox_point_union(const BBox &b, const glm::vec3 &p);

	BBox bbox_union(const BBox &b0, const BBox &b1);

	class ShapePrimitive
	{
	public:
		enum class PrimitiveType
		{
			Cube,
			Sphere,
			Triangle
		} type;
		Geom *geom = nullptr;
		Triangle *tri = nullptr;
		BBox worldBound;
	};

	void refineGeom(
		std::vector<std::shared_ptr<ShapePrimitive>> &prims,
		Scene *scene,
		Geom *geom);

	struct GpuBvhNode
	{
		struct GpuBBox
		{
			glm::vec3 pMin;
			glm::vec3 pMax;
		} worldBound;
		int left, right;
		int splitAxis;
		int numPrims, primOffset;
	};

	struct GpuPrimPtr
	{
		int type; // 0 = cube, 1 = sphere, 2 = triangle
		int index;
	};

	struct GpuBVH
	{
		GpuBvhNode *d_nodes = nullptr;
		GpuPrimPtr *d_primimitives = nullptr;
	};

	class BVH
	{
	public:
		BVH() {}

		void construct(Scene *scene);

		GpuBVH getGpuBuffers() const;

		void clear() { nodes.clear(); }

		void recursiveBuild(std::vector<std::shared_ptr<ShapePrimitive>> &prims, uint32_t depth);

		struct BVHNode
		{
			BVHNode() { left = right = -1; }

			void makeLeaf(const std::vector<std::shared_ptr<ShapePrimitive>> &prims,
				const BBox &b)
			{
				primitives = prims;
				bounds = b;
			}

			void makeInterior(int axis, int l_child, int r_child,
				const std::vector<BVHNode> &nodes)
			{
				split_axis = axis;
				left = l_child;
				right = r_child;
				bounds = bbox_union(nodes[l_child].bounds, nodes[r_child].bounds);
			}

			BBox bounds;
			int left, right;
			int split_axis;
			std::vector<std::shared_ptr<ShapePrimitive>> primitives;
		};

		std::vector<BVHNode> nodes;
	};

#ifdef IMPLEMENT_DEVICE_FUNTIONS
	__device__ bool bboxTryHitSlabs(const GpuBvhNode::GpuBBox &b, int axis, const Ray &r, float &t_near, float &t_far)
	{
		const float slab_min = b.pMin[axis];
		const float slab_max = b.pMax[axis];

		if (fabs(r.direction[axis]) < FLT_EPSILON)
		{
			// parallel
			if (r.origin[axis] < slab_min || r.origin[axis] > slab_max) return false;
		}
		else
		{
			float t0 = (slab_min - r.origin[axis]) / r.direction[axis];
			float t1 = (slab_max - r.origin[axis]) / r.direction[axis];

			if (t0 > t1)
			{
				float tmp = t1;
				t1 = t0;
				t0 = tmp;
			}
			if (t0 > t_near) t_near = t0;
			if (t1 < t_far) t_far = t1;
			if (t_near > t_far || t_far < 0.f) return false;
		}

		return true;
	};

	__device__ bool isBBoxHitByRay(const GpuBvhNode::GpuBBox &bbox, const Ray &r)
	{
		float t_near = -FLT_MAX;
		float t_far = FLT_MAX;

		if (!bboxTryHitSlabs(bbox, 0, r, t_near, t_far) ||
			!bboxTryHitSlabs(bbox, 1, r, t_near, t_far) ||
			!bboxTryHitSlabs(bbox, 2, r, t_near, t_far))
		{
			return false;   // miss
		}

		return true;
	}

	__device__ void getRaySceneIntersection(
		float &t_ret,
		int &hitGeomIdx_ret,
		glm::vec3 &normal_ret,
		const Ray &r,
		GpuBVH bvh,
		const Geom *geoms,
		const Triangle *triangles)
	{
		int top = 0;
		int todo[2 * MAX_BVH_DEPTH]; // store the index of the nodes need to be visited

		glm::vec3 normal;
		int hit_geom_index = -1;
		float t_min = FLT_MAX;
		todo[top++] = 0;  // root

		while (top != 0)
		{
			int node_idx;
			const GpuBvhNode *cur;
			node_idx = todo[--top];
			cur = &bvh.d_nodes[node_idx];

			if (isBBoxHitByRay(cur->worldBound, r))
			{
				if (cur->numPrims > 0)
				{
					// leaf. test intersection against all primitives
					for (int i = 0; i < cur->numPrims; ++i)
					{
						const GpuPrimPtr &prim = bvh.d_primimitives[cur->primOffset + i];
						float t;
						int tmp_geomIdx;
						glm::vec3 tmp_intersect;
						glm::vec3 tmp_normal;
						bool outside;

						if (prim.type == 0) // cube
						{
							tmp_geomIdx = prim.index;
							t = boxIntersectionTest(geoms[prim.index], r, tmp_intersect, tmp_normal, outside);
						}
						else if (prim.type == 1) // sphere
						{
							tmp_geomIdx = prim.index;
							t = sphereIntersectionTest(geoms[prim.index], r, tmp_intersect, tmp_normal, outside);
						}
						else // triangle
						{
							const Triangle &tri = triangles[prim.index];
							const Geom &g = geoms[tri.meshIdx];
							tmp_geomIdx = tri.meshIdx;
							t = triangleIntersectionTest(g, tri, r, tmp_intersect, tmp_normal);
							//printf("%d, %d, %f, %f, %f, %f\n", prim.index, tri.meshIdx, tmp_intersect.x, tmp_intersect.y, tmp_intersect.z, t);
						}

						if (t > 0.f && t_min > t)
						{
							t_min = t;
							hit_geom_index = tmp_geomIdx;
							normal = tmp_normal;
						}
					}
				}
				else
				{
					// check intersection with the near bbox
					// put the far bbox to the todo stack
					if (r.direction[cur->splitAxis] < 0.0f)
					{
						// go to the right sub-bbox first
						todo[top++] = cur->left;
						todo[top++] = cur->right;
					}
					else
					{
						todo[top++] = cur->right;
						todo[top++] = cur->left;
					}
				}
			}
		}

		t_ret = t_min;
		hitGeomIdx_ret = hit_geom_index;
		normal_ret = normal;
	}
#endif
}
