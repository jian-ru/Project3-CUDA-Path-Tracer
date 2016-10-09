#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tinyobj/tiny_obj_loader.h"
#define IMPLEMENT_BVH
#include "scene.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

	bvh = std::make_shared<BVH::BVH>();
	bvh->construct(this);
}

void Scene::loadMesh(Geom *mesh, const std::string &filename)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	// Mesh will be triangulated
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
	
	if (!err.empty()) {
		std::cerr << err << std::endl;
	}

	if (!ret) {
		printf("Failed to load/parse .obj.\n");
		throw std::exception();
	}

	const auto &positions = attrib.vertices;
	const auto &normals = attrib.normals;

	mesh->triListOffset = triangles.size();

	for (const auto &shape : shapes)
	{
		const auto &indices = shape.mesh.indices;
		size_t numFaces = indices.size() / 3;

		for (int i = 0; i < numFaces; ++i)
		{
			Triangle tri;

			tinyobj::index_t idx0 = indices[i * 3];
			tinyobj::index_t idx1 = indices[i * 3 + 1];
			tinyobj::index_t idx2 = indices[i * 3 + 2];

			tri.meshIdx = mesh->idx;
			tri.positions[0] = glm::vec3(
				positions[3 * idx0.vertex_index],
				positions[3 * idx0.vertex_index + 1],
				positions[3 * idx0.vertex_index + 2]);
			tri.positions[1] = glm::vec3(
				positions[3 * idx1.vertex_index],
				positions[3 * idx1.vertex_index + 1],
				positions[3 * idx1.vertex_index + 2]);
			tri.positions[2] = glm::vec3(
				positions[3 * idx2.vertex_index],
				positions[3 * idx2.vertex_index + 1],
				positions[3 * idx2.vertex_index + 2]);
			tri.normals[0] = glm::vec3(
				normals[3 * idx0.normal_index],
				normals[3 * idx0.normal_index + 1],
				normals[3 * idx0.normal_index + 2]);
			tri.normals[1] = glm::vec3(
				normals[3 * idx1.normal_index],
				normals[3 * idx1.normal_index + 1],
				normals[3 * idx1.normal_index + 2]);
			tri.normals[2] = glm::vec3(
				normals[3 * idx2.normal_index],
				normals[3 * idx2.normal_index + 1],
				normals[3 * idx2.normal_index + 2]);
			tri.idx = triangles.size();

			triangles.push_back(tri);
		}
	}

	mesh->numTriangles = triangles.size() - mesh->triListOffset;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
		newGeom.idx = geoms.size();
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
			else if (strcmp(line.c_str(), "mesh") == 0)
			{
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
			}
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
			else if (strcmp(tokens[0].c_str(), "FILE") == 0)
			{
				if (newGeom.type == MESH)
				{
					std::string filename;
					for (int i = 1; i < tokens.size(); ++i)
					{
						filename += tokens[i] + " ";
					}
					loadMesh(&newGeom, filename);
				}
				else
				{
					std::cout << "Warning: Object %d is not a mesh. FILE attribute is ignored.\n";
				}
			}

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 7; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
		}
		else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
			camera.lensRadius = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FOCALD") == 0) {
			camera.focalDistance = atof(tokens[1].c_str());
		}
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec4());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;
		std::unordered_map<std::string, MaterialType> matTypeLut =
		{
			{ "lambert", M_Lambert },
			{ "mirror", M_Mirror },
			{ "glass", M_Glass },
			{ "ashikhminshirley", M_AshikhminShirley },
			{ "torrancesparrow", M_TorranceSparrow },
		};

        //load static properties
		const int kNumAttributes = 8;
        for (int i = 0; i < kNumAttributes; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "TYPE") == 0)
			{
				std::transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), ::tolower);
				newMaterial.type = matTypeLut[tokens[1]];
			}
            else if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}


namespace BVH
{
	BBox::BBox() :
		p_min(glm::vec3(std::numeric_limits<float>::max())),
		p_max(glm::vec3(-std::numeric_limits<float>::max())),
		centroid(glm::vec3(0.f)) {}

	BBox::BBox(const glm::vec3 &min_corner, const glm::vec3 &max_corner)
		: p_min(min_corner), p_max(max_corner)
	{
		centroid = 0.5f * p_min + 0.5f * p_max;
	}

	int BBox::dimMaxExtent() const
	{
		int max_dim = 0;
		int max_val = p_max[0] - p_min[0];

		for (int i = 1; i < 3; ++i)
		{
			if (p_max[i] - p_min[i] > max_val)
			{
				max_dim = i;
				max_val = p_max[i] - p_min[i];
			}
		}

		return max_dim;
	}

	float BBox::surfaceArea() const
	{
		glm::vec3 d = p_max - p_min;
		return 2.0f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
	}

	bool BBox::intersect(Ray r) const
	{
		auto bboxTryHitSlabs = [](const BBox &b, int axis, const Ray &r, float &t_near, float &t_far)
		{
			const float slab_min = b.p_min[axis];
			const float slab_max = b.p_max[axis];

			if (fabs(r.direction[axis]) < FLT_EPSILON)
			{
				// parallel
				if (r.origin[axis] < slab_min || r.origin[axis] > slab_max)
				{
					return false;
				}
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

				if (t0 > t_near)
				{
					t_near = t0;
				}

				if (t1 < t_far)
				{
					t_far = t1;
				}
			}

			return true;
		};

		float t_near = -std::numeric_limits<float>::max();
		float t_far = std::numeric_limits<float>::max();

		if (!bboxTryHitSlabs(*this, 0, r, t_near, t_far) ||
			!bboxTryHitSlabs(*this, 1, r, t_near, t_far) ||
			!bboxTryHitSlabs(*this, 2, r, t_near, t_far))
		{
			return false;   // miss
		}

		if (t_near > t_far || t_far <= 0.f)
		{
			return false;   // miss
		}

		return true;
	}

	BBox bbox_point_union(const BBox &b, const glm::vec3 &p)
	{
		glm::vec3 p_min(std::min(b.p_min[0], p[0]),
			std::min(b.p_min[1], p[1]),
			std::min(b.p_min[2], p[2]));
		glm::vec3 p_max(std::max(b.p_max[0], p[0]),
			std::max(b.p_max[1], p[1]),
			std::max(b.p_max[2], p[2]));

		return BBox(p_min, p_max);
	}

	BBox bbox_union(const BBox &b0, const BBox &b1)
	{
		glm::vec3 p_min(std::min(b0.p_min[0], b1.p_min[0]),
			std::min(b0.p_min[1], b1.p_min[1]),
			std::min(b0.p_min[2], b1.p_min[2]));
		glm::vec3 p_max(std::max(b0.p_max[0], b1.p_max[0]),
			std::max(b0.p_max[1], b1.p_max[1]),
			std::max(b0.p_max[2], b1.p_max[2]));

		return BBox(p_min, p_max);
	}

	void refineGeom(
		std::vector<std::shared_ptr<ShapePrimitive>> &prims,
		Scene *scene,
		Geom *geom)
	{
		static const glm::vec4 unitCubeVerts[8] =
		{
			{ -.5f, -.5f, -.5f, 1.f },
			{ .5f, -.5f, -.5f, 1.f },
			{ -.5f, .5f, -.5f, 1.f },
			{ -.5f, -.5f, .5f, 1.f },
			{ .5f, .5f, -.5f, 1.f },
			{ .5f, -.5f, .5f, 1.f },
			{ -.5f, .5f, .5f, 1.f },
			{ .5f, .5f, .5f, 1.f },
		};

		if (geom->type == CUBE || geom->type == SPHERE)
		{
			std::shared_ptr<ShapePrimitive> prim =
				std::make_shared<ShapePrimitive>();

			for (int i = 0; i < 8; ++i)
			{
				prim->worldBound = bbox_point_union(prim->worldBound,
					glm::vec3(geom->transform * unitCubeVerts[i]));
			}
			prim->geom = geom;
			prim->type = geom->type == CUBE ?
				ShapePrimitive::PrimitiveType::Cube : ShapePrimitive::PrimitiveType::Sphere;
			prims.push_back(prim);
		}
		else if (geom->type == MESH)
		{
			std::vector<Triangle> &tris = scene->triangles;

			for (int i = 0; i < geom->numTriangles; ++i)
			{
				Triangle *tri = &tris[geom->triListOffset + i];

				std::shared_ptr<ShapePrimitive> prim =
					std::make_shared<ShapePrimitive>();

				for (int i = 0; i < 3; ++i)
				{
					prim->worldBound = bbox_point_union(prim->worldBound,
						glm::vec3(geom->transform * glm::vec4(tri->positions[i], 1.f)));
				}
				prim->geom = geom;
				prim->tri = tri;
				prim->type = ShapePrimitive::PrimitiveType::Triangle;
				prims.push_back(prim);
			}
		}
		else
		{
			std::cerr << "Error: refineGeom: Unknown geom type.\n";
			throw std::exception();
		}
	}

	void BVH::construct(Scene *scene)
	{
		std::vector<std::shared_ptr<ShapePrimitive>> prims;

		for (auto &geom : scene->geoms)
		{
			refineGeom(prims, scene, &geom);
		}

		recursiveBuild(prims, 0);
	}

	void BVH::recursiveBuild(std::vector<std::shared_ptr<ShapePrimitive>> &prims, uint32_t depth)
	{
		// find out along which dimension to split
		// split along the dimension with maximum spread of centroids of primitive bounds
		BBox c_bounds;
		int dim;

		for (int i = 0; i < prims.size(); ++i)
		{
			c_bounds = bbox_point_union(c_bounds, prims[i]->worldBound.centroid);
		}

		dim = c_bounds.dimMaxExtent();

		// if all centroids are at the same position, there is no way to split.
		// make a leaf containing all primitives instead.
		if (prims.size() <= BVH_LEAF_SIZE || depth >= MAX_BVH_DEPTH ||
			c_bounds.p_max[dim] == c_bounds.p_min[dim])
		{
			BVHNode leaf_node;
			BBox b;

			for (int i = 0; i < prims.size(); ++i)
			{
				b = bbox_union(b, prims[i]->worldBound);
			}

			leaf_node.makeLeaf(prims, b);
			nodes.push_back(leaf_node);
			return;
		}

		// use surface area heuristic
		// divide the extent long dim into num_buckets buckets
		// so we don't need to consider every possible partition
		// scheme.
		const int num_buckets = 12;

		struct BucketInfo
		{
			BucketInfo() { count = 0; }

			int count;
			BBox bounds;
		};

		BucketInfo buckets[num_buckets];

		for (int i = 0; i < prims.size(); ++i)
		{
			// find out which bucket prims[i] falls in
			int b = num_buckets *
				((prims[i]->worldBound.centroid[dim] - c_bounds.p_min[dim]) /
				(c_bounds.p_max[dim] - c_bounds.p_min[dim]));

			if (b == num_buckets)
			{
				--b;
			}

			// update the bounding box to contain prims[i]'s bounding box
			++(buckets[b].count);
			buckets[b].bounds = bbox_union(buckets[b].bounds, prims[i]->worldBound);
		}

		// it doesn't make sense to split behind the last bucket
		float cost[num_buckets - 1];

		// compute cost for splitting after each bucket
		for (int i = 0; i < num_buckets - 1; ++i)
		{
			BBox b0;        // contains all the bucket bounds in the first partition
			BBox b1;        // contains all the bucket bounds in the second partition
			int count0 = 0; // number of primitives in the first partition
			int count1 = 0; // number of primitives in the second partition

			for (int j = 0; j <= i; ++j)
			{
				count0 += buckets[j].count;
				b0 = bbox_union(b0, buckets[j].bounds);
			}

			for (int j = i + 1; j < num_buckets; ++j)
			{
				count1 += buckets[j].count;
				b1 = bbox_union(b1, buckets[j].bounds);
			}

			// suppose the cost of traversal is 1/8 and the cost of
			// testing intersection with a primitive is 1
			BBox b01 = bbox_union(b0, b1);

			cost[i] = .125f +
				(count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) /
				b01.surfaceArea();
		}

		// find the partition scheme with minimum cost
		float min_cost = cost[0];
		int split_after = 0;

		for (int i = 1; i < num_buckets - 1; ++i)
		{
			if (cost[i] < min_cost)
			{
				min_cost = cost[i];
				split_after = i;
			}
		}

		// split and make an interior node
		// CompareToBucket is a funtor
		struct CompareToBucket
		{
			CompareToBucket(int s, int n, int d, const BBox &b)
			: split_bucket(s), n_buckets(n), dim(d), centroid_bounds(b) {}

			// return true if the given primitive is in the first partition
			bool operator()(std::shared_ptr<const ShapePrimitive> prim) const
			{
				int b = n_buckets *
					((prim->worldBound.centroid[dim] - centroid_bounds.p_min[dim]) /
					(centroid_bounds.p_max[dim] - centroid_bounds.p_min[dim]));

				if (b == n_buckets)
				{
					--b;
				}

				return b <= split_bucket;
			}

			int split_bucket, n_buckets, dim;
			const BBox &centroid_bounds;
		};

		std::vector<std::shared_ptr<ShapePrimitive>>::iterator pmid =
			std::partition(prims.begin(), prims.end(),
			CompareToBucket(split_after, num_buckets, dim, c_bounds));
		std::vector<std::shared_ptr<ShapePrimitive>> first_part(prims.begin(), pmid);
		std::vector<std::shared_ptr<ShapePrimitive>> second_part(pmid, prims.end());

		BVHNode in_node;
		int in_node_idx = nodes.size();

		nodes.push_back(in_node);
		recursiveBuild(first_part, depth + 1);

		int l_child = in_node_idx + 1;
		int r_child = nodes.size();

		recursiveBuild(second_part, depth + 1);
		nodes[in_node_idx].makeInterior(dim, l_child, r_child, nodes);
	}

	GpuBVH BVH::getGpuBuffers() const
	{
		std::vector<GpuBvhNode> h_nodes;
		std::vector<GpuPrimPtr> h_prims;
		GpuBVH ret;

		for (size_t i = 0; i < nodes.size(); ++i)
		{
			const BVHNode &n = nodes[i];
			GpuBvhNode node;
			memset(&node, 0, sizeof(GpuBvhNode));

			node.left = n.left;
			node.right = n.right;
			node.splitAxis = n.split_axis;
			node.worldBound.pMin = n.bounds.p_min;
			node.worldBound.pMax = n.bounds.p_max;

			if (n.primitives.size() > 0)
			{
				node.numPrims = n.primitives.size();
				node.primOffset = h_prims.size();

				for (const auto &pr : n.primitives)
				{
					GpuPrimPtr prim;

					if (pr->type == ShapePrimitive::PrimitiveType::Cube)
					{
						prim.type = 0;
						prim.index = pr->geom->idx;
					}
					else if (pr->type == ShapePrimitive::PrimitiveType::Sphere)
					{
						prim.type = 1;
						prim.index = pr->geom->idx;
					}
					else if (pr->type == ShapePrimitive::PrimitiveType::Triangle)
					{
						prim.type = 2;
						prim.index = pr->tri->idx;
					}
					else
					{
						std::cerr << "Error: BVH::getGpuBuffers: Unknown primitive type.\n";
						throw std::exception();
					}

					h_prims.push_back(prim);
				}
			}

			h_nodes.push_back(node);
		}

		cudaMalloc(&ret.d_nodes, h_nodes.size() * sizeof(GpuBvhNode));
		cudaMalloc(&ret.d_primimitives, h_prims.size() * sizeof(GpuPrimPtr));
		cudaMemcpy(ret.d_nodes, h_nodes.data(), h_nodes.size() * sizeof(GpuBvhNode), cudaMemcpyHostToDevice);
		cudaMemcpy(ret.d_primimitives, h_prims.data(), h_prims.size() * sizeof(GpuPrimPtr), cudaMemcpyHostToDevice);

		return ret;
	}
}
