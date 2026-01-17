#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <stack>

using namespace std;
using json = nlohmann::json;

// ============================================================================
// BVH Construction Helpers
// ============================================================================

// Compute AABB for a triangle
static void computeTriangleBounds(const Triangle& tri, glm::vec3& boundsMin, glm::vec3& boundsMax)
{
    boundsMin = glm::min(glm::min(tri.v0, tri.v1), tri.v2);
    boundsMax = glm::max(glm::max(tri.v0, tri.v1), tri.v2);
}

// Compute centroid of a triangle
static glm::vec3 computeTriangleCentroid(const Triangle& tri)
{
    return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
}

// Surface area of an AABB
static float surfaceArea(const glm::vec3& boundsMin, const glm::vec3& boundsMax)
{
    glm::vec3 d = boundsMax - boundsMin;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Structure for building BVH
struct BVHBuildEntry
{
    int parentIndex;
    int start;
    int end;
    bool isLeftChild;
};

// SAH cost constants
static const float TRAVERSAL_COST = 1.0f;
static const float INTERSECTION_COST = 1.0f;
static const int MAX_LEAF_SIZE = 4;  // Maximum triangles per leaf
static const int SAH_BINS = 12;      // Number of bins for SAH evaluation

// Dummy image loader callback for tinygltf (we don't use textures yet, just geometry)
static bool DummyLoadImageData(tinygltf::Image* image, const int image_idx, std::string* err,
                               std::string* warn, int req_width, int req_height,
                               const unsigned char* bytes, int size, void* user_data) {
    // We're not loading textures, just return success
    // Set minimal image data to avoid issues
    image->width = 1;
    image->height = 1;
    image->component = 4;
    image->bits = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    image->image.resize(4, 255);  // 1x1 white pixel
    return true;
}

// Helper to extract base path from a file path
static std::string getBasePath(const std::string& filepath) {
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return filepath.substr(0, lastSlash + 1);
    }
    return "";
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    
    basePath = getBasePath(filename);
    
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        
        // Load base color (required for all materials)
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        
        // Load PBR parameters with defaults
        newMaterial.metallic = p.contains("METALLIC") ? (float)p["METALLIC"] : 0.0f;
        newMaterial.roughness = p.contains("ROUGHNESS") ? (float)p["ROUGHNESS"] : 1.0f;
        newMaterial.transparency = p.contains("TRANSPARENCY") ? (float)p["TRANSPARENCY"] : 0.0f;
        newMaterial.indexOfRefraction = p.contains("IOR") ? (float)p["IOR"] : 1.5f;
        newMaterial.emittance = p.contains("EMITTANCE") ? (float)p["EMITTANCE"] : 0.0f;
        
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.triangleStart = 0;
        newGeom.triangleCount = 0;
        newGeom.boundingBoxMin = glm::vec3(FLT_MAX);
        newGeom.boundingBoxMax = glm::vec3(-FLT_MAX);
        
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else
        {
            newGeom.type = SPHERE;  // default
        }
        
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // Load mesh from glTF file if this is a mesh type
        if (newGeom.type == MESH && p.contains("FILE"))
        {
            std::string meshFile = basePath + std::string(p["FILE"]);
            int triangleStartIdx = static_cast<int>(triangles.size());
            
            loadGLTF(meshFile, newGeom.transform, newGeom.materialid);
            
            newGeom.triangleStart = triangleStartIdx;
            newGeom.triangleCount = static_cast<int>(triangles.size()) - triangleStartIdx;
            
            // Compute bounding box from loaded triangles
            for (int i = triangleStartIdx; i < static_cast<int>(triangles.size()); i++)
            {
                const Triangle& tri = triangles[i];
                newGeom.boundingBoxMin = glm::min(newGeom.boundingBoxMin, tri.v0);
                newGeom.boundingBoxMin = glm::min(newGeom.boundingBoxMin, tri.v1);
                newGeom.boundingBoxMin = glm::min(newGeom.boundingBoxMin, tri.v2);
                newGeom.boundingBoxMax = glm::max(newGeom.boundingBoxMax, tri.v0);
                newGeom.boundingBoxMax = glm::max(newGeom.boundingBoxMax, tri.v1);
                newGeom.boundingBoxMax = glm::max(newGeom.boundingBoxMax, tri.v2);
            }
            
            cout << "Loaded mesh with " << newGeom.triangleCount << " triangles" << endl;
            cout << "  Bounding box: [" << glm::to_string(newGeom.boundingBoxMin) 
                 << "] to [" << glm::to_string(newGeom.boundingBoxMax) << "]" << endl;
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    
    cout << "Scene loaded: " << geoms.size() << " geometry objects, " 
         << triangles.size() << " triangles total" << endl;
    
    // Build BVH for all meshes
    buildBVH();
}

void Scene::loadGLTF(const std::string& gltfPath, const glm::mat4& transform, int materialId)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    
    // Set custom image loader (we don't use textures yet, just geometry)
    loader.SetImageLoader(DummyLoadImageData, nullptr);
    
    cout << "Loading glTF: " << gltfPath << endl;
    
    bool ret = false;
    // Check file extension to determine binary or ASCII format
    std::string ext = gltfPath.substr(gltfPath.find_last_of('.'));
    if (ext == ".glb")
    {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfPath);
    }
    else
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
    }
    
    if (!warn.empty())
    {
        cout << "glTF Warning: " << warn << endl;
    }
    
    if (!err.empty())
    {
        cerr << "glTF Error: " << err << endl;
    }
    
    if (!ret)
    {
        cerr << "Failed to load glTF: " << gltfPath << endl;
        return;
    }
    
    // Calculate normal matrix for transforming normals
    glm::mat3 normalMatrix = glm::mat3(glm::inverseTranspose(transform));
    
    // Process all meshes in the model
    for (const auto& mesh : model.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
            {
                cout << "  Skipping non-triangle primitive (mode=" << primitive.mode << ")" << endl;
                continue;
            }
            
            // Get position accessor
            const auto& posAccessorIt = primitive.attributes.find("POSITION");
            if (posAccessorIt == primitive.attributes.end())
            {
                cerr << "  Mesh primitive has no POSITION attribute" << endl;
                continue;
            }
            
            const tinygltf::Accessor& posAccessor = model.accessors[posAccessorIt->second];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            
            const float* positions = reinterpret_cast<const float*>(
                &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
            size_t posStride = posAccessor.ByteStride(posBufferView) / sizeof(float);
            if (posStride == 0) posStride = 3;  // tightly packed
            
            // Get normal accessor (optional)
            const float* normals = nullptr;
            size_t normStride = 3;
            const auto& normAccessorIt = primitive.attributes.find("NORMAL");
            if (normAccessorIt != primitive.attributes.end())
            {
                const tinygltf::Accessor& normAccessor = model.accessors[normAccessorIt->second];
                const tinygltf::BufferView& normBufferView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::Buffer& normBuffer = model.buffers[normBufferView.buffer];
                
                normals = reinterpret_cast<const float*>(
                    &normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
                normStride = normAccessor.ByteStride(normBufferView) / sizeof(float);
                if (normStride == 0) normStride = 3;
            }
            
            // Get indices
            if (primitive.indices < 0)
            {
                // Non-indexed geometry
                for (size_t i = 0; i < posAccessor.count; i += 3)
                {
                    Triangle tri;
                    
                    // Get vertex positions and transform to world space
                    for (int v = 0; v < 3; v++)
                    {
                        size_t idx = i + v;
                        glm::vec3 localPos(
                            positions[idx * posStride + 0],
                            positions[idx * posStride + 1],
                            positions[idx * posStride + 2]
                        );
                        glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                        
                        glm::vec3 normal(0, 1, 0);  // default up
                        if (normals)
                        {
                            normal = glm::vec3(
                                normals[idx * normStride + 0],
                                normals[idx * normStride + 1],
                                normals[idx * normStride + 2]
                            );
                            normal = glm::normalize(normalMatrix * normal);
                        }
                        
                        if (v == 0) { tri.v0 = worldPos; tri.n0 = normal; }
                        else if (v == 1) { tri.v1 = worldPos; tri.n1 = normal; }
                        else { tri.v2 = worldPos; tri.n2 = normal; }
                    }
                    
                    // Compute face normal if no vertex normals provided
                    if (!normals)
                    {
                        glm::vec3 faceNormal = glm::normalize(
                            glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                        tri.n0 = tri.n1 = tri.n2 = faceNormal;
                    }
                    
                    tri.materialId = materialId;
                    triangles.push_back(tri);
                }
            }
            else
            {
                // Indexed geometry
                const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                
                const void* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
                
                for (size_t i = 0; i < indexAccessor.count; i += 3)
                {
                    Triangle tri;
                    
                    // Get indices based on component type
                    uint32_t indices[3];
                    for (int v = 0; v < 3; v++)
                    {
                        size_t idx = i + v;
                        switch (indexAccessor.componentType)
                        {
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                                indices[v] = reinterpret_cast<const uint8_t*>(indexData)[idx];
                                break;
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                                indices[v] = reinterpret_cast<const uint16_t*>(indexData)[idx];
                                break;
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                                indices[v] = reinterpret_cast<const uint32_t*>(indexData)[idx];
                                break;
                            default:
                                indices[v] = 0;
                                break;
                        }
                    }
                    
                    // Get vertex positions and transform to world space
                    for (int v = 0; v < 3; v++)
                    {
                        uint32_t idx = indices[v];
                        glm::vec3 localPos(
                            positions[idx * posStride + 0],
                            positions[idx * posStride + 1],
                            positions[idx * posStride + 2]
                        );
                        glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                        
                        glm::vec3 normal(0, 1, 0);  // default up
                        if (normals)
                        {
                            normal = glm::vec3(
                                normals[idx * normStride + 0],
                                normals[idx * normStride + 1],
                                normals[idx * normStride + 2]
                            );
                            normal = glm::normalize(normalMatrix * normal);
                        }
                        
                        if (v == 0) { tri.v0 = worldPos; tri.n0 = normal; }
                        else if (v == 1) { tri.v1 = worldPos; tri.n1 = normal; }
                        else { tri.v2 = worldPos; tri.n2 = normal; }
                    }
                    
                    // Compute face normal if no vertex normals provided
                    if (!normals)
                    {
                        glm::vec3 faceNormal = glm::normalize(
                            glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                        tri.n0 = tri.n1 = tri.n2 = faceNormal;
                    }
                    
                    tri.materialId = materialId;
                    triangles.push_back(tri);
                }
            }
        }
    }
}

// ============================================================================
// BVH Construction Implementation
// ============================================================================

void Scene::buildBVH()
{
    // Initialize BVHInfo for all geoms (even non-meshes will have empty entries)
    bvhInfos.resize(geoms.size());
    
    int totalMeshTriangles = 0;
    int totalBVHNodes = 0;
    
    // First pass: count triangles in meshes and build BVH for each mesh
    for (size_t i = 0; i < geoms.size(); i++)
    {
        if (geoms[i].type == MESH && geoms[i].triangleCount > 0)
        {
            buildMeshBVH(static_cast<int>(i));
            totalMeshTriangles += geoms[i].triangleCount;
            totalBVHNodes += bvhInfos[i].nodeCount;
        }
        else
        {
            // Non-mesh geometry: empty BVH info
            bvhInfos[i].nodeOffset = 0;
            bvhInfos[i].nodeCount = 0;
            bvhInfos[i].triangleOffset = 0;
            bvhInfos[i].triangleCount = 0;
        }
    }
    
    cout << "BVH construction complete: " << totalBVHNodes << " nodes for " 
         << totalMeshTriangles << " triangles" << endl;
}

void Scene::buildMeshBVH(int geomIndex)
{
    Geom& geom = geoms[geomIndex];
    BVHInfo& info = bvhInfos[geomIndex];
    
    int triStart = geom.triangleStart;
    int triCount = geom.triangleCount;
    
    if (triCount == 0) return;
    
    // Store the starting offset for this mesh's BVH nodes
    info.nodeOffset = static_cast<int>(bvhNodes.size());
    info.triangleOffset = triStart;
    info.triangleCount = triCount;
    
    // Create local triangle indices (will be reordered during BVH construction)
    vector<int> triIndices(triCount);
    iota(triIndices.begin(), triIndices.end(), 0);  // 0, 1, 2, ...
    
    // Precompute triangle bounds and centroids
    vector<glm::vec3> triCentroids(triCount);
    vector<glm::vec3> triBoundsMin(triCount);
    vector<glm::vec3> triBoundsMax(triCount);
    
    for (int i = 0; i < triCount; i++)
    {
        const Triangle& tri = triangles[triStart + i];
        computeTriangleBounds(tri, triBoundsMin[i], triBoundsMax[i]);
        triCentroids[i] = computeTriangleCentroid(tri);
    }
    
    // Build stack for iterative construction
    stack<BVHBuildEntry> buildStack;
    
    // Push root node
    BVHBuildEntry rootEntry;
    rootEntry.parentIndex = -1;
    rootEntry.start = 0;
    rootEntry.end = triCount;
    rootEntry.isLeftChild = false;
    buildStack.push(rootEntry);
    
    // Reserve space for nodes (rough estimate: 2*N - 1 for N leaves)
    bvhNodes.reserve(bvhNodes.size() + 2 * triCount);
    
    while (!buildStack.empty())
    {
        BVHBuildEntry entry = buildStack.top();
        buildStack.pop();
        
        int start = entry.start;
        int end = entry.end;
        int numPrimitives = end - start;
        
        // Create new node
        int nodeIndex = static_cast<int>(bvhNodes.size());
        bvhNodes.emplace_back();
        BVHNode& node = bvhNodes.back();
        
        // Update parent's child pointer
        if (entry.parentIndex >= 0)
        {
            if (entry.isLeftChild)
                bvhNodes[entry.parentIndex].leftChild = nodeIndex;
            else
                bvhNodes[entry.parentIndex].rightChild = nodeIndex;
        }
        
        // Compute bounds for all primitives in this node
        node.boundsMin = glm::vec3(FLT_MAX);
        node.boundsMax = glm::vec3(-FLT_MAX);
        for (int i = start; i < end; i++)
        {
            int triIdx = triIndices[i];
            node.boundsMin = glm::min(node.boundsMin, triBoundsMin[triIdx]);
            node.boundsMax = glm::max(node.boundsMax, triBoundsMax[triIdx]);
        }
        
        // Check if this should be a leaf node
        if (numPrimitives <= MAX_LEAF_SIZE)
        {
            // Create leaf node
            node.leftChild = -1;  // Mark as leaf
            node.rightChild = -1;
            node.primitiveOffset = start;  // Will be adjusted later
            node.primitiveCount = numPrimitives;
            continue;
        }
        
        // Compute centroid bounds for SAH
        glm::vec3 centroidMin(FLT_MAX);
        glm::vec3 centroidMax(-FLT_MAX);
        for (int i = start; i < end; i++)
        {
            int triIdx = triIndices[i];
            centroidMin = glm::min(centroidMin, triCentroids[triIdx]);
            centroidMax = glm::max(centroidMax, triCentroids[triIdx]);
        }
        
        // Choose split axis (longest extent of centroid bounds)
        glm::vec3 centroidExtent = centroidMax - centroidMin;
        int splitAxis = 0;
        if (centroidExtent.y > centroidExtent.x) splitAxis = 1;
        if (centroidExtent.z > centroidExtent[splitAxis]) splitAxis = 2;
        
        // If centroids are coincident, create a leaf
        if (centroidExtent[splitAxis] < 1e-6f)
        {
            node.leftChild = -1;
            node.rightChild = -1;
            node.primitiveOffset = start;
            node.primitiveCount = numPrimitives;
            continue;
        }
        
        // SAH binning
        struct Bin {
            int count = 0;
            glm::vec3 boundsMin = glm::vec3(FLT_MAX);
            glm::vec3 boundsMax = glm::vec3(-FLT_MAX);
        };
        Bin bins[SAH_BINS];
        
        float binScale = SAH_BINS / centroidExtent[splitAxis];
        
        // Assign primitives to bins
        for (int i = start; i < end; i++)
        {
            int triIdx = triIndices[i];
            int binIdx = glm::min(SAH_BINS - 1,
                static_cast<int>((triCentroids[triIdx][splitAxis] - centroidMin[splitAxis]) * binScale));
            bins[binIdx].count++;
            bins[binIdx].boundsMin = glm::min(bins[binIdx].boundsMin, triBoundsMin[triIdx]);
            bins[binIdx].boundsMax = glm::max(bins[binIdx].boundsMax, triBoundsMax[triIdx]);
        }
        
        // Evaluate SAH cost for each split position
        float minCost = FLT_MAX;
        int minCostSplit = 0;
        
        // Precompute prefix costs (left side)
        float leftArea[SAH_BINS - 1];
        int leftCount[SAH_BINS - 1];
        glm::vec3 leftBoundsMin = glm::vec3(FLT_MAX);
        glm::vec3 leftBoundsMax = glm::vec3(-FLT_MAX);
        int leftN = 0;
        
        for (int i = 0; i < SAH_BINS - 1; i++)
        {
            leftBoundsMin = glm::min(leftBoundsMin, bins[i].boundsMin);
            leftBoundsMax = glm::max(leftBoundsMax, bins[i].boundsMax);
            leftN += bins[i].count;
            leftArea[i] = surfaceArea(leftBoundsMin, leftBoundsMax);
            leftCount[i] = leftN;
        }
        
        // Compute costs sweeping from right
        glm::vec3 rightBoundsMin = glm::vec3(FLT_MAX);
        glm::vec3 rightBoundsMax = glm::vec3(-FLT_MAX);
        int rightN = 0;
        
        for (int i = SAH_BINS - 1; i > 0; i--)
        {
            rightBoundsMin = glm::min(rightBoundsMin, bins[i].boundsMin);
            rightBoundsMax = glm::max(rightBoundsMax, bins[i].boundsMax);
            rightN += bins[i].count;
            
            float rightArea = surfaceArea(rightBoundsMin, rightBoundsMax);
            float cost = TRAVERSAL_COST + INTERSECTION_COST * 
                (leftCount[i-1] * leftArea[i-1] + rightN * rightArea) / surfaceArea(node.boundsMin, node.boundsMax);
            
            if (cost < minCost)
            {
                minCost = cost;
                minCostSplit = i;
            }
        }
        
        // Check if splitting is better than creating a leaf
        float leafCost = INTERSECTION_COST * numPrimitives;
        if (minCost >= leafCost && numPrimitives <= MAX_LEAF_SIZE * 2)
        {
            // Create leaf instead
            node.leftChild = -1;
            node.rightChild = -1;
            node.primitiveOffset = start;
            node.primitiveCount = numPrimitives;
            continue;
        }
        
        // Partition primitives based on SAH split
        float splitPos = centroidMin[splitAxis] + minCostSplit * centroidExtent[splitAxis] / SAH_BINS;
        
        auto midIter = partition(triIndices.begin() + start, triIndices.begin() + end,
            [&](int triIdx) {
                return triCentroids[triIdx][splitAxis] < splitPos;
            });
        int mid = static_cast<int>(midIter - triIndices.begin());
        
        // Ensure we don't create empty children
        if (mid == start || mid == end)
        {
            mid = (start + end) / 2;
            nth_element(triIndices.begin() + start, triIndices.begin() + mid, triIndices.begin() + end,
                [&](int a, int b) {
                    return triCentroids[a][splitAxis] < triCentroids[b][splitAxis];
                });
        }
        
        // Initialize as interior node (children will be filled in when processed)
        node.leftChild = -1;  // Placeholder
        node.rightChild = -1;
        node.primitiveOffset = 0;
        node.primitiveCount = 0;
        
        // Push children onto stack (right first so left is processed first)
        BVHBuildEntry rightEntry;
        rightEntry.parentIndex = nodeIndex;
        rightEntry.start = mid;
        rightEntry.end = end;
        rightEntry.isLeftChild = false;
        buildStack.push(rightEntry);
        
        BVHBuildEntry leftEntry;
        leftEntry.parentIndex = nodeIndex;
        leftEntry.start = start;
        leftEntry.end = mid;
        leftEntry.isLeftChild = true;
        buildStack.push(leftEntry);
    }
    
    // Reorder triangles according to BVH layout and fix primitive offsets
    vector<Triangle> reorderedTris(triCount);
    vector<int> newTriIndices(triCount);
    
    // First, create the reordered triangle list based on triIndices
    for (int i = 0; i < triCount; i++)
    {
        reorderedTris[i] = triangles[triStart + triIndices[i]];
    }
    
    // Copy back to original triangle array
    for (int i = 0; i < triCount; i++)
    {
        triangles[triStart + i] = reorderedTris[i];
    }
    
    // Update node count
    info.nodeCount = static_cast<int>(bvhNodes.size()) - info.nodeOffset;
    
    // Update geom's bounding box from root node
    if (info.nodeCount > 0)
    {
        const BVHNode& root = bvhNodes[info.nodeOffset];
        geom.boundingBoxMin = root.boundsMin;
        geom.boundingBoxMax = root.boundsMax;
    }
    
    cout << "  Built BVH for mesh " << geomIndex << ": " << info.nodeCount 
         << " nodes, " << triCount << " triangles" << endl;
}
