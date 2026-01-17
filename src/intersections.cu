#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Möller-Trumbore algorithm
    const float EPSILON = 1e-7f;
    
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    
    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);
    
    // Ray is parallel to triangle
    if (a > -EPSILON && a < EPSILON)
    {
        return -1.0f;
    }
    
    float f = 1.0f / a;
    glm::vec3 s = r.origin - tri.v0;
    float u = f * glm::dot(s, h);
    
    // Intersection point is outside triangle
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }
    
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, q);
    
    // Intersection point is outside triangle
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1.0f;
    }
    
    // Compute t to find intersection point
    float t = f * glm::dot(edge2, q);
    
    if (t > EPSILON)
    {
        intersectionPoint = r.origin + t * r.direction;
        
        // Interpolate normal using barycentric coordinates
        float w = 1.0f - u - v;
        normal = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
        
        // Determine if ray hit front or back face
        outside = (a > 0.0f);
        if (!outside)
        {
            normal = -normal;
        }
        
        return t;
    }
    
    return -1.0f;
}

__host__ __device__ bool aabbIntersectionTest(
    const glm::vec3& boxMin,
    const glm::vec3& boxMax,
    const Ray& r,
    float tMin,
    float tMax)
{
    // Optimized slab method for AABB intersection
    glm::vec3 invDir = 1.0f / r.direction;
    
    glm::vec3 t0 = (boxMin - r.origin) * invDir;
    glm::vec3 t1 = (boxMax - r.origin) * invDir;
    
    glm::vec3 tSmall = glm::min(t0, t1);
    glm::vec3 tBig = glm::max(t0, t1);
    
    tMin = glm::max(tMin, glm::max(tSmall.x, glm::max(tSmall.y, tSmall.z)));
    tMax = glm::min(tMax, glm::min(tBig.x, glm::min(tBig.y, tBig.z)));
    
    return tMin <= tMax;
}

__host__ __device__ bool aabbIntersectionTestWithDist(
    const glm::vec3& boxMin,
    const glm::vec3& boxMax,
    const Ray& r,
    const glm::vec3& invDir,
    float tMin,
    float tMax,
    float& tEntry)
{
    // Optimized slab method for AABB intersection with entry distance
    glm::vec3 t0 = (boxMin - r.origin) * invDir;
    glm::vec3 t1 = (boxMax - r.origin) * invDir;
    
    glm::vec3 tSmall = glm::min(t0, t1);
    glm::vec3 tBig = glm::max(t0, t1);
    
    float tEnter = glm::max(tMin, glm::max(tSmall.x, glm::max(tSmall.y, tSmall.z)));
    float tExit = glm::min(tMax, glm::min(tBig.x, glm::min(tBig.y, tBig.z)));
    
    tEntry = tEnter;
    return tEnter <= tExit;
}

__device__ bool bvhIntersect(
    const Ray& ray,
    const BVHNode* bvhNodes,
    int nodeOffset,
    const Triangle* triangles,
    int triangleOffset,
    float currentT,
    float& hitT,
    glm::vec3& hitNormal,
    int& hitMaterialId,
    bool& hitOutside)
{
    // Precompute inverse direction for faster AABB tests
    glm::vec3 invDir = 1.0f / ray.direction;
    
    // Stack for iterative traversal
    int stack[BVH_MAX_STACK_DEPTH];
    int stackPtr = 0;
    
    // Start with root node
    stack[stackPtr++] = nodeOffset;
    
    bool hit = false;
    float closestT = currentT;
    
    while (stackPtr > 0)
    {
        // Pop node from stack
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = bvhNodes[nodeIdx];
        
        // Check if ray intersects this node's bounding box
        float tEntry;
        if (!aabbIntersectionTestWithDist(node.boundsMin, node.boundsMax, 
                                          ray, invDir, 0.0f, closestT, tEntry))
        {
            continue;
        }
        
        if (node.isLeaf())
        {
            // Leaf node: test all triangles
            for (int i = 0; i < node.primitiveCount; i++)
            {
                int triIdx = triangleOffset + node.primitiveOffset + i;
                const Triangle& tri = triangles[triIdx];
                
                glm::vec3 tempIntersect, tempNormal;
                bool tempOutside;
                float t = triangleIntersectionTest(tri, ray, tempIntersect, tempNormal, tempOutside);
                
                if (t > 0.0f && t < closestT)
                {
                    closestT = t;
                    hitT = t;
                    hitNormal = tempNormal;
                    hitMaterialId = tri.materialId;
                    hitOutside = tempOutside;
                    hit = true;
                }
            }
        }
        else
        {
            // Interior node: push children onto stack
            // We want to traverse closer child first, so push farther child first
            const BVHNode& leftChild = bvhNodes[node.leftChild];
            const BVHNode& rightChild = bvhNodes[node.rightChild];
            
            float tLeftEntry, tRightEntry;
            bool hitLeft = aabbIntersectionTestWithDist(leftChild.boundsMin, leftChild.boundsMax,
                                                        ray, invDir, 0.0f, closestT, tLeftEntry);
            bool hitRight = aabbIntersectionTestWithDist(rightChild.boundsMin, rightChild.boundsMax,
                                                         ray, invDir, 0.0f, closestT, tRightEntry);
            
            // Push children in order: farther one first (so closer one is popped first)
            if (hitLeft && hitRight)
            {
                if (tLeftEntry < tRightEntry)
                {
                    // Left is closer, push right first
                    if (stackPtr < BVH_MAX_STACK_DEPTH) stack[stackPtr++] = node.rightChild;
                    if (stackPtr < BVH_MAX_STACK_DEPTH) stack[stackPtr++] = node.leftChild;
                }
                else
                {
                    // Right is closer, push left first
                    if (stackPtr < BVH_MAX_STACK_DEPTH) stack[stackPtr++] = node.leftChild;
                    if (stackPtr < BVH_MAX_STACK_DEPTH) stack[stackPtr++] = node.rightChild;
                }
            }
            else if (hitLeft)
            {
                if (stackPtr < BVH_MAX_STACK_DEPTH) stack[stackPtr++] = node.leftChild;
            }
            else if (hitRight)
            {
                if (stackPtr < BVH_MAX_STACK_DEPTH) stack[stackPtr++] = node.rightChild;
            }
        }
    }
    
    return hit;
}
